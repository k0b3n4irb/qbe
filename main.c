#include "all.h"
#include "config.h"
#include <ctype.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* MSYS2 / MinGW does not ship <execinfo.h>; backtrace() is glibc + Apple
 * libsystem only. Compile out the diagnostic handler on platforms that
 * lack it — Windows crashes are diagnosed via msys2_cproc_diagnostic.yml
 * + the retry loop in opensnes_build.yml instead. */
#if __has_include(<execinfo.h>)
#include <execinfo.h>
#include <unistd.h>
#define HAS_BACKTRACE 1
#endif

Target T;

#ifdef HAS_BACKTRACE
/* Diagnostic signal handler: when QBE crashes on a host where reproducing
 * the failure locally is hard (macOS arm64 strict alignment), we want a
 * stack trace in CI logs instead of a bare "Bus error 10". Pair the
 * addresses with `atos` (macOS) / `addr2line` (Linux) against the build
 * artefact to resolve to source. */
static void
crash_handler(int sig)
{
	void *frames[32];
	int n;
	const char *name =
		sig == SIGBUS  ? "SIGBUS"  :
		sig == SIGSEGV ? "SIGSEGV" :
		sig == SIGABRT ? "SIGABRT" :
		"signal";
	fprintf(stderr, "\n=== qbe: caught %s (signal %d) ===\n", name, sig);
	n = backtrace(frames, 32);
	backtrace_symbols_fd(frames, n, STDERR_FILENO);
	fputs("=== end backtrace ===\n", stderr);
	signal(sig, SIG_DFL);
	raise(sig);
}
#endif

/* OpenSNES function inlining: 2-pass parse architecture.
 *
 * To honour C99 inline semantics (inline definition is not a standalone
 * external definition) while still letting the inline pass see every
 * body during its source TU, we split the per-function pipeline at the
 * `inline_check` boundary:
 *
 *   - Pass 1 (in the parse callback `func`): SSA cleanup + `inline_record`.
 *     The Fn is then COLLECTED into the linked list below and `freeall()`
 *     is NOT called, so its pool-allocated IR survives until pass 2.
 *
 *   - Pass 1.b (after `parse()` returns): a module-wide loop runs
 *     `inline_check` on every collected Fn. This populates direct/inlined/
 *     declined/indirect counters on every inline-marked callee, with full
 *     TU visibility.
 *
 *   - Pass 2 (still after `parse()`): a second loop runs the rest of the
 *     pipeline (gvn → isel → spill → rega) and emits each Fn, skipping
 *     fully-consumed inline-hinted functions (see `inline_fully_consumed`).
 *
 * Memory cost is O(TU size): the per-function `alloc()` pool keeps growing
 * during parse instead of being recycled between functions. For typical
 * SNES SDK TUs (≤ a few dozen functions, < 1 MB total IR), this is
 * negligible. The previous design used `open_memstream` to buffer each
 * function's asm output; that was a POSIX-2008 dependency unavailable on
 * Microsoft UCRT, which is why the design moved to in-place 2-pass. */
typedef struct CollectedFn CollectedFn;
struct CollectedFn {
    Fn *fn;
    CollectedFn *next;
};
static CollectedFn *collected_head;
static CollectedFn *collected_tail;

char debug['Z'+1] = {
	['P'] = 0, /* parsing */
	['M'] = 0, /* memory optimization */
	['N'] = 0, /* ssa construction */
	['C'] = 0, /* copy elimination */
	['F'] = 0, /* constant folding */
	['A'] = 0, /* abi lowering */
	['I'] = 0, /* instruction selection */
	['L'] = 0, /* liveness */
	['S'] = 0, /* spilling */
	['R'] = 0, /* reg. allocation */
};

extern Target T_amd64_sysv;
extern Target T_amd64_apple;
extern Target T_arm64;
extern Target T_arm64_apple;
extern Target T_rv64;
extern Target T_w65816;

static Target *tlist[] = {
	&T_amd64_sysv,
	&T_amd64_apple,
	&T_arm64,
	&T_arm64_apple,
	&T_rv64,
	&T_w65816,
	0
};
static FILE *outf;
static int dbg;

static void
data(Dat *d)
{
	if (dbg)
		return;
	/* OpenSNES function inlining: data items that reference a symbol
	 * (e.g., `static void (*fp)(void) = foo;` -> `data $fp = { l $foo }`)
	 * count as indirect references to that symbol. Used by the inline
	 * pass to decide whether the symbol's standalone body must still
	 * be emitted.
	 *
	 * IMPORTANT — parsedat (parse.c) declares the local `Dat d` without
	 * initialising it before the DStart callback fires, so `d->isref`
	 * and `d->u.ref.name` carry stack garbage at that point. On Linux
	 * the bytes happen to be zero often enough that the test below
	 * fails; on macOS arm64 the stack is occasionally non-zero and the
	 * garbage pointer survives both checks, dereferencing into invalid
	 * memory and producing the SIGBUS hunted in chantier A6.11. Gate
	 * the lookup on a payload-bearing data type so we never trust the
	 * isref byte during the DStart/DEnd bookends. */
	if (d->type != DStart && d->type != DEnd
	&&  d->isref && d->u.ref.name)
		inline_record_dat_ref(d->u.ref.name);
	emitdat(d, outf);
	if (d->type == DEnd) {
		fputs("/* end data */\n\n", outf);
		/* NOTE: no freeall() here — collected functions need to survive
		 * the rest of the parse. Pool is reclaimed in one shot by the
		 * post-parse `emit_collected` path. */
	}
}

/* OpenSNES function inlining: pass 2 — the rest of the per-function
 * pipeline (gvn → simpl → isel → rega) plus the asm emit. Split out of
 * `func()` so the post-parse `emit_collected` loop can run it on every
 * function AFTER the module-wide `inline_check` phase has finalised
 * consumption tallies. */
static void
finalize_and_emit(Fn *fn, FILE *out)
{
	uint n;

	gvn(fn);
	fillcfg(fn);
	filluse(fn);
	filldom(fn);
	gcm(fn);
	filluse(fn);
	ssacheck(fn);
	T.abi1(fn);
	simpl(fn);
	fillcfg(fn);
	filluse(fn);
	T.isel(fn);
	fillcfg(fn);
	filllive(fn);
	fillloop(fn);
	fillcost(fn);
	if (!T.skiprega) {
		spill(fn);
		rega(fn);
		fillcfg(fn);
		simpljmp(fn);
	}
	/* When skiprega is set (w65816), phi nodes are preserved and
	 * lowered to copies by the emit code (emitphimoves).
	 * We skip simpljmp since it requires phi nodes to be gone. */
	fillcfg(fn);
	assert(fn->rpo[0] == fn->start);
	for (n=0;; n++)
		if (n == fn->nblk-1) {
			fn->rpo[n]->link = 0;
			break;
		} else
			fn->rpo[n]->link = fn->rpo[n+1];
	T.emitfn(fn, out);
	fprintf(out, "/* end function %s */\n\n", fn->name);
}

static void
func(Fn *fn)
{
	CollectedFn *c;

	if (dbg)
		fprintf(stderr, "**** Function %s ****", fn->name);
	if (debug['P']) {
		fprintf(stderr, "\n> After parsing:\n");
		printfn(fn, stderr);
	}
	T.abi0(fn);
	fillcfg(fn);
	filluse(fn);
	promote(fn);
	filluse(fn);
	ssa(fn);
	filluse(fn);
	ssacheck(fn);
	fillalias(fn);
	loadopt(fn);
	filluse(fn);
	fillalias(fn);
	coalesce(fn);
	filluse(fn);
	filldom(fn);
	ssacheck(fn);
	/* OpenSNES function inlining: hook after the SSA cleanup pipeline
	 * (promote/ssa/loadopt/coalesce). At this point allocas are gone
	 * and copies are dense; the heuristic measures canonical IR. */
	inline_record(fn);

	if (dbg) {
		/* Debug mode: run the rest of the pipeline immediately so the
		 * per-pass `debug[...]` traces fire in their natural order. No
		 * asm output is expected. */
		inline_check(fn);
		gvn(fn);
		fillcfg(fn); filluse(fn); filldom(fn); gcm(fn);
		filluse(fn); ssacheck(fn);
		T.abi1(fn); simpl(fn);
		fillcfg(fn); filluse(fn);
		T.isel(fn);
		fillcfg(fn); filllive(fn); fillloop(fn); fillcost(fn);
		if (!T.skiprega) {
			spill(fn); rega(fn);
			fillcfg(fn); simpljmp(fn);
		}
		fillcfg(fn);
		fprintf(stderr, "\n");
		freeall();
		return;
	}

	/* Normal mode: collect the Fn for the post-parse 2-pass emit. The
	 * Fn's pool memory must survive until pass 2 runs, so NO freeall()
	 * happens here. */
	c = emalloc(sizeof *c);
	c->fn = fn;
	c->next = NULL;
	if (collected_tail) collected_tail->next = c;
	else                collected_head = c;
	collected_tail = c;
}

/* OpenSNES function inlining: post-parse 2-pass emit.
 *
 * Pass 1.b: run `inline_check` on every collected Fn so each callee's
 *           consumption counters reflect every TU caller.
 * Pass 2:   run the rest of the per-function pipeline and emit asm,
 *           skipping inline-hinted functions that are fully consumed
 *           (see `inline_fully_consumed`).
 *
 * One module-wide `freeall()` at the end reclaims the pool that grew
 * across every parsed function. */
static void
emit_collected(FILE *out)
{
	CollectedFn *c;

	/* Pass 1.b: module-wide inline_check phase */
	for (c = collected_head; c; c = c->next)
		inline_check(c->fn);

	/* Pass 2: finalize + emit, skipping fully-consumed inline bodies */
	for (c = collected_head; c; c = c->next) {
		Fn *fn = c->fn;
		if (fn->lnk.inline_hint && inline_fully_consumed(fn->name))
			continue;
		finalize_and_emit(fn, out);
	}

	freeall();
	collected_head = collected_tail = NULL;
}

static void
dbgfile(char *fn)
{
	emitdbgfile(fn, outf);
}

int
main(int ac, char *av[])
{
	Target **t;
	FILE *inf, *hf;
	char *f, *sep;
	int c;

#ifdef HAS_BACKTRACE
	signal(SIGBUS,  crash_handler);
	signal(SIGSEGV, crash_handler);
#endif

	T = Deftgt;
	outf = stdout;
	while ((c = getopt(ac, av, "hd:o:t:")) != -1)
		switch (c) {
		case 'd':
			for (; *optarg; optarg++)
				if (isalpha(*optarg)) {
					debug[toupper(*optarg)] = 1;
					dbg = 1;
				}
			break;
		case 'o':
			if (strcmp(optarg, "-") != 0) {
				outf = fopen(optarg, "w");
				if (!outf) {
					fprintf(stderr, "cannot open '%s'\n", optarg);
					exit(1);
				}
			}
			break;
		case 't':
			if (strcmp(optarg, "?") == 0) {
				puts(T.name);
				exit(0);
			}
			for (t=tlist;; t++) {
				if (!*t) {
					fprintf(stderr, "unknown target '%s'\n", optarg);
					exit(1);
				}
				if (strcmp(optarg, (*t)->name) == 0) {
					T = **t;
					break;
				}
			}
			break;
		case 'h':
		default:
			hf = c != 'h' ? stderr : stdout;
			fprintf(hf, "%s [OPTIONS] {file.ssa, -}\n", av[0]);
			fprintf(hf, "\t%-11s prints this help\n", "-h");
			fprintf(hf, "\t%-11s output to file\n", "-o file");
			fprintf(hf, "\t%-11s generate for a target among:\n", "-t <target>");
			fprintf(hf, "\t%-11s ", "");
			for (t=tlist, sep=""; *t; t++, sep=", ") {
				fprintf(hf, "%s%s", sep, (*t)->name);
				if (*t == &Deftgt)
					fputs(" (default)", hf);
			}
			fprintf(hf, "\n");
			fprintf(hf, "\t%-11s dump debug information\n", "-d <flags>");
			exit(c != 'h');
		}

	do {
		f = av[optind];
		if (!f || strcmp(f, "-") == 0) {
			inf = stdin;
			f = "-";
		} else {
			inf = fopen(f, "r");
			if (!inf) {
				fprintf(stderr, "cannot open '%s'\n", f);
				exit(1);
			}
		}
		parse(inf, f, dbgfile, data, func);
		fclose(inf);
	} while (++optind < ac);

	if (!dbg) {
		emit_collected(outf);
		T.emitfin(outf);
	}

	exit(0);
}
