#define _POSIX_C_SOURCE 200809L  /* open_memstream */
#include "all.h"
#include "config.h"
#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Target T;

/* OpenSNES function inlining: pending function emissions.
 *
 * To honour C99 inline semantics (inline definition is not a standalone
 * external definition) while still letting the inline pass see the
 * body during its source TU, we defer asm emission of each function to
 * a memory buffer. After all functions are parsed and processed, we
 * decide which buffers to flush to outf:
 *
 *   - non-inline-hint fns: always flush
 *   - inline-hint fns with at least one inlined caller AND no decline:
 *     suppress (fully consumed)
 *   - inline-hint fns with any decline or no callers: flush (safe default) */
typedef struct PendingFn PendingFn;
struct PendingFn {
    char name[NString];
    char *buf;
    size_t len;
    int inline_hint;
    PendingFn *next;
};
static PendingFn *pending_head;
static PendingFn *pending_tail;

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
	 * be emitted. */
	if (d->isref && d->u.ref.name)
		inline_record_dat_ref(d->u.ref.name);
	emitdat(d, outf);
	if (d->type == DEnd) {
		fputs("/* end data */\n\n", outf);
		freeall();
	}
}

static void
func(Fn *fn)
{
	uint n;

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
	inline_check(fn);
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
	if (!dbg) {
		/* OpenSNES function inlining: emit to a per-fn memory buffer so
		 * we can suppress fully-inlined functions at the final flush. */
		PendingFn *p = emalloc(sizeof *p);
		strncpy(p->name, fn->name, NString-1);
		p->name[NString-1] = 0;
		p->inline_hint = fn->lnk.inline_hint;
		p->buf = NULL;
		p->len = 0;
		p->next = NULL;
		FILE *ms = open_memstream(&p->buf, &p->len);
		if (!ms) {
			fprintf(stderr, "open_memstream failed\n");
			abort();
		}
		T.emitfn(fn, ms);
		fprintf(ms, "/* end function %s */\n\n", fn->name);
		fclose(ms);
		if (pending_tail) pending_tail->next = p;
		else              pending_head = p;
		pending_tail = p;
	} else
		fprintf(stderr, "\n");
	freeall();
}

/* OpenSNES function inlining: flush deferred function buffers to outf,
 * skipping inline-hinted functions that were fully consumed by the
 * inline pass (every direct caller in this TU inlined, no indirect
 * references, at least one caller existed). */
static void
flush_pending(FILE *out)
{
    PendingFn *p, *next;
    for (p = pending_head; p; p = next) {
        next = p->next;
        if (!(p->inline_hint && inline_fully_consumed(p->name)))
            fwrite(p->buf, 1, p->len, out);
        free(p->buf);
        free(p);
    }
    pending_head = pending_tail = NULL;
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
		flush_pending(outf);
		T.emitfin(outf);
	}

	exit(0);
}
