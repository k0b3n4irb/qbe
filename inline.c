/*
 * OpenSNES function inlining — step 2.a (eligibility check + tracing).
 *
 * Architecture (per .claude/notes/chantiers/function_inlining_audit.md):
 *
 *   - As each function is parsed and BEFORE its lowering pipeline runs,
 *     main.c calls inline_record(fn). We scan the IR and store a tiny
 *     summary record in a PHeap-backed linked list. The summary survives
 *     freeall() between functions; the IR itself does NOT (we don't need
 *     it yet — splicing is step 2.b).
 *
 *   - When a CALLER's pipeline runs, inline_check(fn) walks each Ocall
 *     and looks up the target by name. If the target is in the table
 *     AND eligibility holds (single block, ≤ INLINE_MAX_INSTR, no nested
 *     calls, no alloc), CC_TRACE_INLINE=1 logs "WOULD INLINE". Step 2.b
 *     replaces the trace with actual splicing.
 *
 * Heuristic gate (conservative, user-chosen for v1):
 *
 *   inline_hint     — function had `inline` keyword in C source
 *   blk_count == 1  — single basic block (no control flow to clone)
 *   has_calls == 0  — no nested calls (avoid cascading inline + recursion)
 *   has_allocs == 0 — no stack allocations (avoid alloc-bookkeeping)
 *   ins_count ≤ 8   — small body (override via CC_INLINE_MAX_INSTR=N)
 *
 * Order constraint: a callee must be RECORDED before its first call site
 * is CHECKED. cproc emits functions in source order, so forward references
 * fall through to a jsl (the table lookup misses); that is acceptable.
 */

#include "all.h"
#include <stdlib.h>

#define INLINE_MAX_INSTR 16  /* bumped from 8 in wave 4 retrofit (2026-05-13);
                              * lets ~10-15 IR-instr lib helpers (textSetPos,
                              * fixCos, colorMathEnable) inline. CC_INLINE_MAX_INSTR=N
                              * overrides per build. */

typedef struct InlRec InlRec;
typedef struct InlBody InlBody;

/* Saved body of an inline-eligible function. PHeap-backed so it survives
 * freeall() between functions. Cons are deep-copied; sym.id values are
 * interned-string handles that stay valid across functions. */
struct InlBody {
    Ref *params;          /* param Refs (RTmp) in declaration order */
    int  nparams;
    Ins *ins;             /* flat array of non-Opar body instructions */
    int  nins;
    Con *con;             /* copy of callee's fn->con */
    int  ncon;
    Ref  retval;          /* value passed to terminal Jret* */
    int  cls;             /* return class (Kw, Kl, etc.) */
    int  max_tmp;         /* highest RTmp val in ins/retval/params */
};

struct InlRec {
    char name[NString];
    char hint;            /* function had `inline` keyword */
    char eligible;        /* 1 if passes heuristic */
    char has_calls;
    char has_allocs;
    char has_phis;        /* any phi node in any block (= merges) */
    char has_cond;        /* any conditional jump (non-linear flow) */
    int  ins_count;       /* total instructions across all blocks */
    int  blk_count;       /* number of basic blocks */
    /* Consumption tracking (chantier extension: cproc emit suppression).
     * For an inline-hint function to be "fully consumed" in this TU and
     * thus safe to skip at final emit:
     *   - at least one direct caller existed in this TU (n_direct > 0)
     *   - every direct call site was successfully inlined (n_declined == 0)
     *   - no indirect (function-pointer) reference appeared (n_indirect == 0)
     * Otherwise we emit the standalone body to satisfy linker. */
    int  n_direct;        /* direct Ocall references to this fn seen */
    int  n_inlined;       /* of which were successfully spliced */
    int  n_declined;      /* of which were declined (couldn't inline) */
    int  n_indirect;      /* indirect references (function-ptr use) */
    InlBody *body;        /* deep-clone of body, or NULL if not eligible */
    /* Whole-function inlining of a static function with one call site
     * (2026-10-10, OpenSNES issue #166, pattern 9): */
    Fn  *fn;              /* the function itself; alive until the module is emitted */
    char is_static;       /* not exported */
    char checked;         /* inline_check has run on it: its body is final */
    char busy;            /* inline_check is running on it (a cycle: decline) */
    char absorbed;        /* its one call site took its body: do not emit it */
    int  sites;           /* direct calls to it in the module */
    InlRec *next;
};

static InlRec *inl_head;

/* Forward declarations. */
static InlBody *save_body(Fn *fn, int cls);
static int      splice_at(Fn *caller, Blk *blk, int call_idx, InlBody *body);
static InlRec  *inline_lookup(char *name);

static int
inline_max_instr(void)
{
    char *e;
    int v;

    e = getenv("CC_INLINE_MAX_INSTR");
    if (!e || !*e) return INLINE_MAX_INSTR;
    v = atoi(e);
    if (v <= 0) return INLINE_MAX_INSTR;
    return v;
}

void
inline_record(Fn *fn)
{
    InlRec *r;
    Blk *b;
    Ins *i;
    int max;

    r = inline_lookup(fn->name);
    if (r) {
        /* Already recorded (a TU may emit a function's body and then
         * have a redeclaration somewhere). Don't overwrite — keep the
         * original record's consumption counters intact. */
        return;
    }
    r = emalloc(sizeof *r);
    strncpy(r->name, fn->name, NString-1);
    r->name[NString-1] = 0;
    r->hint = fn->lnk.inline_hint;
    r->ins_count = 0;
    r->blk_count = 0;
    r->has_calls = 0;
    r->has_allocs = 0;
    r->has_phis = 0;
    r->has_cond = 0;
    r->body = NULL;
    r->n_direct = 0;
    r->n_inlined = 0;
    r->n_declined = 0;
    r->n_indirect = 0;
    r->fn = fn;
    r->is_static = !fn->lnk.export;
    r->checked = 0;
    r->busy = 0;
    r->absorbed = 0;
    r->sites = 0;

    for (b = fn->start; b; b = b->link) {
        r->blk_count++;
        if (b->phi) r->has_phis = 1;
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            /* Debug metadata (dbgloc) is zero-cost and must NOT affect inlinability,
             * else a debug build inlines differently from release — and a C99
             * `inline` helper sitting on the size threshold (e.g. colorMathEnable)
             * would spill a standalone external definition, duplicating the symbol
             * the canonical .c already emits (link error). */
            if (i->op == Odbgloc) continue;
            r->ins_count++;
            if (i->op == Ocall) r->has_calls = 1;
            if (isalloc(i->op)) r->has_allocs = 1;
        }
        /* Non-linear flow: anything other than Jjmp / Jret* */
        if (b->jmp.type != Jjmp && !isret(b->jmp.type))
            r->has_cond = 1;
    }

    max = inline_max_instr();
    /* Linear-flow heuristic (v1): allow multiple blocks if they form a
     * straight-line sequence (only unconditional jmp + final ret). This
     * accommodates cproc's `@start; jmp @body; @body; ...; ret` pattern
     * that survives SSA cleanup as 2 blocks for even trivial helpers. */
    r->eligible = r->hint
        && !r->has_phis
        && !r->has_cond
        && !r->has_calls
        && !r->has_allocs
        && r->ins_count <= max;

    /* Save body for splice if eligible. */
    if (r->eligible)
        r->body = save_body(fn, fn->retty);

    if (getenv("CC_TRACE_INLINE")) {
        char *trace = getenv("CC_TRACE_INLINE");
        fprintf(stderr,
            "[inline] record %s: hint=%d blk=%d ins=%d phis=%d cond=%d "
            "calls=%d allocs=%d (max=%d) -> %s\n",
            r->name, r->hint, r->blk_count, r->ins_count,
            r->has_phis, r->has_cond, r->has_calls, r->has_allocs, max,
            r->eligible ? "ELIGIBLE" : "rejected");
        if (r->eligible && atoi(trace) >= 2) {
            fprintf(stderr, "[inline] body of %s:\n", r->name);
            printfn(fn, stderr);
        }
    }

    r->next = inl_head;
    inl_head = r;
}

static InlRec *
inline_lookup(char *name)
{
    InlRec *r;
    for (r = inl_head; r; r = r->next)
        if (strcmp(r->name, name) == 0) return r;
    return NULL;
}

/* Walk a Ref and update max_tmp if it's an RTmp with a higher index. */
static void
track_ref(Ref r, int *max)
{
    if (rtype(r) == RTmp && (int)r.val > *max)
        *max = r.val;
}

/* Deep-copy an eligible function's body into a PHeap-backed InlBody.
 * The body excludes Opar* (parameters are tracked separately); the
 * caller will splice them by mapping to its own Oarg* values. */
static InlBody *
save_body(Fn *fn, int cls)
{
    InlBody *b;
    Blk *blk;
    Ins *i;
    int nparams, n, j, idx;

    nparams = 0;
    n = 0;
    for (blk = fn->start; blk; blk = blk->link) {
        for (i = blk->ins; i < &blk->ins[blk->nins]; i++) {
            if (ispar(i->op)) nparams++;
            else              n++;
        }
    }

    b = emalloc(sizeof *b);
    b->nparams = 0;
    b->nins = 0;
    b->max_tmp = 0;
    b->cls = cls;
    b->retval = R;
    b->params = nparams ? emalloc(nparams * sizeof(Ref)) : NULL;
    b->ins = n ? emalloc(n * sizeof(Ins)) : NULL;
    b->ncon = fn->ncon;
    b->con = emalloc(fn->ncon * sizeof(Con));
    memcpy(b->con, fn->con, fn->ncon * sizeof(Con));

    for (blk = fn->start; blk; blk = blk->link) {
        for (i = blk->ins; i < &blk->ins[blk->nins]; i++) {
            if (ispar(i->op)) {
                idx = b->nparams++;
                b->params[idx] = i->to;
                track_ref(i->to, &b->max_tmp);
            } else {
                b->ins[b->nins] = *i;
                track_ref(i->to, &b->max_tmp);
                for (j = 0; j < 2; j++)
                    track_ref(i->arg[j], &b->max_tmp);
                b->nins++;
            }
        }
        if (isret(blk->jmp.type) && blk->jmp.type != Jret0) {
            b->retval = blk->jmp.arg;
            track_ref(b->retval, &b->max_tmp);
        }
    }

    return b;
}

/* Copy callee's con into caller's con array (using the canonical newcon
 * path, which dedupes if equivalent entry already exists), return ref. */
static Ref
remap_con(Fn *caller, InlBody *body, int callee_con_val)
{
    if (callee_con_val < 0 || callee_con_val >= body->ncon)
        return CON(callee_con_val);
    return newcon(&body->con[callee_con_val], caller);
}

/* Remap a Ref from callee numbering to caller numbering.
 * - RTmp matching a param: replaced by the corresponding caller arg value
 * - RTmp non-param: offset by tmp_offset
 * - RCon: copied into caller's con array
 * - Other (RSlot etc.): passed through. */
static Ref
remap_ref(Ref r, Fn *caller, InlBody *body, Ref *args, int tmp_offset)
{
    int j;
    if (req(r, R))
        return R;
    switch (rtype(r)) {
    case RTmp:
        for (j = 0; j < body->nparams; j++) {
            if (req(r, body->params[j]))
                return args[j];
        }
        return TMP(r.val + tmp_offset);
    case RCon:
        return remap_con(caller, body, r.val);
    default:
        return r;
    }
}

/* Try to inline `body` at the Ocall instruction `call_ins` within `blk`.
 * Returns 1 on success (caller IR was rewritten), 0 if inlining declined
 * (e.g. arg count mismatch — fall back to jsl). */
static int
splice_at(Fn *caller, Blk *blk, int call_idx, InlBody *body)
{
    Ins *old_ins, *ni, *new_ins;
    Ins call_ins, *iarg;
    int i, j, k, new_nins, new_cap, arg_start, tmp_offset;
    Ref args[16];  /* up to 16 params — well above realistic inline cases */
    Ref retval_remapped;
    Ins copy_ins;

    if (body->nparams > 16)
        return 0;

    /* Locate the preceding Oarg* (in source order, immediately before
     * the call). Iterate the nparams instructions before call_idx. */
    arg_start = call_idx - body->nparams;
    if (arg_start < 0)
        return 0;
    iarg = &blk->ins[arg_start];
    for (i = 0; i < body->nparams; i++) {
        if (!isarg(iarg[i].op))
            return 0;
        args[i] = iarg[i].arg[0];
    }

    call_ins = blk->ins[call_idx];
    /* Offset applied to remap callee non-param tmps to fresh caller tmps:
     *   callee TMP(T) (T >= Tmp0) -> caller TMP(T + tmp_offset)
     * with tmp_offset chosen so the remapped tmps are fresh in the caller. */
    tmp_offset = (int)caller->ntmp - Tmp0;

    /* Allocate fresh caller tmps to cover every callee tmp val up to
     * body->max_tmp. Some of those tmps correspond to params (mapped to
     * args at remap time, never actually used) but it's harmless to
     * allocate them — costs a few unused Tmp slots. */
    while ((int)caller->ntmp <= body->max_tmp + tmp_offset)
        newtmp("inl", body->cls, caller);

    /* Build the rewritten ins[] for this block:
     * - copy ins[0 .. arg_start-1] as-is
     * - emit remapped body ins
     * - emit Ocopy of retval into call_ins.to (if non-void)
     * - copy ins[call_idx+1 ..] as-is
     *
     * Important: blk->ins must remain a vec (allocated via vnew) so
     * downstream passes that call vgrow on it (simpl, mem, etc.)
     * see the expected magic-word header. */
    new_cap = blk->nins + body->nins + 1;
    new_ins = vnew(new_cap, sizeof(Ins), PFn);
    new_nins = 0;
    old_ins = blk->ins;

    for (i = 0; i < arg_start; i++)
        new_ins[new_nins++] = old_ins[i];

    for (j = 0; j < body->nins; j++) {
        ni = &new_ins[new_nins++];
        *ni = body->ins[j];
        ni->to = remap_ref(ni->to, caller, body, args, tmp_offset);
        ni->arg[0] = remap_ref(ni->arg[0], caller, body, args, tmp_offset);
        ni->arg[1] = remap_ref(ni->arg[1], caller, body, args, tmp_offset);
    }

    /* Wire the return value into the caller's call destination */
    if (!req(call_ins.to, R) && !req(body->retval, R)) {
        retval_remapped = remap_ref(body->retval, caller, body, args, tmp_offset);
        copy_ins = (Ins){.op = Ocopy, .cls = call_ins.cls,
                         .to = call_ins.to,
                         .arg = {retval_remapped, R},
                         .volat = 0};
        new_ins[new_nins++] = copy_ins;
    }

    for (k = call_idx + 1; k < (int)blk->nins; k++)
        new_ins[new_nins++] = old_ins[k];

    blk->ins = new_ins;
    blk->nins = new_nins;

    return 1;
}

/* ---------------------------------------------------------------------------
 * Whole-function inlining (2026-10-10; OpenSNES issue #166, pattern 9).
 *
 * A game's per-entity tick is a handful of small `static` functions called
 * once each from one loop. Each call pushes its arguments, enters, rebuilds
 * the index it was given and returns; and a function that calls another is
 * not a leaf, so it loses the direct-page frame. Measured on a real game:
 * the functions whose body is calls gained 4 to 8 % from everything done
 * inside a block, against 16 to 28 % for those whose body is array accesses.
 *
 * A function that is not exported, whose address is not taken and that has
 * exactly ONE call site in the module is that call site's code: its body
 * replaces the call and the function is not emitted. No size limit is
 * needed — the code exists once either way.
 *
 * A function called from several places is copied only where the source
 * asks for it: `static inline`, up to INLINE_MAX_BIG instructions of IR.
 * The first inliner above still takes the tiny single-block ones; this one
 * takes those it declined (control flow, calls inside). The standalone
 * function disappears when every call was inlined, as before.
 *
 * Unlike the first inliner above, which pastes a few instructions into a
 * block, this one handles control flow. The caller's block is cut at the
 * call; the callee's blocks are cloned between the two halves with fresh
 * temporaries; each parameter is the argument, extended as the parameter's
 * type says; each `ret` jumps to the second half, where a phi (or a copy,
 * for a single `ret`) receives the value. The clone is made from the
 * callee's IR as it stands, after its own calls were inlined: the module is
 * walked bottom-up, and a cycle is declined.
 *
 * Declined: a leaf of some size when its caller still calls something else
 * (see auto_inline_ok), a callee that still holds an alloc (a local whose address
 * escapes), a struct or variadic parameter, a caller that would pass the
 * back end's limit on temporaries. QBE_NO_AUTO_INLINE=1 turns it off.
 * ------------------------------------------------------------------------- */
#define AUTO_INLINE_TMP_ROOM 1800       /* under the back end's 2048 */

#define INLINE_MAX_BIG 160      /* IR instructions; CC_INLINE_MAX_BIG=N overrides */

static int inl_seq;

static int
inline_leaf_max(void)
{
    char *e = getenv("CC_INLINE_LEAF_MAX");

    return e ? atoi(e) : 16;
}

static int
inline_max_big(void)
{
    char *e = getenv("CC_INLINE_MAX_BIG");

    return e && atoi(e) > 0 ? atoi(e) : INLINE_MAX_BIG;
}

static Ref
clone_ref(Ref r, Fn *caller, Fn *callee, Ref *tmap)
{
    if (req(r, R))
        return R;
    switch (rtype(r)) {
    case RTmp:
        return r.val < Tmp0 ? r : tmap[r.val];
    case RCon:
        return newcon(&callee->con[r.val], caller);
    default:
        return r;
    }
}

static int
auto_inline_ok(Fn *caller, InlRec *rec)
{
    Fn *callee = rec->fn;
    Blk *b;
    Ins *i;
    int ccalls, calls;

    if (getenv("QBE_NO_AUTO_INLINE"))
        return 0;
    if (!callee || callee == caller || !rec->is_static)
        return 0;
    if (rec->busy || addrtaken(rec->name))
        return 0;
    /* one call site: always. Several: only where the source says `inline`,
     * and under a size (the body is then copied at each site). */
    if (rec->sites != 1
        && !(rec->hint && rec->ins_count <= inline_max_big()))
        return 0;
    if (callee->vararg || callee->dynalloc)
        return 0;
    if (caller->ntmp + callee->ntmp >= AUTO_INLINE_TMP_ROOM)
        return 0;
    for (b = callee->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            if (isalloc(i->op))
                return 0;
            if (i->op == Oparc || i->op == Opare || i->op == Oargc
                || i->op == Oarge || i->op == Oargv || i->op == Ovastart
                || i->op == Ovaarg)
                return 0;
            if (ispar(i->op) && b != callee->start)
                return 0;
        }
        if (b->jmp.type != Jjmp && b->jmp.type != Jjnz && b->jmp.type != Jhlt
            && !isret(b->jmp.type))
            return 0;
        if (b->jmp.type == Jretc || b->jmp.type == Jrets || b->jmp.type == Jretd)
            return 0;                   /* a struct or a float returned */
    }
    /* A leaf keeps its temporaries in the direct page; a function that
     * calls keeps them on the stack, a cycle dearer at each access and out
     * of reach of the peephole's rules on direct-page slots. Pouring a leaf
     * into a caller that still calls something else trades one call for
     * that, at every access: measured on a real game (2026-10-10), its
     * logic ran 2 % SLOWER with every leaf absorbed and 0.4 % faster with
     * the leaves left alone; in the corpus a 16 KB fill inlined into main
     * made mode7/extbg boot four frames later. So a leaf goes into such a
     * caller only when it is tiny (the call then costs more than the
     * body); into a caller whose every call can be absorbed, always — that
     * caller ends up a leaf itself. CC_INLINE_LEAF_MAX=N moves the size. */
    ccalls = 0;
    for (b = callee->start; b; b = b->link)
        for (i = b->ins; i < &b->ins[b->nins]; i++)
            if (i->op == Ocall)
                ccalls++;
    if (!ccalls && rec->ins_count > inline_leaf_max()) {
        /* does the caller keep a call once everything that can be
         * absorbed into it has been? (a guess: a candidate counted here
         * may still be declined; then a leaf was absorbed for little) */
        calls = 0;
        for (b = caller->start; b; b = b->link)
            for (i = b->ins; i < &b->ins[b->nins]; i++) {
                InlRec *o;
                Con *c;

                if (i->op != Ocall)
                    continue;
                o = 0;
                if (rtype(i->arg[0]) == RCon) {
                    c = &caller->con[i->arg[0].val];
                    if (c->type == CAddr)
                        o = inline_lookup(str(c->sym.id));
                }
                if (o && o->eligible)
                    continue;           /* the small inliner takes it */
                if (o && o->fn && o->fn != caller && o->is_static && !o->busy
                    && !addrtaken(o->name)
                    && (o->sites == 1
                        || (o->hint && o->ins_count <= inline_max_big())))
                    continue;
                calls++;
            }
        if (calls)
            return 0;
    }
    return 1;
}

/* Replace the call at blk->ins[call_idx] by the body of `callee`. Returns
 * the block that holds what followed the call (0: declined, nothing done). */
static Blk *
splice_fn(Fn *caller, Blk *blk, int call_idx, Fn *callee)
{
    Ins call_ins, *i, *ni, *head, *tail, *iarg;
    Blk *b, *c, *cont, *succ, **bmap, *last, *firstc;
    Phi *p, *np, *retphi;
    Ref *tmap, *retval;
    Blk **retblk;
    int npar, nargs, arg_start, nb, k, n, nret, nhead, ntail, seq;
    uint t, a;

    npar = 0;
    for (i = callee->start->ins; i < &callee->start->ins[callee->start->nins]; i++)
        if (ispar(i->op))
            npar++;
    nargs = 0;
    for (k = call_idx - 1; k >= 0 && isarg(blk->ins[k].op); k--)
        nargs++;
    if (nargs != npar)
        return 0;
    arg_start = call_idx - npar;
    iarg = &blk->ins[arg_start];
    call_ins = blk->ins[call_idx];
    seq = ++inl_seq;

    /* fresh temporaries, one per callee temporary, of the same class */
    tmap = emalloc(callee->ntmp * sizeof tmap[0]);
    for (t = Tmp0; t < (uint)callee->ntmp; t++)
        tmap[t] = newtmp("inl", callee->tmp[t].cls, caller);

    /* the callee's blocks, cloned */
    nb = 0;
    for (b = callee->start; b; b = b->link)
        b->visit = nb++;
    bmap = emalloc(nb * sizeof bmap[0]);
    for (b = callee->start; b; b = b->link) {
        c = newblk();
        snprintf(c->name, NString, "inl%d.%.40s", seq, b->name);
        bmap[b->visit] = c;
    }
    cont = newblk();
    snprintf(cont->name, NString, "inl%d.ret", seq);

    retval = emalloc(nb * sizeof retval[0]);
    retblk = emalloc(nb * sizeof retblk[0]);
    nret = 0;
    for (b = callee->start; b; b = b->link) {
        c = bmap[b->visit];
        /* phis */
        for (p = b->phi; p; p = p->link) {
            np = alloc(sizeof *np);
            np->to = clone_ref(p->to, caller, callee, tmap);
            np->cls = p->cls;
            np->narg = p->narg;
            np->arg = vnew(p->narg, sizeof np->arg[0], PFn);
            np->blk = vnew(p->narg, sizeof np->blk[0], PFn);
            for (a = 0; a < p->narg; a++) {
                np->arg[a] = clone_ref(p->arg[a], caller, callee, tmap);
                np->blk[a] = bmap[p->blk[a]->visit];
            }
            np->link = c->phi;
            c->phi = np;
        }
        /* instructions; a parameter becomes its argument, extended as the
         * parameter's type says (the caller's value is the full word) */
        c->ins = vnew(b->nins + 1, sizeof(Ins), PFn);
        n = 0;
        k = 0;
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            ni = &c->ins[n++];
            if (ispar(i->op)) {
                *ni = (Ins){ .cls = i->cls, .to = tmap[i->to.val],
                             .arg = { iarg[k].arg[0], R } };
                switch (i->op) {
                case Oparsb: ni->op = Oextsb; break;
                case Oparub: ni->op = Oextub; break;
                case Oparsh: ni->op = Oextsh; break;
                case Oparuh: ni->op = Oextuh; break;
                default:     ni->op = Ocopy;  break;
                }
                k++;
                continue;
            }
            *ni = *i;
            ni->to = clone_ref(i->to, caller, callee, tmap);
            ni->arg[0] = clone_ref(i->arg[0], caller, callee, tmap);
            ni->arg[1] = clone_ref(i->arg[1], caller, callee, tmap);
        }
        c->nins = n;
        /* the way out */
        if (isret(b->jmp.type)) {
            if (b->jmp.type != Jret0) {
                retval[nret] = clone_ref(b->jmp.arg, caller, callee, tmap);
                if (isretbh(b->jmp.type)) {
                    /* a byte or half returned: what the caller reads is the
                     * value extended as the return type says */
                    Ref ext = newtmp("inl", Kw, caller);

                    ni = &c->ins[c->nins++];
                    *ni = (Ins){ .cls = Kw, .to = ext, .arg = { retval[nret], R } };
                    ni->op = b->jmp.type == Jretsb ? Oextsb
                           : b->jmp.type == Jretub ? Oextub
                           : b->jmp.type == Jretsh ? Oextsh : Oextuh;
                    retval[nret] = ext;
                }
                retblk[nret] = c;
                nret++;
            }
            c->jmp.type = Jjmp;
            c->jmp.arg = R;
            c->s1 = cont;
            c->s2 = 0;
        } else {
            c->jmp.type = b->jmp.type;
            c->jmp.arg = clone_ref(b->jmp.arg, caller, callee, tmap);
            c->s1 = b->s1 ? bmap[b->s1->visit] : 0;
            c->s2 = b->s2 ? bmap[b->s2->visit] : 0;
        }
    }

    /* the second half of the caller's block */
    ntail = (int)blk->nins - call_idx - 1;
    tail = vnew(ntail + 1, sizeof(Ins), PFn);
    n = 0;
    retphi = 0;
    if (!req(call_ins.to, R) && nret == 1) {
        tail[n++] = (Ins){ .op = Ocopy, .cls = call_ins.cls, .to = call_ins.to,
                           .arg = { retval[0], R } };
    } else if (!req(call_ins.to, R) && nret > 1) {
        retphi = alloc(sizeof *retphi);
        retphi->to = call_ins.to;
        retphi->cls = call_ins.cls;
        retphi->narg = nret;
        retphi->arg = vnew(nret, sizeof retphi->arg[0], PFn);
        retphi->blk = vnew(nret, sizeof retphi->blk[0], PFn);
        for (k = 0; k < nret; k++) {
            retphi->arg[k] = retval[k];
            retphi->blk[k] = retblk[k];
        }
        retphi->link = 0;
    }
    for (k = call_idx + 1; k < (int)blk->nins; k++)
        tail[n++] = blk->ins[k];
    cont->ins = tail;
    cont->nins = n;
    cont->phi = retphi;
    cont->jmp = blk->jmp;
    cont->s1 = blk->s1;
    cont->s2 = blk->s2;
    /* whoever the block jumped to now hears from its second half */
    for (k = 0; k < 2; k++) {
        succ = k ? cont->s2 : cont->s1;
        if (!succ || (k && succ == cont->s1))
            continue;
        for (p = succ->phi; p; p = p->link)
            for (a = 0; a < p->narg; a++)
                if (p->blk[a] == blk)
                    p->blk[a] = cont;
    }

    /* the first half: what came before the arguments, then into the clone */
    nhead = arg_start;
    head = vnew(nhead ? nhead : 1, sizeof(Ins), PFn);
    for (k = 0; k < nhead; k++)
        head[k] = blk->ins[k];
    blk->ins = head;
    blk->nins = nhead;
    firstc = bmap[callee->start->visit];
    blk->jmp.type = Jjmp;
    blk->jmp.arg = R;
    blk->s1 = firstc;
    blk->s2 = 0;

    /* layout: first half, the clone in the callee's order, second half */
    last = blk;
    cont->link = blk->link;
    for (b = callee->start; b; b = b->link) {
        last->link = bmap[b->visit];
        last = bmap[b->visit];
    }
    last->link = cont;
    caller->nblk += nb + 1;

    free(tmap);
    free(bmap);
    free(retval);
    free(retblk);
    return cont;
}

/* Count indirect (RTmp arg) references to inline-marked functions in
 * the caller's IR. A function-pointer-store of $foo also counts: the
 * symbol's address is being taken, so the standalone is needed. */
static void
count_indirect_refs(Fn *fn)
{
    Blk *b;
    Ins *i;
    Con *c;
    InlRec *rec;
    int j;

    for (b = fn->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            /* Walk both args for RCon CAddr references. Direct call
             * targets (arg[0] of Ocall) are handled in inline_check. */
            for (j = 0; j < 2; j++) {
                if (rtype(i->arg[j]) != RCon) continue;
                /* Skip Ocall arg[0] — handled in inline_check */
                if (i->op == Ocall && j == 0) continue;
                c = &fn->con[i->arg[j].val];
                if (c->type != CAddr) continue;
                rec = inline_lookup(str(c->sym.id));
                if (rec && rec->hint) rec->n_indirect++;
            }
        }
    }
}

void
inline_check(Fn *fn)
{
    Blk *b;
    Ins *i;
    Con *c;
    InlRec *rec;
    char *target;
    char *trace;
    int idx, ok;

    InlRec *self;
    Blk *cont;

    trace = getenv("CC_TRACE_INLINE");
    self = inline_lookup(fn->name);
    if (self) {
        if (self->checked || self->busy)
            return;
        self->busy = 1;
    }

    /* Walk blocks. We restart per-block after each splice because
     * blk->ins is rewritten. */
    for (b = fn->start; b; b = b->link) {
restart:
        for (idx = 0, i = b->ins; i < &b->ins[b->nins]; i++, idx++) {
            if (i->op != Ocall) continue;
            if (rtype(i->arg[0]) != RCon) continue;
            c = &fn->con[i->arg[0].val];
            if (c->type != CAddr) continue;
            target = str(c->sym.id);
            rec = inline_lookup(target);
            if (rec && !rec->eligible && auto_inline_ok(fn, rec)) {
                /* its own calls first: the body cloned is the final one */
                inline_check(rec->fn);
                if (auto_inline_ok(fn, rec)
                    && (cont = splice_fn(fn, b, idx, rec->fn)) != 0) {
                    if (rec->hint) {
                        /* inline_fully_consumed() decides on these */
                        rec->n_direct++;
                        rec->n_inlined++;
                    } else
                        rec->absorbed = 1;
                    if (trace)
                        fprintf(stderr, "[inline] %s absorbed %s (%s)\n",
                                fn->name, rec->name,
                                rec->sites == 1 ? "one call site" : "static inline");
                    b = cont;
                    goto restart;
                }
            }
            if (!rec || !rec->eligible || !rec->body) {
                if (rec && rec->hint) {
                    /* Inline-marked but not eligible (or body missing
                     * post-clone) — counts as decline. */
                    rec->n_direct++;
                    rec->n_declined++;
                }
                if (trace) {
                    fprintf(stderr,
                        "[inline] check %s -> %s: %s\n",
                        fn->name, target,
                        rec ? (rec->eligible ? "no body saved"
                                             : "ineligible")
                            : "not in table (forward ref?)");
                }
                continue;
            }
            rec->n_direct++;
            ok = splice_at(fn, b, idx, rec->body);
            if (ok) rec->n_inlined++;
            else    rec->n_declined++;
            if (trace) {
                fprintf(stderr,
                    "[inline] splice %s -> %s @ blk %s idx %d: %s\n",
                    fn->name, rec->name, b->name, idx,
                    ok ? "INLINED" : "declined");
                if (ok && atoi(trace) >= 2) {
                    fprintf(stderr, "[inline] caller after splice:\n");
                    printfn(fn, stderr);
                }
            }
            if (ok) {
                /* IR rewrote: re-scan this block from start (the
                 * Ocall is gone, but new ins may exist). */
                goto restart;
            }
        }
    }

    /* Track indirect references after splicing (any remaining CAddr ref
     * to an inline-marked fn that's NOT an Ocall target means the
     * symbol's address was taken; standalone must be emitted). */
    count_indirect_refs(fn);

    /* a function left without a call is a leaf (parse.c decided on the
     * calls it read) */
    {
        int calls = 0;

        for (b = fn->start; b; b = b->link)
            for (i = b->ins; i < &b->ins[b->nins]; i++)
                if (i->op == Ocall)
                    calls = 1;
        if (!calls)
            fn->leaf = 1;
    }
    if (self) {
        self->busy = 0;
        self->checked = 1;
    }
}

/* Before any inlining: how many direct calls each function has in the
 * module. */
void
inline_count_sites(Fn *fn)
{
    Blk *b;
    Ins *i;
    Con *c;
    InlRec *rec;

    for (b = fn->start; b; b = b->link)
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            if (i->op != Ocall || rtype(i->arg[0]) != RCon)
                continue;
            c = &fn->con[i->arg[0].val];
            if (c->type != CAddr)
                continue;
            rec = inline_lookup(str(c->sym.id));
            if (rec)
                rec->sites++;
        }
}

/* Its one call site took its body: there is nothing left to emit. */
int
inline_absorbed(const char *name)
{
    InlRec *r;

    for (r = inl_head; r; r = r->next)
        if (strcmp(r->name, name) == 0)
            return r->absorbed;
    return 0;
}

/* Record an indirect (data-section) reference to a symbol. Called from
 * main.c's data callback when a Dat line references a symbol by name
 * (e.g., `static void (*fp)(void) = foo;` emits `data $fp = { l $foo }`).
 * A data-section reference is equivalent to taking the function's
 * address and prevents standalone-emit suppression. */
void
inline_record_dat_ref(const char *name)
{
    InlRec *r;
    for (r = inl_head; r; r = r->next) {
        if (strcmp(r->name, name) == 0) {
            r->n_indirect++;
            if (getenv("CC_TRACE_INLINE")) {
                fprintf(stderr,
                    "[inline] dat ref to %s -> n_indirect=%d\n",
                    name, r->n_indirect);
            }
            return;
        }
    }
}

/* Returns 1 if `name`'s standalone emission can be safely skipped in
 * this TU. Two suppress conditions for an inline-hinted function:
 *
 *  (a) Fully consumed: at least one direct caller existed and every
 *      direct call site was successfully inlined, no indirect refs.
 *  (b) Header-only inclusion: the body was parsed (because the header
 *      defining it was included) but this TU has neither a direct
 *      caller nor an indirect reference. Emitting the standalone here
 *      would create a duplicate symbol with whatever TU is the
 *      canonical definition site for the symbol.
 *
 * The "canonical definition site" is signalled by an indirect
 * reference: a TU that wants to provide the external standalone (e.g.,
 * to satisfy a callback registration or a fallback for non-inlining
 * TUs) takes the function's address — `void *p = (void *)fn;` — which
 * increments n_indirect and prevents suppression. */
int
inline_fully_consumed(const char *name)
{
    InlRec *r;
    for (r = inl_head; r; r = r->next) {
        if (strcmp(r->name, name) == 0) {
            if (!r->hint) return 0;
            /* (b) Header-only inclusion */
            if (r->n_direct == 0 && r->n_indirect == 0) {
                if (getenv("CC_TRACE_INLINE")) {
                    fprintf(stderr,
                        "[inline] suppress emit of %s (unused in TU)\n",
                        name);
                }
                return 1;
            }
            /* (a) Fully consumed */
            if (r->n_direct > 0 && r->n_declined == 0
                && r->n_indirect == 0) {
                if (getenv("CC_TRACE_INLINE")) {
                    fprintf(stderr,
                        "[inline] suppress emit of %s: direct=%d "
                        "inlined=%d (all consumed)\n",
                        name, r->n_direct, r->n_inlined);
                }
                return 1;
            }
            return 0;
        }
    }
    return 0;
}
