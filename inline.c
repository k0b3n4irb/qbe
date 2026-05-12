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

#define INLINE_MAX_INSTR 8

typedef struct InlRec InlRec;
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
    InlRec *next;
};

static InlRec *inl_head;

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

    for (b = fn->start; b; b = b->link) {
        r->blk_count++;
        if (b->phi) r->has_phis = 1;
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
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

    if (getenv("CC_TRACE_INLINE")) {
        fprintf(stderr,
            "[inline] record %s: hint=%d blk=%d ins=%d phis=%d cond=%d "
            "calls=%d allocs=%d (max=%d) -> %s\n",
            r->name, r->hint, r->blk_count, r->ins_count,
            r->has_phis, r->has_cond, r->has_calls, r->has_allocs, max,
            r->eligible ? "ELIGIBLE" : "rejected");
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

void
inline_check(Fn *fn)
{
    Blk *b;
    Ins *i;
    Con *c;
    InlRec *rec;
    char *target;

    if (!getenv("CC_TRACE_INLINE"))
        return;

    for (b = fn->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            if (i->op != Ocall) continue;
            if (rtype(i->arg[0]) != RCon) continue;
            c = &fn->con[i->arg[0].val];
            if (c->type != CAddr) continue;
            target = str(c->sym.id);
            rec = inline_lookup(target);
            if (!rec) {
                fprintf(stderr,
                    "[inline] check %s -> %s: not in table (forward ref?)\n",
                    fn->name, target);
                continue;
            }
            fprintf(stderr, "[inline] check %s -> %s: %s\n",
                    fn->name, rec->name,
                    rec->eligible ? "WOULD INLINE" : "ineligible");
        }
    }
}
