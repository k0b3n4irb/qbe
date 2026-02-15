/*
 * QBE 65816 Backend - Code Emission
 *
 * Maps QBE's virtual registers R0-R7 to direct page $00-$0F.
 * Maps SSA temps to stack slots starting at $10.
 * Uses A as scratch for all ALU operations.
 *
 * For local variables (alloc), we track which temps hold stack addresses
 * and use direct stack-relative addressing for loads/stores through them.
 *
 * Alloc tracking is done in abi0 (before promote() removes alloc instructions)
 * and the results are stored in w65816_alloc_size[] and w65816_alloc_slots.
 */

#include "all.h"

/* Strip .L prefix from symbol names for WLA-DX compatibility.
 * WLA-DX doesn't support labels starting with '.'
 * e.g., ".Lstring.1" becomes "string.1"
 */
static char *
stripsym(char *name)
{
    if (name[0] == '.' && name[1] == 'L')
        return name + 2;
    return name;
}

static FILE *outf;
static int framesize;  /* Current function's frame size */
static int argbytes;   /* Bytes of arguments pushed for current call */

/* Offset from SP to caller's parameter area, past local frame + JSL return address.
 * Stack layout after prologue (framesize F):
 *   S+1 to S+F:   local frame (F bytes)
 *   S+F+1:        PCL  }
 *   S+F+2:        PCH  } JSL return address (3 bytes)
 *   S+F+3:        PBR  }
 *   S+F+4:        first param low byte
 * Param slot=-2 (first) -> offset = F + 2 + (-slot) = F + 4
 */
#define PARAM_OFFSET 2  /* framesize + PARAM_OFFSET + (-slot) */

/* A-register cache: track which Ref is currently in A to skip redundant loads */
static int acache_valid;
static Ref acache_ref;

static void acache_set(Ref r) { acache_valid = 1; acache_ref = r; }
static void acache_invalidate(void) { acache_valid = 0; }
static int acache_has(Ref r) { return acache_valid && req(acache_ref, r); }

/* 8/16-bit A-mode tracking: avoid redundant sep #$20 / rep #$20 */
static int amode_8bit;  /* 0 = 16-bit (default), 1 = 8-bit */

/* Tracks whether the last emitload call actually emitted an lda instruction.
 * When true, N and Z flags reflect the loaded value (no cmp.w #0 needed).
 * When false (A-cache hit), flags are stale and cmp.w #0 is required. */
static int last_load_emitted;

static void emit_sep20(void) {
    if (!amode_8bit) {
        fprintf(outf, "\tsep #$20\n");
        amode_8bit = 1;
    }
    /* 8-bit mode changes A semantics — invalidate cache */
    acache_invalidate();
}

static void emit_rep20(void) {
    if (amode_8bit) {
        fprintf(outf, "\trep #$20\n");
        amode_8bit = 0;
    }
}

/* Emit PLX-based stack cleanup: pops N bytes (must be even) using PLX.
 * PLX is 5 cycles on 65816 with 16-bit X and does NOT clobber A.
 * For non-void calls with small cleanup, this avoids the expensive
 * tax/tsa/clc/adc/tas/txa sequence (13 cycles).
 * Threshold: PLX for ≤ 4 bytes (non-void) or ≤ 2 bytes (void).
 */
static void
emit_plx_cleanup(int nbytes)
{
    int nplx = nbytes / 2;
    for (int j = 0; j < nplx; j++)
        fprintf(outf, "\tplx\n");
}

/* Check if opcode is a word/long comparison */
static int
is_cmp_op(int op)
{
    return (op >= Ocmpw && op <= Ocmpw1)
        || (op >= Ocmpl && op <= Ocmpl1);
}

/* Check if opcode is a commutative ALU operation.
 * Used by dead store analysis (Case 3) and operand swap optimization. */
static int
is_commutative_op(int op)
{
    return op == Oadd || op == Oand || op == Oor || op == Oxor;
}

/*
 * For a comparison opcode, determine:
 * - swap: whether to swap operands before comparing
 * - bt: branch instruction name when condition is TRUE
 * - bf: branch instruction name when condition is FALSE
 *
 * Mapping is derived from existing boolean materialization code:
 *  ceq: load a, cmp b → Z=1 means TRUE → beq/bne
 *  cne: load a, cmp b → Z=0 means TRUE → bne/beq
 *  cslt: load a, cmp b → N=1 means TRUE → bmi/bpl
 *  csge: load a, cmp b → N=0 means TRUE → bpl/bmi
 *  csgt: swap, load b, cmp a → N=1 means TRUE → bmi/bpl
 *  csle: swap, load b, cmp a → N=0 means TRUE → bpl/bmi
 *  cult: load a, cmp b → C=0 means TRUE → bcc/bcs
 *  cuge: load a, cmp b → C=1 means TRUE → bcs/bcc
 *  cugt: swap, load b, cmp a → C=0 means TRUE → bcc/bcs
 *  cule: swap, load b, cmp a → C=1 means TRUE → bcs/bcc
 */
static void
cmp_branch_info(int op, int *swap, const char **bt, const char **bf)
{
    *swap = 0;
    switch (op) {
    case Oceqw: case Oceql:
        *bt = "beq"; *bf = "bne"; break;
    case Ocnew: case Ocnel:
        *bt = "bne"; *bf = "beq"; break;
    case Ocsltw: case Ocsltl:
        *bt = "bmi"; *bf = "bpl"; break;
    case Ocsgew: case Ocsgel:
        *bt = "bpl"; *bf = "bmi"; break;
    case Ocsgtw: case Ocsgtl:
        *swap = 1; *bt = "bmi"; *bf = "bpl"; break;
    case Ocslew: case Ocslel:
        *swap = 1; *bt = "bpl"; *bf = "bmi"; break;
    case Ocultw: case Ocultl:
        *bt = "bcc"; *bf = "bcs"; break;
    case Ocugew: case Ocugel:
        *bt = "bcs"; *bf = "bcc"; break;
    case Ocugtw: case Ocugtl:
        *swap = 1; *bt = "bcc"; *bf = "bcs"; break;
    case Oculew: case Oculel:
        *swap = 1; *bt = "bcs"; *bf = "bcc"; break;
    default:
        *bt = "bne"; *bf = "beq"; break;
    }
}

/* Leaf function optimization: parameter alias propagation + dead return store */
#define MAX_ALIAS_TEMPS 256
static int temp_alias[MAX_ALIAS_TEMPS];    /* 0 = no alias, negative = param slot */
static int alloc_param[MAX_ALIAS_TEMPS];   /* 0 = no param, negative = param slot for alloc */
static int leaf_opt;                        /* 1 = leaf optimizations active */

/* Dead return store elimination */
static int temp_use_count[MAX_ALIAS_TEMPS];
static int temp_is_retval[MAX_ALIAS_TEMPS];
static int skip_dead_retstore_temp;  /* temp index to skip store, or -1 */

/* Dead store elimination: temps whose stack slot stores can be skipped */
static int temp_is_dead_store[MAX_ALIAS_TEMPS];

/* Comparison+branch fusion state:
 * When a comparison instruction's result is used only by the block's jnz,
 * we skip boolean materialization (0/1) and emit a direct compare+branch.
 */
static int fused_cmp;       /* 1 if a fused comparison is pending */
static int fused_cmp_op;    /* the comparison opcode */
static Ref fused_cmp_r0;    /* first operand */
static Ref fused_cmp_r1;    /* second operand */

/* Pre-pass: count how many times each temp is used as an operand */
static void
count_temp_uses(Fn *fn)
{
    Blk *b;
    Ins *i;
    Phi *p;
    int a, n, idx;

    memset(temp_use_count, 0, sizeof(temp_use_count));
    memset(temp_is_retval, 0, sizeof(temp_is_retval));
    for (b = fn->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            for (a = 0; a < 2; a++) {
                if (rtype(i->arg[a]) == RTmp && i->arg[a].val >= Tmp0) {
                    idx = i->arg[a].val - Tmp0;
                    if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                        temp_use_count[idx]++;
                }
            }
        }
        if (isret(b->jmp.type) && rtype(b->jmp.arg) == RTmp && b->jmp.arg.val >= Tmp0) {
            idx = b->jmp.arg.val - Tmp0;
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS) {
                temp_use_count[idx]++;
                temp_is_retval[idx] = 1;
            }
        }
        for (p = b->phi; p; p = p->link)
            for (n = 0; n < p->narg; n++)
                if (rtype(p->arg[n]) == RTmp && p->arg[n].val >= Tmp0) {
                    idx = p->arg[n].val - Tmp0;
                    if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                        temp_use_count[idx]++;
                }
    }
}

/* Pre-scan: build alias table from param loads, storew into allocs, and copy chains */
static void
build_alias_table(Fn *fn)
{
    Blk *b;
    Ins *i;
    Ref r0, r1;
    int idx, src_idx, dst_idx;
    int alloc_store_count[MAX_ALIAS_TEMPS];  /* how many stores target each alloc */

    memset(temp_alias, 0, sizeof(temp_alias));
    memset(alloc_param, 0, sizeof(alloc_param));
    if (!leaf_opt)
        return;

    /* First: count stores to each alloc slot.
     * Only allocs with EXACTLY 1 store are pure param shadows.
     * If a param is modified (e.g., count-- in a loop), the alloc
     * receives 2+ stores and must NOT be optimized away. */
    memset(alloc_store_count, 0, sizeof(alloc_store_count));
    for (b = fn->start; b; b = b->link)
        for (i = b->ins; i < &b->ins[b->nins]; i++)
            if (i->op == Ostorew || i->op == Ostoreh || i->op == Ostoreb
                || i->op == Ostorel) {
                r1 = i->arg[1];
                if (rtype(r1) == RTmp && r1.val >= Tmp0) {
                    dst_idx = r1.val - Tmp0;
                    if (dst_idx >= 0 && dst_idx < MAX_ALIAS_TEMPS
                        && w65816_alloc_size[dst_idx] > 0)
                        alloc_store_count[dst_idx]++;
                }
            }

    for (b = fn->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            r0 = i->arg[0];
            r1 = i->arg[1];

            /* Oloadsw from negative slot (param) → alias result temp */
            if ((i->op == Oloadsw || i->op == Oloaduw || i->op == Oload)
                && rtype(r0) == RSlot && rsval(r0) < 0
                && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
                idx = i->to.val - Tmp0;
                if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                    temp_alias[idx] = rsval(r0);
            }
            /* storew <val>, <alloc_temp> — if val is param-aliased AND
             * alloc receives exactly 1 store, mark as param shadow */
            else if ((i->op == Ostorew || i->op == Ostoreh)
                && rtype(r0) == RTmp && r0.val >= Tmp0
                && rtype(r1) == RTmp && r1.val >= Tmp0) {
                src_idx = r0.val - Tmp0;
                dst_idx = r1.val - Tmp0;
                if (src_idx >= 0 && src_idx < MAX_ALIAS_TEMPS && temp_alias[src_idx] != 0
                    && dst_idx >= 0 && dst_idx < MAX_ALIAS_TEMPS
                    && w65816_alloc_size[dst_idx] > 0
                    && alloc_store_count[dst_idx] == 1) {
                    alloc_param[dst_idx] = temp_alias[src_idx];
                }
            }
            /* loadw <alloc_temp> — if alloc holds param, alias result */
            else if ((i->op == Oloadsw || i->op == Oloaduw || i->op == Oload)
                && rtype(r0) == RTmp && r0.val >= Tmp0
                && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
                src_idx = r0.val - Tmp0;
                if (src_idx >= 0 && src_idx < MAX_ALIAS_TEMPS
                    && alloc_param[src_idx] != 0) {
                    idx = i->to.val - Tmp0;
                    if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                        temp_alias[idx] = alloc_param[src_idx];
                }
            }
            /* Ocopy: propagate alias through copy chains */
            else if (i->op == Ocopy
                && rtype(r0) == RTmp && r0.val >= Tmp0
                && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
                src_idx = r0.val - Tmp0;
                dst_idx = i->to.val - Tmp0;
                if (src_idx >= 0 && src_idx < MAX_ALIAS_TEMPS && temp_alias[src_idx] != 0
                    && dst_idx >= 0 && dst_idx < MAX_ALIAS_TEMPS)
                    temp_alias[dst_idx] = temp_alias[src_idx];
            }
            /* Oextuh/etc: propagate alias through no-op extensions.
             * NOTE: Oextub/Oextsb are NOT included — they perform real work
             * (AND #$00FF / sign-extend) and break the alias chain. */
            else if ((i->op == Oextuh || i->op == Oextsh
                      || i->op == Oextuw || i->op == Oextsw)
                && rtype(r0) == RTmp && r0.val >= Tmp0
                && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
                src_idx = r0.val - Tmp0;
                dst_idx = i->to.val - Tmp0;
                if (src_idx >= 0 && src_idx < MAX_ALIAS_TEMPS && temp_alias[src_idx] != 0
                    && dst_idx >= 0 && dst_idx < MAX_ALIAS_TEMPS)
                    temp_alias[dst_idx] = temp_alias[src_idx];
            }
        }
    }
}

/* Check if instruction is a no-op (emits no real code) given current alias/param state.
 * Used by mark_dead_stores to skip over instructions that don't touch A. */
static int
is_nop_instruction(Ins *i)
{
    int idx;

    /* Alloc with param-shadow → nop when frameless */
    if (i->op == Oalloc4 || i->op == Oalloc8 || i->op == Oalloc16) {
        if (rtype(i->to) == RTmp && i->to.val >= Tmp0) {
            idx = i->to.val - Tmp0;
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS && alloc_param[idx] != 0)
                return 1;
        }
        return 0;
    }

    /* Copy with aliased source → skipped */
    if (i->op == Ocopy && rtype(i->arg[0]) == RTmp && i->arg[0].val >= Tmp0) {
        idx = i->arg[0].val - Tmp0;
        if (idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0)
            return 1;
    }

    /* Store into param-shadow alloc → skipped */
    if ((i->op == Ostorew || i->op == Ostoreh)
        && rtype(i->arg[1]) == RTmp && i->arg[1].val >= Tmp0) {
        idx = i->arg[1].val - Tmp0;
        if (idx >= 0 && idx < MAX_ALIAS_TEMPS && alloc_param[idx] != 0)
            return 1;
    }

    /* Load from param slot or param-shadow alloc → aliased, skipped */
    if (i->op == Oloadsw || i->op == Oloaduw || i->op == Oload) {
        if (rtype(i->arg[0]) == RSlot && rsval(i->arg[0]) < 0)
            return 1;
        if (rtype(i->arg[0]) == RTmp && i->arg[0].val >= Tmp0) {
            idx = i->arg[0].val - Tmp0;
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS && alloc_param[idx] != 0)
                return 1;
        }
    }

    /* No-op extensions with aliased source → skipped.
     * NOTE: Oextub/Oextsb excluded — they emit real code (AND/sign-extend). */
    if ((i->op == Oextuh || i->op == Oextsh
         || i->op == Oextsw || i->op == Oextuw)
        && rtype(i->arg[0]) == RTmp && i->arg[0].val >= Tmp0) {
        idx = i->arg[0].val - Tmp0;
        if (idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0)
            return 1;
    }

    return 0;
}

/* Check if instruction consumes arg[0] via emitload (A-cache hit possible).
 * Instructions that load r1 first (swapped cmp) or use emitload_adj with
 * sp_adjust > 0 (Oarg) cannot benefit from A-cache for r0. */
static int
consumes_r0_via_emitload(Ins *i)
{
    switch (i->op) {
    /* Swapped comparisons: r0 goes through emitop2 (slot read, not A-cache) */
    case Ocsgtw: case Ocsgtl:
    case Ocslew: case Ocslel:
    case Ocugtw: case Ocugtl:
    case Oculew: case Oculel:
        return 0;
    /* Arg push: uses emitload_adj which checks A-cache (hits if just produced) */
    case Oarg: case Oargsb: case Oargub: case Oargsh: case Oarguh:
        return 1;
    case Ocall:
        return 0;
    /* Alloc: doesn't load arg[0] value */
    case Oalloc4: case Oalloc8: case Oalloc16:
        return 0;
    /* Mul: variable*variable loads r1 first; conservative exclude */
    case Omul:
        return 0;
    default:
        return 1;
    }
}

/* Mark temps whose stack slot stores can be safely skipped.
 * A temp is a dead store if:
 *  Case 1: use_count==0 and not a jnz/ret arg → truly unused
 *  Case 2: use_count==1, used as arg[0] of immediately next real instruction
 *           in the same block, and that instruction loads r0 via emitload
 *           → A-cache serves the value, slot never read
 * Exclusions: aliased temps, alloc instructions, phi args, jnz args */
static void
mark_dead_stores(Fn *fn)
{
    Blk *b;
    Ins *i, *next;
    Phi *p;
    int idx, n;
    int is_phi_arg[MAX_ALIAS_TEMPS];
    int is_jnz_arg[MAX_ALIAS_TEMPS];

    memset(temp_is_dead_store, 0, sizeof(temp_is_dead_store));
    memset(is_phi_arg, 0, sizeof(is_phi_arg));
    memset(is_jnz_arg, 0, sizeof(is_jnz_arg));

    /* Identify phi args and jnz args (both need their slots) */
    for (b = fn->start; b; b = b->link) {
        for (p = b->phi; p; p = p->link)
            for (n = 0; n < p->narg; n++)
                if (rtype(p->arg[n]) == RTmp && p->arg[n].val >= Tmp0) {
                    idx = p->arg[n].val - Tmp0;
                    if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                        is_phi_arg[idx] = 1;
                }
        if (b->jmp.type == Jjnz && rtype(b->jmp.arg) == RTmp
            && b->jmp.arg.val >= Tmp0) {
            idx = b->jmp.arg.val - Tmp0;
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS)
                is_jnz_arg[idx] = 1;
        }
    }

    for (b = fn->start; b; b = b->link) {
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            if (rtype(i->to) != RTmp || i->to.val < Tmp0)
                continue;
            idx = i->to.val - Tmp0;
            if (idx < 0 || idx >= MAX_ALIAS_TEMPS)
                continue;

            /* Exclude aliased temps (already optimized) */
            if (temp_alias[idx] != 0)
                continue;
            /* Exclude alloc instructions (produce addresses, not values) */
            if (i->op == Oalloc4 || i->op == Oalloc8 || i->op == Oalloc16)
                continue;
            /* Exclude phi args (need slot for inter-block moves) */
            if (is_phi_arg[idx])
                continue;
            /* Exclude jnz args — UNLESS the temp is a fusable comparison.
             * When comparison+branch fusion fires (is_cmp_op, last in block,
             * use_count==0), emitjmp re-emits the comparison from operands
             * and never reads the result from its stack slot → dead store. */
            if (is_jnz_arg[idx]) {
                int fusable = (is_cmp_op(i->op)
                    && temp_use_count[idx] == 0
                    && i == &b->ins[b->nins] - 1);
                if (!fusable)
                    continue;
            }

            /* Case 1: never used as operand (jnz/ret not counted here) */
            if (temp_use_count[idx] == 0 && !temp_is_retval[idx]) {
                temp_is_dead_store[idx] = 1;
                continue;
            }

            /* Case 2: used once as arg[0] of immediately next real instruction */
            if (temp_use_count[idx] == 1 && !temp_is_retval[idx]) {
                next = i + 1;
                while (next < &b->ins[b->nins] && is_nop_instruction(next))
                    next++;

                if (next < &b->ins[b->nins]
                    && rtype(next->arg[0]) == RTmp
                    && next->arg[0].val >= Tmp0
                    && (next->arg[0].val - Tmp0) == idx
                    && consumes_r0_via_emitload(next)) {
                    temp_is_dead_store[idx] = 1;
                }
            }

            /* Case 3: used once as arg[1] of immediately next commutative instruction.
             * The commutative swap will load r1 from A-cache (since we just produced it)
             * and use r0 as the emitop2 operand, so the slot store is never read. */
            if (temp_use_count[idx] == 1 && !temp_is_retval[idx]
                && !temp_is_dead_store[idx]) {
                next = i + 1;
                while (next < &b->ins[b->nins] && is_nop_instruction(next))
                    next++;

                if (next < &b->ins[b->nins]
                    && rtype(next->arg[1]) == RTmp
                    && next->arg[1].val >= Tmp0
                    && (next->arg[1].val - Tmp0) == idx
                    && is_commutative_op(next->op)) {
                    temp_is_dead_store[idx] = 1;
                }
            }
        }
    }
}

/* Check if a leaf function can be frameless (no real stack slots needed) */
static int
can_be_frameless(Fn *fn)
{
    Blk *b;
    Ins *i;
    int a, idx;

    if (fn->dynalloc) return 0;

    /* Check alloc slots: allow param-shadow allocs (fully optimized away),
     * but reject allocs used for other purposes */
    if (w65816_alloc_slots > 0) {
        for (b = fn->start; b; b = b->link)
            for (i = b->ins; i < &b->ins[b->nins]; i++)
                for (a = 0; a < 2; a++) {
                    Ref r = i->arg[a];
                    if (rtype(r) == RTmp && r.val >= Tmp0) {
                        idx = r.val - Tmp0;
                        if (idx >= 0 && idx < MAX_ALLOC_TEMPS
                            && w65816_alloc_size[idx] > 0
                            && alloc_param[idx] == 0)
                            return 0;  /* non-param alloc used as operand */
                    }
                }
    }

    /* Check that all non-aliased, non-dead-retval temps are absent */
    for (b = fn->start; b; b = b->link) {
        int is_ret_block = isret(b->jmp.type);
        Ins *last = (b->nins > 0) ? &b->ins[b->nins - 1] : NULL;
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            if (rtype(i->to) != RTmp || i->to.val < Tmp0) continue;
            idx = i->to.val - Tmp0;
            if (idx < 0 || idx >= MAX_ALIAS_TEMPS) continue;

            /* Skip alloc instructions — handled above */
            if (i->op == Oalloc4 || i->op == Oalloc8 || i->op == Oalloc16) continue;

            /* Skip storew/storeh into param-shadow alloc slots */
            if ((i->op == Ostorew || i->op == Ostoreh) && !rtype(i->to)) continue;

            /* Skip aliased temps (param copies / copy chains) */
            if (temp_alias[idx] != 0) continue;

            /* Skip dead store temps (slot never read, A-cache serves value) */
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_is_dead_store[idx]) continue;

            /* Skip dead return stores */
            if (is_ret_block && i == last && req(i->to, b->jmp.arg)
                && temp_use_count[idx] == 1 && temp_is_retval[idx]) continue;

            /* This temp needs a real stack slot → can't go frameless */
            return 0;
        }
    }
    return 1;
}

/* Alloc slot offset tracking - computed from w65816_alloc_size */
static int allocslot[MAX_ALLOC_TEMPS];  /* stack offset for each alloc temp */

/*
 * Get the stack slot offset if this temp is from alloc, -1 otherwise.
 * The offset is into the alloc region of the frame.
 */
static int
getallocslot(Ref r, Fn *fn)
{
    (void)fn;
    if (rtype(r) != RTmp)
        return -1;
    int idx = r.val - Tmp0;
    if (idx >= 0 && idx < MAX_ALLOC_TEMPS && w65816_alloc_size[idx] > 0)
        return allocslot[idx];
    return -1;
}

/* Direct page address for register */
static int
regaddr(int r)
{
    if (r >= R0 && r <= R7)
        return (r - R0) * 2;
    return 0;
}

/* Check if ref is a virtual register (R0-R7) */
static int
isvreg(Ref r)
{
    return rtype(r) == RTmp && r.val >= R0 && r.val <= R7;
}

/*
 * Assign a slot to a temp if not already assigned
 */
static int
maybeassign(Ref r, Fn *fn, int slot)
{
    if (rtype(r) == RTmp && r.val >= Tmp0) {
        if (fn->tmp[r.val].slot < 0) {
            fn->tmp[r.val].slot = slot++;
        }
    }
    return slot;
}

/*
 * Safe phi coalescing: Only coalesce if the phi argument is:
 * 1. Not used elsewhere in the block that defines it (would lose value)
 * 2. Not itself a phi result (avoid circular dependencies)
 *
 * For safety, we now only coalesce constants and single-use temps that
 * don't conflict with other phis in the same block.
 *
 * Actually, the safest approach is to NOT coalesce at all and rely on
 * explicit phi moves (emitphimoves). This is less efficient but correct.
 */
static void
coalescephi(Phi *p, Fn *fn)
{
    /* Just ensure the phi result has a slot */
    if (rtype(p->to) != RTmp || p->to.val < Tmp0)
        return;

    if (fn->tmp[p->to.val].slot < 0) {
        fn->tmp[p->to.val].slot = fn->slot++;
    }

    /* Don't coalesce phi arguments - let emitphimoves handle copies */
}

/*
 * Assign stack slots to all unassigned temps.
 * Called before emission when skiprega is set.
 * Must handle ALL refs (destinations and operands) since
 * optimization passes may create new temps.
 *
 * IMPORTANT: Process phi nodes first and coalesce their arguments
 * so that phi argument writes update the phi result directly.
 */
static int
assignslots(Fn *fn)
{
    Blk *b;
    Ins *i;
    Phi *p;
    int n;
    int maxslot = fn->slot;

    /* First pass: coalesce phi arguments with their results */
    for (b = fn->start; b; b = b->link) {
        for (p = b->phi; p; p = p->link) {
            coalescephi(p, fn);
        }
    }

    /* Update maxslot after phi coalescing */
    maxslot = fn->slot;

    /* Second pass: assign remaining slots */
    for (b = fn->start; b; b = b->link) {
        /* Process phi nodes - result already handled, check args */
        for (p = b->phi; p; p = p->link) {
            maxslot = maybeassign(p->to, fn, maxslot);
            for (n = 0; n < p->narg; n++)
                maxslot = maybeassign(p->arg[n], fn, maxslot);
        }
        /* Process instructions */
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            maxslot = maybeassign(i->to, fn, maxslot);
            maxslot = maybeassign(i->arg[0], fn, maxslot);
            maxslot = maybeassign(i->arg[1], fn, maxslot);
        }
        /* Process jump arguments */
        maxslot = maybeassign(b->jmp.arg, fn, maxslot);
    }
    return maxslot;
}

/*
 * Load value into accumulator
 * sp_adjust: additional offset to add for stack-relative loads
 *            (used during argument pushing when SP has already changed)
 */
static void
emitload_adj(Ref r, Fn *fn, int sp_adjust)
{
    Con *c;
    int slot;

    if (acache_has(r)) {
        last_load_emitted = 0;
        return;
    }

    switch (rtype(r)) {
    case RTmp:
        if (r.val >= R0 && r.val <= R7) {
            /* Virtual register - direct page */
            fprintf(outf, "\tlda.b $%02X\n", regaddr(r.val));
        } else if (r.val >= Tmp0) {
            /* Check for leaf-opt alias to param slot */
            int idx = r.val - Tmp0;
            if (leaf_opt && idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0) {
                int neg_slot = temp_alias[idx];
                fprintf(outf, "\tlda %d,s\n", framesize + PARAM_OFFSET + (-neg_slot) + sp_adjust);
            } else {
                /* Spilled temp */
                slot = fn->tmp[r.val].slot;
                if (slot >= 0)
                    fprintf(outf, "\tlda %d,s\n", (slot + 1) * 2 + sp_adjust);
                else
                    fprintf(outf, "\t; unallocated temp %d\n", r.val);
            }
        } else {
            fprintf(outf, "\t; unknown temp %d\n", r.val);
        }
        break;
    case RCon:
        c = &fn->con[r.val];
        if (c->type == CBits) {
            fprintf(outf, "\tlda.w #%d\n", (int)(c->bits.i & 0xFFFF));
        } else if (c->type == CAddr) {
            fprintf(outf, "\tlda.w #%s", stripsym(str(c->sym.id)));
            if (c->bits.i)
                fprintf(outf, "+%d", (int)c->bits.i);
            fprintf(outf, "\n");
        }
        break;
    case RSlot:
        slot = rsval(r);
        if (slot < 0) {
            /* Negative slot = parameter from caller's frame */
            /* Stack layout after prologue (framesize F):
             *   S+1 to S+F: local frame (F bytes)
             *   S+(F+1): PCL  }
             *   S+(F+2): PCH  } JSL return address (3 bytes)
             *   S+(F+3): PBR  }
             *   S+(F+4): first param low byte
             *   S+(F+5): first param high byte
             * slot=-2 means first param, slot=-4 means second, etc.
             * Offset = framesize + 4 + (-slot) - 2 = framesize + PARAM_OFFSET + (-slot)
             */
            fprintf(outf, "\tlda %d,s\n", framesize + PARAM_OFFSET + (-slot) + sp_adjust);
        } else {
            /* Positive slot = local variable in our frame */
            fprintf(outf, "\tlda %d,s\n", (slot + 1) * 2 + sp_adjust);
        }
        break;
    default:
        fprintf(outf, "\t; unknown ref type %d\n", rtype(r));
        break;
    }
    /* After loading, A changed — invalidate any stale cache entry.
     * The cache will be properly set by emitstore() after the
     * instruction finishes computing and stores its result.
     */
    last_load_emitted = 1;
    acache_invalidate();
}

/* Convenience wrapper for normal loads (no SP adjustment) */
static void
emitload(Ref r, Fn *fn)
{
    emitload_adj(r, fn, 0);
}

/*
 * Store accumulator
 */
static void
emitstore(Ref r, Fn *fn)
{
    int slot;
    int stored = 0;

    if (req(r, R))
        return;

    /* Leaf opt: skip store to aliased temp (A already has the value) */
    if (leaf_opt && rtype(r) == RTmp && r.val >= Tmp0) {
        int idx = r.val - Tmp0;
        if (idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0) {
            acache_set(r);
            return;
        }
    }

    /* Dead store elimination: skip store when temp's slot is never read */
    if (rtype(r) == RTmp && r.val >= Tmp0) {
        int idx = r.val - Tmp0;
        if (idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_is_dead_store[idx]) {
            acache_set(r);
            return;
        }
    }

    /* Dead return store elimination: skip store when temp is only used as retval */
    if (skip_dead_retstore_temp >= 0 && rtype(r) == RTmp && r.val >= Tmp0) {
        int idx = r.val - Tmp0;
        if (idx == skip_dead_retstore_temp) {
            acache_set(r);
            return;
        }
    }

    switch (rtype(r)) {
    case RTmp:
        if (r.val >= R0 && r.val <= R7) {
            fprintf(outf, "\tsta.b $%02X\n", regaddr(r.val));
            stored = 1;
        } else if (r.val >= Tmp0) {
            slot = fn->tmp[r.val].slot;
            if (slot >= 0) {
                fprintf(outf, "\tsta %d,s\n", (slot + 1) * 2);
                stored = 1;
            }
        }
        break;
    case RSlot:
        slot = rsval(r);
        if (slot < 0)
            fprintf(outf, "\tsta %d,s\n", framesize + PARAM_OFFSET + (-slot));
        else
            fprintf(outf, "\tsta %d,s\n", (slot + 1) * 2);
        stored = 1;
        break;
    default:
        break;
    }
    /* After sta, A still holds the stored value — set cache */
    if (stored)
        acache_set(r);
}

/*
 * Emit ALU second operand
 */
static void
emitop2(char *op, Ref r, Fn *fn)
{
    Con *c;
    int slot;

    switch (rtype(r)) {
    case RTmp:
        if (r.val >= R0 && r.val <= R7) {
            fprintf(outf, "\t%s.b $%02X\n", op, regaddr(r.val));
        } else if (r.val >= Tmp0) {
            /* Check for leaf-opt alias to param slot */
            int idx = r.val - Tmp0;
            if (leaf_opt && idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0) {
                int neg_slot = temp_alias[idx];
                fprintf(outf, "\t%s %d,s\n", op, framesize + PARAM_OFFSET + (-neg_slot));
            } else {
                slot = fn->tmp[r.val].slot;
                if (slot >= 0)
                    fprintf(outf, "\t%s %d,s\n", op, (slot + 1) * 2);
            }
        }
        break;
    case RCon:
        c = &fn->con[r.val];
        if (c->type == CBits) {
            fprintf(outf, "\t%s.w #%d\n", op, (int)(c->bits.i & 0xFFFF));
        } else if (c->type == CAddr) {
            fprintf(outf, "\t%s.w #%s", op, stripsym(str(c->sym.id)));
            if (c->bits.i)
                fprintf(outf, "+%d", (int)c->bits.i);
            fprintf(outf, "\n");
        }
        break;
    case RSlot:
        slot = rsval(r);
        if (slot < 0)
            fprintf(outf, "\t%s %d,s\n", op, framesize + PARAM_OFFSET + (-slot));
        else
            fprintf(outf, "\t%s %d,s\n", op, (slot + 1) * 2);
        break;
    default:
        break;
    }
}

/*
 * Emit instruction
 */
static void
emitins(Ins *i, Fn *fn)
{
    Ref r0, r1;
    Con *c;

    r0 = i->arg[0];
    r1 = i->arg[1];

    /* Ensure 16-bit A mode before each instruction.
     * Byte operations (Ostoreb, Oloadsb, Oloadub) switch to 8-bit internally.
     * This allows consecutive byte ops to stay in 8-bit mode (emit_rep20 is
     * a no-op when already in 16-bit mode).
     */
    switch (i->op) {
    case Ostoreb:
    case Oloadsb:
    case Oloadub:
        /* These manage their own mode — don't force 16-bit */
        break;
    default:
        emit_rep20();
        break;
    }

    switch (i->op) {
    case Oadd:
        /* INC/DEC optimization: inc a (2 cyc) vs clc;adc #1 (5 cyc) */
        if (rtype(r1) == RCon && fn->con[r1.val].type == CBits
            && (fn->con[r1.val].bits.i & 0xFFFF) == 1) {
            emitload(r0, fn);
            fprintf(outf, "\tinc a\n");
        } else if (rtype(r0) == RCon && fn->con[r0.val].type == CBits
            && (fn->con[r0.val].bits.i & 0xFFFF) == 1) {
            emitload(r1, fn);  /* commutative: swap */
            fprintf(outf, "\tinc a\n");
        } else if (rtype(r1) == RCon && fn->con[r1.val].type == CBits
            && (fn->con[r1.val].bits.i & 0xFFFF) == 0xFFFF) {
            emitload(r0, fn);  /* add -1 = dec */
            fprintf(outf, "\tdec a\n");
        } else if (rtype(r0) == RCon && fn->con[r0.val].type == CBits
            && (fn->con[r0.val].bits.i & 0xFFFF) == 0xFFFF) {
            emitload(r1, fn);  /* commutative: swap, add -1 = dec */
            fprintf(outf, "\tdec a\n");
        /* Commutative swap: if r1 is in A-cache, use it as the loaded operand */
        } else if (acache_has(r1) && !acache_has(r0)) {
            emitload(r1, fn);  /* A-cache hit, no emission */
            fprintf(outf, "\tclc\n");
            emitop2("adc", r0, fn);
        } else {
            emitload(r0, fn);
            fprintf(outf, "\tclc\n");
            emitop2("adc", r1, fn);
        }
        emitstore(i->to, fn);
        break;

    case Osub:
        /* DEC optimization: dec a (2 cyc) vs sec;sbc #1 (5 cyc) */
        if (rtype(r1) == RCon && fn->con[r1.val].type == CBits
            && (fn->con[r1.val].bits.i & 0xFFFF) == 1) {
            emitload(r0, fn);
            fprintf(outf, "\tdec a\n");
        } else if (rtype(r1) == RCon && fn->con[r1.val].type == CBits
            && (fn->con[r1.val].bits.i & 0xFFFF) == 0xFFFF) {
            emitload(r0, fn);  /* sub -1 = inc */
            fprintf(outf, "\tinc a\n");
        } else {
            emitload(r0, fn);
            fprintf(outf, "\tsec\n");
            emitop2("sbc", r1, fn);
        }
        emitstore(i->to, fn);
        break;

    case Oneg:
        /* Negate: result = -value = ~value + 1 (two's complement) */
        emitload(r0, fn);
        fprintf(outf, "\teor.w #$FFFF\n");
        fprintf(outf, "\tinc a\n");
        emitstore(i->to, fn);
        break;

    case Omul:
        /* 65816 has no MUL instruction - need to use a loop or library call */
        /* For now, emit a simple shift-add loop for small constants */
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int val = c->bits.i;
            if (val == 0) {
                fprintf(outf, "\tlda.w #0\n");
            } else if (val == 1) {
                emitload(r0, fn);
            } else if (val == 2) {
                emitload(r0, fn);
                fprintf(outf, "\tasl a\n");
            } else if (val == 4) {
                emitload(r0, fn);
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
            } else if (val == 8) {
                emitload(r0, fn);
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
            } else if (val == 16) {
                emitload(r0, fn);
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
            } else if (val == 32) {
                emitload(r0, fn);
                fprintf(outf, "\txba\n");  /* swap bytes = *256 */
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");  /* /8 = *32 */
            } else if (val == 3) {
                /* x*3 = x*2 + x */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tclc\n");
                fprintf(outf, "\tadc.w tcc__r9\n");
            } else if (val == 5) {
                /* x*5 = x*4 + x */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tclc\n");
                fprintf(outf, "\tadc.w tcc__r9\n");
            } else if (val == 6) {
                /* x*6 = (x*2 + x) * 2 = x*3*2 */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tclc\n");
                fprintf(outf, "\tadc.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
            } else if (val == 7) {
                /* x*7 = x*8 - x */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tsec\n");
                fprintf(outf, "\tsbc.w tcc__r9\n");
            } else if (val == 9) {
                /* x*9 = x*8 + x */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tclc\n");
                fprintf(outf, "\tadc.w tcc__r9\n");
            } else if (val == 10) {
                /* x*10 = (x*4 + x) * 2 = x*5*2 */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tasl a\n");
                fprintf(outf, "\tclc\n");
                fprintf(outf, "\tadc.w tcc__r9\n");
                fprintf(outf, "\tasl a\n");
            } else {
                /* General case: use stack for multiplier, call __mul */
                emitload(r1, fn);
                fprintf(outf, "\tpha\n");
                emitload_adj(r0, fn, 2);  /* Adjust for pushed value */
                fprintf(outf, "\tpha\n");
                fprintf(outf, "\tjsl __mul16\n");
                /* PLX cleanup: 4 bytes (2 args), doesn't clobber A */
                emit_plx_cleanup(4);
            }
        } else {
            /* Variable * variable - call __mul16 */
            emitload(r1, fn);
            fprintf(outf, "\tpha\n");
            emitload_adj(r0, fn, 2);  /* Adjust for pushed value */
            fprintf(outf, "\tpha\n");
            fprintf(outf, "\tjsl __mul16\n");
            /* PLX cleanup: 4 bytes (2 args), doesn't clobber A */
            emit_plx_cleanup(4);
        }
        emitstore(i->to, fn);
        break;

    case Odiv:
    case Oudiv:
        /* Division - use shifts for powers of 2, library call otherwise */
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int val = c->bits.i;
            if (val == 1) {
                emitload(r0, fn);
            } else if (val == 2) {
                emitload(r0, fn);
                fprintf(outf, "\tlsr a\n");
            } else if (val == 4) {
                emitload(r0, fn);
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
            } else if (val == 8) {
                emitload(r0, fn);
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
            } else if (val == 16) {
                emitload(r0, fn);
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
                fprintf(outf, "\tlsr a\n");
            } else if (val == 256) {
                emitload(r0, fn);
                fprintf(outf, "\txba\n");  /* swap bytes = /256 */
                fprintf(outf, "\tand.w #$00FF\n");
            } else {
                /* General case: call __div16 */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r0\n");
                emitload(r1, fn);
                fprintf(outf, "\tsta.w tcc__r1\n");
                fprintf(outf, "\tjsl __div16\n");
                fprintf(outf, "\tlda.w tcc__r0\n");
            }
        } else {
            /* Variable / variable - call __div16 */
            emitload(r0, fn);
            fprintf(outf, "\tsta.w tcc__r0\n");
            emitload(r1, fn);
            fprintf(outf, "\tsta.w tcc__r1\n");
            fprintf(outf, "\tjsl __div16\n");
            fprintf(outf, "\tlda.w tcc__r0\n");
        }
        emitstore(i->to, fn);
        break;

    case Orem:
    case Ourem:
        /* Modulo - use AND for powers of 2, library call otherwise */
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int val = c->bits.i;
            if (val == 2) {
                emitload(r0, fn);
                fprintf(outf, "\tand.w #1\n");
            } else if (val == 4) {
                emitload(r0, fn);
                fprintf(outf, "\tand.w #3\n");
            } else if (val == 8) {
                emitload(r0, fn);
                fprintf(outf, "\tand.w #7\n");
            } else if (val == 16) {
                emitload(r0, fn);
                fprintf(outf, "\tand.w #15\n");
            } else if (val == 256) {
                emitload(r0, fn);
                fprintf(outf, "\tand.w #255\n");
            } else {
                /* General case: call __mod16 */
                emitload(r0, fn);
                fprintf(outf, "\tsta.w tcc__r0\n");
                emitload(r1, fn);
                fprintf(outf, "\tsta.w tcc__r1\n");
                fprintf(outf, "\tjsl __mod16\n");
                fprintf(outf, "\tlda.w tcc__r0\n");
            }
        } else {
            /* Variable % variable - call __mod16 */
            emitload(r0, fn);
            fprintf(outf, "\tsta.w tcc__r0\n");
            emitload(r1, fn);
            fprintf(outf, "\tsta.w tcc__r1\n");
            fprintf(outf, "\tjsl __mod16\n");
            fprintf(outf, "\tlda.w tcc__r0\n");
        }
        emitstore(i->to, fn);
        break;

    case Oand:
        if (acache_has(r1) && !acache_has(r0)) {
            emitload(r1, fn);
            emitop2("and", r0, fn);
        } else {
            emitload(r0, fn);
            emitop2("and", r1, fn);
        }
        emitstore(i->to, fn);
        break;

    case Oor:
        if (acache_has(r1) && !acache_has(r0)) {
            emitload(r1, fn);
            emitop2("ora", r0, fn);
        } else {
            emitload(r0, fn);
            emitop2("ora", r1, fn);
        }
        emitstore(i->to, fn);
        break;

    case Oxor:
        if (acache_has(r1) && !acache_has(r0)) {
            emitload(r1, fn);
            emitop2("eor", r0, fn);
        } else {
            emitload(r0, fn);
            emitop2("eor", r1, fn);
        }
        emitstore(i->to, fn);
        break;

    case Oshl:
        emitload(r0, fn);
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int cnt = (int)c->bits.i;
            if (cnt >= 16) {
                /* Shift ≥ 16: result is always 0 */
                fprintf(outf, "\tlda.w #0\n");
            } else if (cnt >= 8) {
                /* XBA optimization: swap bytes + mask + remaining shifts */
                fprintf(outf, "\txba\n");
                fprintf(outf, "\tand.w #$FF00\n");
                for (int j = 0; j < cnt - 8; j++)
                    fprintf(outf, "\tasl a\n");
            } else {
                for (int j = 0; j < cnt; j++)
                    fprintf(outf, "\tasl a\n");
            }
        } else {
            /* Variable-count shift: save A, load count into X, loop */
            fprintf(outf, "\tpha\n");
            emitload(r1, fn);
            fprintf(outf, "\ttax\n");
            fprintf(outf, "\tpla\n");
            fprintf(outf, "\tcpx #0\n");
            fprintf(outf, "\tbeq +\n");
            fprintf(outf, "-\tasl a\n");
            fprintf(outf, "\tdex\n");
            fprintf(outf, "\tbne -\n");
            fprintf(outf, "+\n");
        }
        emitstore(i->to, fn);
        break;

    case Osar:
        /* Arithmetic shift right: preserves sign for negative values. */
        emitload(r0, fn);
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int cnt = (int)c->bits.i;
            if (cnt >= 16) {
                /* Shift ≥ 16: result is 0 (positive) or -1 (negative) */
                fprintf(outf, "\tcmp.w #$8000\n");
                fprintf(outf, "\tlda.w #0\n");
                fprintf(outf, "\tbcc +\n");
                fprintf(outf, "\tdec a\n");  /* A = $FFFF = -1 */
                fprintf(outf, "+\n");
            } else if (cnt >= 8) {
                /* XBA + mask + sign extend byte + remaining sar shifts */
                fprintf(outf, "\txba\n");
                fprintf(outf, "\tand.w #$00FF\n");
                /* Sign extend: if byte bit 7 set, fill high byte */
                fprintf(outf, "\tcmp.w #$0080\n");
                fprintf(outf, "\tbcc +\n");
                fprintf(outf, "\tora.w #$FF00\n");
                fprintf(outf, "+\n");
                for (int j = 0; j < cnt - 8; j++) {
                    fprintf(outf, "\tcmp.w #$8000\n");
                    fprintf(outf, "\tror a\n");
                }
            } else {
                for (int j = 0; j < cnt; j++) {
                    fprintf(outf, "\tcmp.w #$8000\n");
                    fprintf(outf, "\tror a\n");
                }
            }
        } else {
            fprintf(outf, "\tpha\n");
            emitload(r1, fn);
            fprintf(outf, "\ttax\n");
            fprintf(outf, "\tpla\n");
            fprintf(outf, "\tcpx #0\n");
            fprintf(outf, "\tbeq +\n");
            fprintf(outf, "-\tcmp.w #$8000\n");
            fprintf(outf, "\tror a\n");
            fprintf(outf, "\tdex\n");
            fprintf(outf, "\tbne -\n");
            fprintf(outf, "+\n");
        }
        emitstore(i->to, fn);
        break;

    case Oshr:
        emitload(r0, fn);
        if (rtype(r1) == RCon) {
            c = &fn->con[r1.val];
            int cnt = (int)c->bits.i;
            if (cnt >= 16) {
                /* Shift ≥ 16: result is always 0 */
                fprintf(outf, "\tlda.w #0\n");
            } else if (cnt >= 8) {
                /* XBA + mask + remaining shifts */
                fprintf(outf, "\txba\n");
                fprintf(outf, "\tand.w #$00FF\n");
                for (int j = 0; j < cnt - 8; j++)
                    fprintf(outf, "\tlsr a\n");
            } else {
                for (int j = 0; j < cnt; j++)
                    fprintf(outf, "\tlsr a\n");
            }
        } else {
            fprintf(outf, "\tpha\n");
            emitload(r1, fn);
            fprintf(outf, "\ttax\n");
            fprintf(outf, "\tpla\n");
            fprintf(outf, "\tcpx #0\n");
            fprintf(outf, "\tbeq +\n");
            fprintf(outf, "-\tlsr a\n");
            fprintf(outf, "\tdex\n");
            fprintf(outf, "\tbne -\n");
            fprintf(outf, "+\n");
        }
        emitstore(i->to, fn);
        break;

    case Ocopy:
        /* Leaf opt: propagate alias through copy chains */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0) {
            int src_idx = r0.val - Tmp0;
            if (src_idx >= 0 && src_idx < MAX_ALIAS_TEMPS && temp_alias[src_idx] != 0) {
                int dst_idx = (rtype(i->to) == RTmp && i->to.val >= Tmp0) ? i->to.val - Tmp0 : -1;
                if (dst_idx >= 0 && dst_idx < MAX_ALIAS_TEMPS) {
                    temp_alias[dst_idx] = temp_alias[src_idx];
                    break;  /* skip copy entirely */
                }
            }
        }
        if (!req(i->to, r0)) {
            emitload(r0, fn);
            emitstore(i->to, fn);
        }
        break;

    /* Comparison operations - produce 0 or 1 */
    case Oceqw:
    case Oceql:
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbeq +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocnew:
    case Ocnel:
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbne +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocsltw:
    case Ocsltl:
        /* Signed less than */
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbmi +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocsgtw:
    case Ocsgtl:
        /* Signed greater than: swap operands and use less than */
        emitload(r1, fn);
        emitop2("cmp", r0, fn);
        fprintf(outf, "\tbmi +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocslew:
    case Ocslel:
        /* Signed less or equal: !(a > b) */
        emitload(r1, fn);
        emitop2("cmp", r0, fn);
        fprintf(outf, "\tbmi +\n");
        fprintf(outf, "\tlda.w #1\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #0\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocsgew:
    case Ocsgel:
        /* Signed greater or equal: !(a < b) */
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbmi +\n");
        fprintf(outf, "\tlda.w #1\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #0\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocultw:
    case Ocultl:
        /* Unsigned less than */
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbcc +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocugtw:
    case Ocugtl:
        /* Unsigned greater than */
        emitload(r1, fn);
        emitop2("cmp", r0, fn);
        fprintf(outf, "\tbcc +\n");
        fprintf(outf, "\tlda.w #0\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #1\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Oculew:
    case Oculel:
        /* Unsigned less or equal */
        emitload(r1, fn);
        emitop2("cmp", r0, fn);
        fprintf(outf, "\tbcc +\n");
        fprintf(outf, "\tlda.w #1\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #0\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ocugew:
    case Ocugel:
        /* Unsigned greater or equal */
        emitload(r0, fn);
        emitop2("cmp", r1, fn);
        fprintf(outf, "\tbcc +\n");
        fprintf(outf, "\tlda.w #1\n");
        fprintf(outf, "\tbra ++\n");
        fprintf(outf, "+\tlda.w #0\n");
        fprintf(outf, "++\n");
        emitstore(i->to, fn);
        break;

    case Ostorel:
        /* Store long (32-bit) - store low word then high word at +2 */
        /* For 65816, we treat this as storing a 32-bit value in two parts */
        emitload(r0, fn);  /* Load low 16 bits */
        {
            int aslot = getallocslot(r1, fn);
            if (aslot >= 0) {
                /* Store to stack-allocated local variable */
                fprintf(outf, "\tsta %d,s\n", (aslot + 1) * 2);
                /* High word is typically 0 for near pointers */
                fprintf(outf, "\tlda.w #0\n");
                fprintf(outf, "\tsta %d,s\n", (aslot + 1) * 2 + 2);
            } else if (isvreg(r1)) {
                fprintf(outf, "\tsta ($%02X)\n", regaddr(r1.val));
                fprintf(outf, "\tlda.w #0\n");
                fprintf(outf, "\tldy.w #2\n");
                fprintf(outf, "\tsta ($%02X),y\n", regaddr(r1.val));
            } else if (rtype(r1) == RCon) {
                /* Direct address constant or symbol */
                c = &fn->con[r1.val];
                if (c->type == CAddr) {
                    fprintf(outf, "\tsta.w %s", stripsym(str(c->sym.id)));
                    if (c->bits.i)
                        fprintf(outf, "+%d", (int)c->bits.i);
                    fprintf(outf, "\n");
                    /* Store high word (always 0 for near pointers) */
                    fprintf(outf, "\tstz.w %s+%d\n", stripsym(str(c->sym.id)),
                            (int)c->bits.i + 2);
                } else {
                    fprintf(outf, "\tsta.l $%06lX\n", (unsigned long)c->bits.i);
                    fprintf(outf, "\tlda.w #0\n");
                    fprintf(outf, "\tsta.l $%06lX\n", (unsigned long)c->bits.i + 2);
                }
            } else {
                /* Address in temp - indirect store */
                fprintf(outf, "\tpha\n");
                emitload_adj(r1, fn, 2);
                fprintf(outf, "\ttax\n");
                fprintf(outf, "\tpla\n");
                fprintf(outf, "\tsta.l $0000,x\n");
                /* Store high word at +2 */
                fprintf(outf, "\tlda.w #0\n");
                fprintf(outf, "\tsta.l $0002,x\n");
            }
        }
        acache_invalidate();
        break;

    case Ostorew:
    case Ostoreh:
        /* Leaf opt: skip storew into param-shadow alloc slot */
        if (leaf_opt && rtype(r1) == RTmp && r1.val >= Tmp0) {
            int aidx = r1.val - Tmp0;
            if (aidx >= 0 && aidx < MAX_ALIAS_TEMPS && alloc_param[aidx] != 0)
                break;  /* param already in caller frame, skip copy */
        }
        /* stz optimization: store zero to symbol without loading A */
        if (rtype(r0) == RCon && fn->con[r0.val].type == CBits
            && (fn->con[r0.val].bits.i & 0xFFFF) == 0
            && rtype(r1) == RCon && fn->con[r1.val].type == CAddr) {
            c = &fn->con[r1.val];
            fprintf(outf, "\tstz.w %s", stripsym(str(c->sym.id)));
            if (c->bits.i)
                fprintf(outf, "+%d", (int)c->bits.i);
            fprintf(outf, "\n");
            acache_invalidate();
            break;
        }
        emitload(r0, fn);
        {
            int aslot = getallocslot(r1, fn);
            if (aslot >= 0) {
                /* Store to stack-allocated local variable */
                fprintf(outf, "\tsta %d,s\n", (aslot + 1) * 2);
            } else if (isvreg(r1)) {
                fprintf(outf, "\tsta ($%02X)\n", regaddr(r1.val));
            } else if (rtype(r1) == RCon) {
                /* Direct address constant or symbol */
                c = &fn->con[r1.val];
                if (c->type == CAddr) {
                    /* Symbol address - emit symbol name */
                    fprintf(outf, "\tsta.w %s", stripsym(str(c->sym.id)));
                    if (c->bits.i)
                        fprintf(outf, "+%d", (int)c->bits.i);
                    fprintf(outf, "\n");
                } else {
                    /* Literal address */
                    fprintf(outf, "\tsta.l $%06lX\n", (unsigned long)c->bits.i);
                }
            } else {
                /* Address in temp - indirect store */
                /* IMPORTANT: After pha, stack offsets change by 2 */
                fprintf(outf, "\tpha\n");
                emitload_adj(r1, fn, 2);  /* Adjust for pushed value */
                fprintf(outf, "\ttax\n");
                fprintf(outf, "\tpla\n");
                fprintf(outf, "\tsta.l $0000,x\n");
            }
        }
        acache_invalidate();
        break;

    case Ostoreb:
        /* Byte store with mode tracking.
         * For constant zero to symbol: use stz (no lda needed).
         * For constant values: switch to 8-bit first, use 8-bit immediate (2 bytes).
         * For variable values: load in 16-bit, then switch to 8-bit for store.
         */
        if (rtype(r0) == RCon && fn->con[r0.val].type == CBits
            && (fn->con[r0.val].bits.i & 0xFF) == 0
            && rtype(r1) == RCon && fn->con[r1.val].type == CAddr) {
            /* Byte store zero to symbol: use stz */
            emit_sep20();
            c = &fn->con[r1.val];
            fprintf(outf, "\tstz.w %s", stripsym(str(c->sym.id)));
            if (c->bits.i)
                fprintf(outf, "+%d", (int)c->bits.i);
            fprintf(outf, "\n");
            acache_invalidate();
            break;
        }
        if (rtype(r0) == RCon && fn->con[r0.val].type == CBits) {
            /* Constant value: switch to 8-bit first, then 8-bit immediate */
            int val = fn->con[r0.val].bits.i & 0xFF;
            emit_sep20();
            fprintf(outf, "\tlda #%d\n", val);  /* 8-bit immediate (2 bytes) */
        } else {
            /* Variable value: load in 16-bit mode, then switch */
            emitload(r0, fn);
            emit_sep20();
        }
        {
            int aslot = getallocslot(r1, fn);
            if (aslot >= 0) {
                /* Store to stack-allocated local variable */
                fprintf(outf, "\tsta %d,s\n", (aslot + 1) * 2);
            } else if (isvreg(r1)) {
                fprintf(outf, "\tsta ($%02X)\n", regaddr(r1.val));
            } else if (rtype(r1) == RCon) {
                /* Direct address constant or symbol */
                c = &fn->con[r1.val];
                if (c->type == CAddr) {
                    /* Symbol address - emit symbol name */
                    fprintf(outf, "\tsta.w %s", stripsym(str(c->sym.id)));
                    if (c->bits.i)
                        fprintf(outf, "+%d", (int)c->bits.i);
                    fprintf(outf, "\n");
                } else {
                    /* Literal address */
                    fprintf(outf, "\tsta.l $%06lX\n", (unsigned long)c->bits.i);
                }
            } else {
                /* Address in temp - indirect store */
                /* A already has the value, load addr to X, then store */
                /* IMPORTANT: After pha, stack offsets change by 2 */
                emit_rep20();               /* Need 16-bit for address */
                fprintf(outf, "\tpha\n");       /* Save value */
                emitload_adj(r1, fn, 2);        /* Adjust for pushed value */
                fprintf(outf, "\ttax\n");
                fprintf(outf, "\tpla\n");
                emit_sep20();               /* Back to 8-bit for store */
                fprintf(outf, "\tsta.l $0000,x\n");
            }
        }
        /* Don't emit_rep20() here — let next instruction decide.
         * If next is another Ostoreb, we stay in 8-bit mode (saving 6 cycles).
         * emitins() ensures 16-bit for all non-byte ops at entry.
         */
        acache_invalidate();
        break;

    case Oloadsw:
    case Oloaduw:
    case Oload:
        /* Leaf opt: alias param slot loads instead of copying */
        if (leaf_opt && rtype(r0) == RSlot && rsval(r0) < 0) {
            int idx = (rtype(i->to) == RTmp && i->to.val >= Tmp0) ? i->to.val - Tmp0 : -1;
            if (idx >= 0 && idx < MAX_ALIAS_TEMPS) {
                temp_alias[idx] = rsval(r0);
                break;  /* skip the load+store entirely */
            }
        }
        /* Leaf opt: loadw from param-shadow alloc → alias to param */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0) {
            int aidx = r0.val - Tmp0;
            if (aidx >= 0 && aidx < MAX_ALIAS_TEMPS && alloc_param[aidx] != 0) {
                int didx = (rtype(i->to) == RTmp && i->to.val >= Tmp0) ? i->to.val - Tmp0 : -1;
                if (didx >= 0 && didx < MAX_ALIAS_TEMPS) {
                    temp_alias[didx] = alloc_param[aidx];
                    break;  /* skip the load+store — will read from caller frame */
                }
            }
        }
        {
            int aslot = getallocslot(r0, fn);
            if (aslot >= 0) {
                /* Load from stack-allocated local variable */
                fprintf(outf, "\tlda %d,s\n", (aslot + 1) * 2);
            } else if (rtype(r0) == RSlot) {
                /* Direct stack access (e.g., loading parameter) */
                emitload(r0, fn);
            } else if (isvreg(r0)) {
                /* Indirect through register */
                fprintf(outf, "\tlda ($%02X)\n", regaddr(r0.val));
            } else if (rtype(r0) == RCon && fn->con[r0.val].type == CAddr) {
                /* Direct load from global/extern symbol */
                Con *c = &fn->con[r0.val];
                fprintf(outf, "\tlda.w %s", stripsym(str(c->sym.id)));
                if (c->bits.i)
                    fprintf(outf, "+%d", (int)c->bits.i);
                fprintf(outf, "\n");
            } else {
                /* Address in temp - indirect load */
                emitload(r0, fn);
                fprintf(outf, "\ttax\n");
                fprintf(outf, "\tlda.l $0000,x\n");
            }
        }
        emitstore(i->to, fn);
        break;

    case Oloadsb:
    case Oloadub:
        /* Load byte from memory, zero/sign extend to 16-bit */
        if (isvreg(r0)) {
            /* Pointer in virtual register - use indirect */
            emit_sep20();
            fprintf(outf, "\tlda ($%02X)\n", regaddr(r0.val));
            emit_rep20();
        } else if (rtype(r0) == RCon && fn->con[r0.val].type == CAddr) {
            /* Direct load from global/extern symbol */
            Con *c = &fn->con[r0.val];
            emit_sep20();
            fprintf(outf, "\tlda.w %s", stripsym(str(c->sym.id)));
            if (c->bits.i)
                fprintf(outf, "+%d", (int)c->bits.i);
            fprintf(outf, "\n");
            emit_rep20();
        } else {
            /* Pointer in stack slot - load addr, then indirect through X */
            emitload(r0, fn);  /* Load pointer value to A */
            fprintf(outf, "\ttax\n");  /* Transfer to X */
            emit_sep20();
            fprintf(outf, "\tlda.l $0000,x\n");  /* Load byte from memory */
            emit_rep20();
        }
        fprintf(outf, "\tand.w #$00FF\n");  /* Zero extend */
        if (i->op == Oloadsb) {
            /* Sign extend */
            fprintf(outf, "\tcmp.w #$0080\n");
            fprintf(outf, "\tbcc +\n");
            fprintf(outf, "\tora.w #$FF00\n");
            fprintf(outf, "+\n");
        }
        emitstore(i->to, fn);
        break;

    case Oextsb:
        /* Sign extend byte to word */
        emitload(r0, fn);
        fprintf(outf, "\tand.w #$00FF\n");
        fprintf(outf, "\tcmp.w #$0080\n");
        fprintf(outf, "\tbcc +\n");
        fprintf(outf, "\tora.w #$FF00\n");
        fprintf(outf, "+\n");
        emitstore(i->to, fn);
        break;

    case Oextub:
        /* Zero extend byte to word */
        emitload(r0, fn);
        fprintf(outf, "\tand.w #$00FF\n");
        emitstore(i->to, fn);
        break;

    case Oextsh:
        /* Sign extend half (already 16-bit, no-op for 65816) */
        /* Leaf opt: propagate alias through nop extension */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0
            && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
            int si = r0.val - Tmp0;
            int di = i->to.val - Tmp0;
            if (si >= 0 && si < MAX_ALIAS_TEMPS && temp_alias[si] != 0
                && di >= 0 && di < MAX_ALIAS_TEMPS) {
                temp_alias[di] = temp_alias[si];
                break;
            }
        }
        emitload(r0, fn);
        emitstore(i->to, fn);
        break;

    case Oextuh:
        /* Zero extend half (already 16-bit, no-op for 65816) */
        /* Leaf opt: propagate alias through nop extension */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0
            && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
            int si = r0.val - Tmp0;
            int di = i->to.val - Tmp0;
            if (si >= 0 && si < MAX_ALIAS_TEMPS && temp_alias[si] != 0
                && di >= 0 && di < MAX_ALIAS_TEMPS) {
                temp_alias[di] = temp_alias[si];
                break;
            }
        }
        emitload(r0, fn);
        emitstore(i->to, fn);
        break;

    case Oextsw:
        /* Sign extend word to long - for 16-bit 65816, just copy the value. */
        /* Leaf opt: propagate alias through nop extension */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0
            && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
            int si = r0.val - Tmp0;
            int di = i->to.val - Tmp0;
            if (si >= 0 && si < MAX_ALIAS_TEMPS && temp_alias[si] != 0
                && di >= 0 && di < MAX_ALIAS_TEMPS) {
                temp_alias[di] = temp_alias[si];
                break;
            }
        }
        emitload(r0, fn);
        emitstore(i->to, fn);
        break;

    case Oextuw:
        /* Zero extend word to long - for 16-bit 65816, just copy the value. */
        /* Leaf opt: propagate alias through nop extension */
        if (leaf_opt && rtype(r0) == RTmp && r0.val >= Tmp0
            && rtype(i->to) == RTmp && i->to.val >= Tmp0) {
            int si = r0.val - Tmp0;
            int di = i->to.val - Tmp0;
            if (si >= 0 && si < MAX_ALIAS_TEMPS && temp_alias[si] != 0
                && di >= 0 && di < MAX_ALIAS_TEMPS) {
                temp_alias[di] = temp_alias[si];
                break;
            }
        }
        emitload(r0, fn);
        emitstore(i->to, fn);
        break;

    case Oarg:
    case Oargsb:
    case Oargub:
    case Oargsh:
    case Oarguh:
        /* Push argument to stack.
         * Use pea.w for constants (saves 1 byte + 2 cycles, doesn't touch A).
         * For variables, use lda+pha. pha doesn't modify A, so set A-cache after.
         */
        if (rtype(r0) == RCon) {
            Con *ac = &fn->con[r0.val];
            if (ac->type == CBits) {
                fprintf(outf, "\tpea.w %d\n", (int)(ac->bits.i & 0xFFFF));
            } else if (ac->type == CAddr) {
                fprintf(outf, "\tpea.w %s", stripsym(str(ac->sym.id)));
                if (ac->bits.i)
                    fprintf(outf, "+%d", (int)ac->bits.i);
                fprintf(outf, "\n");
            } else {
                emitload_adj(r0, fn, argbytes);
                fprintf(outf, "\tpha\n");
                acache_set(r0);  /* pha doesn't change A — restore cache */
            }
        } else {
            emitload_adj(r0, fn, argbytes);
            fprintf(outf, "\tpha\n");
            acache_set(r0);  /* pha doesn't change A — restore cache */
        }
        argbytes += 2;  /* All args pushed as 16-bit */
        break;

    case Ocall:
        if (rtype(r0) == RCon) {
            c = &fn->con[r0.val];
            if (c->type == CAddr) {
                fprintf(outf, "\tjsl %s\n", stripsym(str(c->sym.id)));
            } else {
                /* Numeric constant as call target (unusual) */
                fprintf(outf, "\tjsl $%06lX\n", (unsigned long)c->bits.i);
            }
        } else {
            /* Indirect call: function pointer in temp/slot.
             * Load the 16-bit address, store to DP scratch (tcc__r9),
             * set bank byte to current bank ($00), then jml [tcc__r9].
             * Push return address for RTL to work correctly.
             */
            emitload_adj(r0, fn, argbytes);
            fprintf(outf, "\tsta.b tcc__r9\n");
            emit_sep20();
            fprintf(outf, "\tlda #$00\n");
            fprintf(outf, "\tsta.b tcc__r9+2\n");
            emit_rep20();
            fprintf(outf, "\tphk\n");
            fprintf(outf, "\tpea ++-1\n");
            fprintf(outf, "\tjml [tcc__r9]\n");
            fprintf(outf, "++\n");
        }
        {
            int cleanup = argbytes;
            int has_retval = !req(i->to, R);
            argbytes = 0;
            if (cleanup > 0) {
                int nplx = cleanup / 2;
                /* PLX threshold: 5 cycles/plx vs arithmetic (13 w/retval, 9 void).
                 * PLX for ≤ 4 bytes (non-void) or ≤ 2 bytes (void). */
                int plx_limit = has_retval ? 2 : 1;
                if (nplx <= plx_limit) {
                    /* PLX cleanup: doesn't clobber A, preserves return value */
                    emit_plx_cleanup(cleanup);
                } else {
                    /* Arithmetic cleanup: clobbers A, needs tax/txa for retval */
                    if (has_retval)
                        fprintf(outf, "\ttax\n");
                    fprintf(outf, "\ttsa\n");
                    fprintf(outf, "\tclc\n");
                    fprintf(outf, "\tadc.w #%d\n", cleanup);
                    fprintf(outf, "\ttas\n");
                    if (has_retval)
                        fprintf(outf, "\ttxa\n");
                }
            }
            if (has_retval) {
                emitstore(i->to, fn);
            } else {
                acache_invalidate();
            }
        }
        break;

    case Oalloc4:
    case Oalloc8:
    case Oalloc16:
        if (framesize == 0)
            break;  /* frameless — allocs are dead, skip */
        /* alloc returns a pointer to stack space.
         * The slot was already assigned in the pre-pass.
         * We just emit a no-op here since loads/stores through
         * this temp use direct stack-relative addressing.
         */
        {
            int aslot = getallocslot(i->to, fn);
            if (aslot >= 0) {
                /* Store slot offset to temp (for debugging/consistency) */
                fprintf(outf, "\tlda.w #%d\n", (aslot + 1) * 2);
                emitstore(i->to, fn);
            }
        }
        break;

    default:
        fprintf(outf, "\t; unhandled op %d\n", i->op);
        acache_invalidate();
        break;
    }

    /* A-cache is now maintained entirely by emitstore() (sets on store)
     * and emitload_adj() (invalidates on miss). No second switch needed.
     */
}

/*
 * Emit phi moves for a jump from 'from' to 'to'.
 * For each phi in 'to', find the argument from 'from' and:
 * - If constant: store constant to phi result's slot
 * - If temp with different slot: copy to phi result's slot
 *   (With coalescing, temps should already share slots)
 */
static void
emitphimoves(Blk *from, Blk *to, Fn *fn)
{
    Phi *p;
    int n;
    int dstslot;

    if (!to)
        return;

    for (p = to->phi; p; p = p->link) {
        /* Find the argument corresponding to 'from' */
        for (n = 0; n < p->narg; n++) {
            if (p->blk[n] == from) {
                /* Get destination slot (phi result) */
                if (rtype(p->to) != RTmp || p->to.val < Tmp0)
                    continue;
                dstslot = fn->tmp[p->to.val].slot;
                if (dstslot < 0)
                    continue;

                /* Handle the argument */
                if (rtype(p->arg[n]) == RCon) {
                    /* Constant: load and store to phi slot */
                    emitload(p->arg[n], fn);
                    fprintf(outf, "\tsta %d,s\n", (dstslot + 1) * 2);
                } else if (rtype(p->arg[n]) == RTmp && p->arg[n].val >= Tmp0) {
                    int idx = p->arg[n].val - Tmp0;
                    if (leaf_opt && idx >= 0 && idx < MAX_ALIAS_TEMPS && temp_alias[idx] != 0) {
                        /* Aliased to param slot — load from caller frame */
                        int neg_slot = temp_alias[idx];
                        fprintf(outf, "\tlda %d,s\n", framesize + PARAM_OFFSET + (-neg_slot));
                        fprintf(outf, "\tsta %d,s\n", (dstslot + 1) * 2);
                    } else {
                        /* Temp: check if we need to copy */
                        int srcslot = fn->tmp[p->arg[n].val].slot;
                        if (srcslot >= 0 && srcslot != dstslot) {
                            /* Different slots - need to copy */
                            fprintf(outf, "\tlda %d,s\n", (srcslot + 1) * 2);
                            fprintf(outf, "\tsta %d,s\n", (dstslot + 1) * 2);
                        }
                        /* If same slot, no copy needed (coalesced) */
                    }
                }
                break;
            }
        }
    }
}

/*
 * Emit jump/return
 */
static void
emitjmp(Blk *b, Fn *fn)
{
    switch (b->jmp.type) {
    case Jret0:
    case Jretw:
    case Jretl:
        if (!req(b->jmp.arg, R))
            emitload(b->jmp.arg, fn);
        break;
    case Jjmp:
        emitphimoves(b, b->s1, fn);
        if (b->s1 != b->link)
            fprintf(outf, "\tjmp @%s\n", b->s1->name);
        break;
    case Jjnz:
        if (fused_cmp) {
            /* Fused comparison+branch: emit compare and conditional branch
             * directly, skipping boolean materialization (0/1 on stack).
             * Saves ~6-10 instructions per conditional. */
            int swap;
            const char *bt, *bf;
            cmp_branch_info(fused_cmp_op, &swap, &bt, &bf);
            if (swap) {
                emitload(fused_cmp_r1, fn);
                emitop2("cmp", fused_cmp_r0, fn);
            } else {
                emitload(fused_cmp_r0, fn);
                emitop2("cmp", fused_cmp_r1, fn);
            }
            acache_invalidate();
            if (b->s1 == b->link) {
                /* True falls through: skip false path on TRUE */
                fprintf(outf, "\t%s +\n", bt);
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
            } else if (b->s2 == b->link) {
                /* False falls through: skip true path on FALSE */
                fprintf(outf, "\t%s +\n", bf);
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s2, fn);
            } else {
                /* Neither falls through */
                fprintf(outf, "\t%s +\n", bt);
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
            }
            fused_cmp = 0;
        } else {
            emitload(b->jmp.arg, fn);
            /* Only emit cmp.w #0 when A-cache hit skipped the lda.
             * When emitload actually emitted an lda, Z flag is already
             * correctly set from the loaded value. */
            if (!last_load_emitted)
                fprintf(outf, "\tcmp.w #0\n");
            if (b->s1 == b->link) {
                /* True branch falls through - invert: branch on zero to s2 */
                fprintf(outf, "\tbne +\n");
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
            } else if (b->s2 == b->link) {
                /* False branch falls through - branch on nonzero to s1 */
                fprintf(outf, "\tbeq +\n");
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s2, fn);
            } else {
                /* Neither falls through */
                fprintf(outf, "\tbne +\n");
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
            }
        }
        break;
    default:
        if (b->jmp.type >= Jjf && b->jmp.type <= Jjf1) {
            emitload(b->jmp.arg, fn);
            if (!last_load_emitted)
                fprintf(outf, "\tcmp.w #0\n");
            if (b->s1 == b->link) {
                fprintf(outf, "\tbne +\n");
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
            } else if (b->s2 == b->link) {
                fprintf(outf, "\tbeq +\n");
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s2, fn);
            } else {
                fprintf(outf, "\tbne +\n");
                emitphimoves(b, b->s2, fn);
                fprintf(outf, "\tjmp @%s\n", b->s2->name);
                fprintf(outf, "+\n");
                emitphimoves(b, b->s1, fn);
                fprintf(outf, "\tjmp @%s\n", b->s1->name);
            }
        }
        break;
    }
}

void
w65816_emitfn(Fn *fn, FILE *f)
{
    Blk *b;
    Ins *i;
    int slot;

    outf = f;

    /*
     * Compute allocslot[] offsets from w65816_alloc_size[] (set by abi0)
     * Alloc temps get stack slots first (lowest offsets from SP)
     */
    slot = 0;
    for (int j = 0; j < MAX_ALLOC_TEMPS; j++) {
        if (w65816_alloc_size[j] > 0) {
            allocslot[j] = slot;
            slot += w65816_alloc_size[j];
        } else {
            allocslot[j] = -1;
        }
    }

    /* Reserve slots for allocs in fn->slot so assignslots starts after them */
    fn->slot = slot;

    /* Assign slots to any unassigned temps (when skiprega is set) */
    fn->slot = assignslots(fn);

    /* Total frame size: alloc slots + temp slots + 1 for alignment */
    framesize = (fn->slot + 1) * 2;
    argbytes = 0;  /* Reset argument tracking for this function */

    /* Leaf function optimization pre-passes */
    leaf_opt = !fn->dynalloc;
    skip_dead_retstore_temp = -1;
    count_temp_uses(fn);
    build_alias_table(fn);
    mark_dead_stores(fn);

    /* Frame elimination: if all temps are aliased or dead-retval, go frameless */
    if (leaf_opt && can_be_frameless(fn))
        framesize = 0;

    /* NOTE: temp_alias is NOT reset here. build_alias_table() already computed
     * correct aliases. The emission pass uses these aliases via emitload_adj()
     * and emitstore() to redirect param accesses to the caller's frame.
     * Resetting here caused a mismatch: can_be_frameless() saw aliases,
     * set framesize=0, but the emission lost the aliases and wrote to
     * non-existent local slots, corrupting the caller's stack. */

    /* Debug: show slot assignments */
    fprintf(outf, "\n; Function: %s (framesize=%d, slots=%d, alloc_slots=%d, fn_leaf=%d, leaf_opt=%d)\n",
            fn->name, framesize, fn->slot, w65816_alloc_slots, fn->leaf, leaf_opt);
    for (int t = Tmp0; t < fn->ntmp; t++) {
        int aslot = (t - Tmp0 >= 0 && t - Tmp0 < MAX_ALLOC_TEMPS) ? allocslot[t - Tmp0] : -1;
        fprintf(outf, "; temp %d: slot=%d, alloc=%d\n", t, fn->tmp[t].slot, aslot);
    }
    fprintf(outf, ".SECTION \".text.%s\" SUPERFREE\n", fn->name);
    fprintf(outf, "%s:\n", fn->name);

    /* Prologue — no php/plp (PARAM_OFFSET=2 assumes no P byte on stack).
     * rep #$20 ensures 16-bit A: assembly callers (NMI handler, mode7)
     * may call C functions after sep #$20 (8-bit A mode).
     * Phase 2 (emit_rep20) ensures 16-bit before every return. */
    fprintf(outf, "\trep #$20\n");
    if (framesize > 2) {
        fprintf(outf, "\ttsa\n");
        fprintf(outf, "\tsec\n");
        fprintf(outf, "\tsbc.w #%d\n", framesize);
        fprintf(outf, "\ttas\n");
    }

    acache_invalidate();
    amode_8bit = 0;  /* Start in 16-bit mode (calling convention) */
    for (b = fn->start; b; b = b->link) {
        /* Ensure 16-bit mode at block entry — incoming edges may differ */
        emit_rep20();
        fprintf(outf, "@%s:\n", b->name);
        acache_invalidate();

        fused_cmp = 0;
        for (i = b->ins; i < &b->ins[b->nins]; i++) {
            /* Comparison+branch fusion: if the last instruction is a
             * comparison whose result is used only by this block's jnz,
             * skip boolean materialization and let emitjmp emit
             * a direct compare+conditional branch instead. */
            if (i == &b->ins[b->nins] - 1
                && b->jmp.type == Jjnz
                && is_cmp_op(i->op)
                && rtype(i->to) == RTmp && i->to.val >= Tmp0
                && req(i->to, b->jmp.arg)) {
                int cidx = i->to.val - Tmp0;
                if (cidx >= 0 && cidx < MAX_ALIAS_TEMPS
                    && temp_use_count[cidx] == 0) {
                    fused_cmp = 1;
                    fused_cmp_op = i->op;
                    fused_cmp_r0 = i->arg[0];
                    fused_cmp_r1 = i->arg[1];
                    continue;  /* skip emitins for this comparison */
                }
            }

            /* Dead return store elimination: detect last instruction
             * producing a temp that's only used as the return value */
            skip_dead_retstore_temp = -1;
            if (leaf_opt && i == &b->ins[b->nins] - 1 && isret(b->jmp.type)
                && rtype(i->to) == RTmp && i->to.val >= Tmp0
                && req(i->to, b->jmp.arg)) {
                int idx = i->to.val - Tmp0;
                if (idx >= 0 && idx < MAX_ALIAS_TEMPS
                    && temp_use_count[idx] == 1 && temp_is_retval[idx])
                    skip_dead_retstore_temp = idx;
            }
            emitins(i, fn);
            skip_dead_retstore_temp = -1;
        }

        if (isret(b->jmp.type)) {
            /* Ensure 16-bit before epilogue (emitjmp + stack cleanup) */
            emit_rep20();
            emitjmp(b, fn);
            if (framesize > 2) {
                int nplx = framesize / 2;
                int has_retval = (b->jmp.type != Jret0);
                /* PLX threshold: 5 cycles/plx vs arithmetic (13 w/retval, 9 void) */
                int plx_limit = has_retval ? 2 : 1;
                if (nplx <= plx_limit) {
                    /* PLX epilogue: doesn't clobber A, preserves return value */
                    emit_plx_cleanup(framesize);
                } else {
                    if (has_retval)
                        fprintf(outf, "\ttax\n");
                    fprintf(outf, "\ttsa\n");
                    fprintf(outf, "\tclc\n");
                    fprintf(outf, "\tadc.w #%d\n", framesize);
                    fprintf(outf, "\ttas\n");
                    if (has_retval)
                        fprintf(outf, "\ttxa\n");
                }
            }
            fprintf(outf, "\trtl\n");
        } else {
            /* Ensure 16-bit before jump (phi moves use 16-bit lda/sta) */
            emit_rep20();
            emitjmp(b, fn);
        }
    }

    fprintf(outf, ".ENDS\n");
}

void
w65816_emitfin(FILE *f)
{
    fprintf(f, "\n; End of generated code\n");
}
