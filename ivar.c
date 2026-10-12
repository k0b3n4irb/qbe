#include "all.h"

/* OpenSNES (stage 2, T2, 2026-10-12): a loop counter rewritten as the
 * scaled index it is only used as.
 *
 * `for (k = 0; k < N; k++) a[k] = v;` reaches here as
 *
 *     @hd  %i =w phi @pre 0, @latch %i1
 *          %c =w cultw %i, N
 *          jnz %c, @body, @exit
 *     @body
 *          %e =l extuh %i
 *          %m =l mul %e, 2
 *          %p =l add %m, $a
 *          storew %v, %p
 *     @latch
 *          %i1 =w add %i, 1
 *
 * and the w65816 backend, which keeps a temp in the direct page and
 * indexes with X, emits `lda i / asl a / tax` before the access and
 * `lda i / inc a / sta i / cmp #N` after it. When EVERY use of the
 * counter is the same scaled index c*i (through `extuh` then `mul c` or
 * `shl`, or `mul c` in w), its own increment and its bound test, the
 * counter is replaced by j = c*i: a phi initialised to c*i0, stepping by
 * c*step, tested against c*N; `extuh j` takes the place of the product,
 * the old phi and its add go. Nothing else is created: a second counter
 * next to the first would cost this target more than the `asl` it saves.
 * What the rewrite buys is a counter that IS the index — T3 keeps it in
 * X and the loop becomes `sta a,x / inx / inx / cpx #c*N / bcc`.
 *
 * The rewrite is exact when nothing wraps at 16 bits: the init and the
 * bound are constants, the step a positive constant, and c*(N - 1 + step)
 * fits; the loop runs with i in [i0, N) because the test is the header's
 * `cultw` and the body is its true branch. A counter with another use, a
 * variable bound, two scales, several latches or several entries is left
 * alone. QBE_NO_IVAR=1 switches the pass off. requires rpo, pred, use;
 * maintains them (filluse at the end).
 */

static BSet inloop[1];
static Blk *latch;
static uint nlatch;

static int
conval(Fn *fn, Ref r, int64_t *v)
{
	Con *c;

	if (rtype(r) != RCon)
		return 0;
	c = &fn->con[r.val];
	if (c->type != CBits)
		return 0;
	*v = c->bits.i;
	return 1;
}

/* the add that steps the counter: `%i1 =w add %i, step`, in the loop */
static Ins *
stepins(Fn *fn, Ref i, Ref i1, int64_t *step)
{
	Tmp *t;
	Ins *d;

	if (rtype(i1) != RTmp)
		return 0;
	t = &fn->tmp[i1.val];
	d = t->def;
	if (!d || d->op != Oadd || d->cls != Kw)
		return 0;
	if (!bshas(inloop, t->bid))
		return 0;
	if (req(d->arg[0], i) && conval(fn, d->arg[1], step))
		return d;
	if (req(d->arg[1], i) && conval(fn, d->arg[0], step))
		return d;
	return 0;
}

/* a product by a constant, as a scale: `mul x, c`, `mul c, x` or `shl x, k` */
static int
scaleof(Fn *fn, Ins *m, Ref x, int64_t *c)
{
	int64_t v;

	if (m->op == Omul) {
		if (req(m->arg[0], x) && conval(fn, m->arg[1], &v))
			{ *c = v; return 1; }
		if (req(m->arg[1], x) && conval(fn, m->arg[0], &v))
			{ *c = v; return 1; }
		return 0;
	}
	if (m->op == Oshl && req(m->arg[0], x) && conval(fn, m->arg[1], &v)
	&& v >= 0 && v < 15) {
		*c = (int64_t)1 << v;
		return 1;
	}
	return 0;
}

/* every use of r1 becomes r2; the use lists are rebuilt by the caller */
static void
substuses(Fn *fn, Ref r1, Ref r2)
{
	Tmp *t;
	Use *u;
	Blk *b;
	Phi *q;
	Ref *pr;
	int n;

	t = &fn->tmp[r1.val];
	for (u=t->use; u<&t->use[t->nuse]; u++) {
		b = fn->rpo[u->bid];
		switch (u->type) {
		case UPhi:
			q = u->u.phi;
			for (pr=q->arg; pr<&q->arg[q->narg]; pr++)
				if (req(*pr, r1))
					*pr = r2;
			break;
		case UIns:
			for (n=0; n<2; n++)
				if (req(u->u.ins->arg[n], r1))
					u->u.ins->arg[n] = r2;
			break;
		case UJmp:
			if (req(b->jmp.arg, r1))
				b->jmp.arg = r2;
			break;
		case UXXX:
			die("unreachable");
		}
	}
}

/* Insert `ni` right after `at` in block b. Definitions move: filluse after. */
static void
insafter(Blk *b, Ins *at, Ins ni)
{
	Ins *v, *i;
	uint n;

	v = vnew(b->nins + 1, sizeof v[0], PFn);
	n = 0;
	for (i=b->ins; i<&b->ins[b->nins]; i++) {
		v[n++] = *i;
		if (i == at)
			v[n++] = ni;
	}
	idup(b, v, n);
}

/* One counter of the loop headed by hd (preheader pre, latch set). Returns
 * 1 when a rewrite was done (def pointers are stale: the caller refills). */
static int
rewrite(Fn *fn, Blk *hd, Blk *pre, Phi *p)
{
	Tmp *t, *te;
	Use *u, *ue;
	Ins *add, *cmp, *e, *m, *i;
	Ref init, next, j, jn;
	Blk *b;
	int64_t i0, step, n, c, cc, v;
	uint ipre, ilatch, nidx;
	Ins **muls;
	uint nmul, k, k2;

	if (p->narg != 2 || p->cls != Kw)
		return 0;
	if (p->blk[0] == pre && p->blk[1] == latch)
		ipre = 0, ilatch = 1;
	else if (p->blk[1] == pre && p->blk[0] == latch)
		ipre = 1, ilatch = 0;
	else
		return 0;
	init = p->arg[ipre];
	next = p->arg[ilatch];
	if (!conval(fn, init, &i0) || i0 < 0 || i0 > 0xFFFF)
		return 0;
	add = stepins(fn, p->to, next, &step);
	if (!add || step <= 0 || step > 0xFFFF)
		return 0;
	/* the stepped value feeds the phi and nothing else: `i + 1` used as a
	 * value elsewhere (a colour, a count) is a use of the counter */
	if (fn->tmp[next.val].nuse != 1)
		return 0;

	/* the bound: the header's jnz on `cultw i, N`, true into the loop */
	if (hd->jmp.type != Jjnz || rtype(hd->jmp.arg) != RTmp)
		return 0;
	cmp = fn->tmp[hd->jmp.arg.val].def;
	if (!cmp || cmp->op != Ocultw || !req(cmp->arg[0], p->to)
	|| !conval(fn, cmp->arg[1], &n) || n < 0 || n > 0xFFFF)
		return 0;
	if (fn->tmp[hd->jmp.arg.val].bid != hd->id)
		return 0;
	if (!bshas(inloop, hd->s1->id) || bshas(inloop, hd->s2->id))
		return 0;
	if (fn->tmp[hd->jmp.arg.val].nuse != 1)
		return 0;

	/* every other use of i: the same scale, through extuh or directly */
	t = &fn->tmp[p->to.val];
	c = 0;
	nidx = 0;
	muls = vnew(0, sizeof muls[0], PFn);
	nmul = 0;
	for (u=t->use; u<&t->use[t->nuse]; u++) {
		if (u->type != UIns)
			return 0;
		i = u->u.ins;
		if (i == add || i == cmp)
			continue;
		if (!bshas(inloop, u->bid))
			return 0;
		if (i->op == Oextuh && i->cls == Kl) {
			te = &fn->tmp[i->to.val];
			if (te->nuse == 0)
				return 0;
			for (ue=te->use; ue<&te->use[te->nuse]; ue++) {
				if (ue->type != UIns)
					return 0;
				m = ue->u.ins;
				if (m->cls != Kl || !scaleof(fn, m, i->to, &cc))
					return 0;
				if (nidx && cc != c)
					return 0;
				c = cc;
				nidx++;
				vgrow(&muls, ++nmul);
				muls[nmul-1] = m;
			}
		} else if (i->cls == Kw && scaleof(fn, i, p->to, &cc)) {
			if (nidx && cc != c)
				return 0;
			c = cc;
			nidx++;
			vgrow(&muls, ++nmul);
			muls[nmul-1] = i;
		} else
			return 0;
	}
	if (nidx == 0 || c <= 1 || c > 0xFFFF)
		return 0;

	/* nothing wraps: the largest value the new counter takes is
	 * c * (N - 1 + step), the last i that passed the test plus a step */
	v = c * (n - 1 + step);
	if (n > 0 && v > 0xFFFF)
		return 0;
	if (c * i0 > 0xFFFF || c * n > 0xFFFF)
		return 0;

	/* j = c*i: the phi, its step, the products and the test rewritten */
	j = newtmp("ivar", Kw, fn);
	jn = newtmp("ivar", Kw, fn);
	fn->tmp[j.val].bid = hd->id;
	for (k=0; k<nmul; k++) {
		m = muls[k];
		if (m->op == Oextuh)
			continue;
		if (m->cls == Kl) {
			/* mul (extuh i), c  ->  the extuh itself, on j */
			e = 0;
			for (k2=0; k2<2; k2++)
				if (rtype(m->arg[k2]) == RTmp
				&& fn->tmp[m->arg[k2].val].def
				&& fn->tmp[m->arg[k2].val].def->op == Oextuh)
					e = fn->tmp[m->arg[k2].val].def;
			assert(e);
			e->arg[0] = j;
			substuses(fn, m->to, e->to);
			*m = (Ins){.op = Onop};
		} else {
			substuses(fn, m->to, j);
			*m = (Ins){.op = Onop};
		}
	}
	cmp->arg[0] = j;
	cmp->arg[1] = getcon(c * n, fn);
	p->to = j;
	p->arg[ipre] = getcon(c * i0, fn);
	p->arg[ilatch] = jn;
	b = fn->rpo[fn->tmp[next.val].bid];
	/* the old counter's add goes (its only use was the phi, and its operand
	 * has no definition any more); the new one takes its place */
	*add = (Ins){.op = Onop};
	insafter(b, add, (Ins){.op = Oadd, .cls = Kw, .to = jn,
		.arg = {j, getcon(c * step, fn)}});
	if (debug['V'])
		fprintf(stderr, "  ivar: %s: %s -> %s * %d (step %d, bound %d)\n",
			fn->name, t->name, fn->tmp[j.val].name, (int)c,
			(int)(c * step), (int)(c * n));
	return 1;
}

void
ivar(Fn *fn)
{
	Blk *hd, *pre, *b;
	Phi *p;
	uint n, k, npre;
	int again;

	if (getenv("QBE_NO_IVAR"))
		return;
	bsinit(inloop, fn->nblk);
	do {
		again = 0;
		for (n=0; n<fn->nblk && !again; n++) {
			hd = fn->rpo[n];
			bszero(inloop);
			latch = 0;
			nlatch = 0;
			for (k=0; k<hd->npred; k++)
				if (hd->pred[k]->id >= n) {
					latch = hd->pred[k];
					nlatch++;
				}
			if (nlatch != 1)
				continue;
			for (b=fn->start; b; b=b->link)
				b->visit = -1u;
			bsset(inloop, hd->id);
			/* the blocks that reach the latch without leaving hd */
			{
				Blk **stk;
				uint nstk, q;
				stk = vnew(1, sizeof stk[0], PFn);
				nstk = 0;
				if (latch != hd) {
					bsset(inloop, latch->id);
					stk[nstk++] = latch;
				}
				while (nstk) {
					b = stk[--nstk];
					for (q=0; q<b->npred; q++) {
						if (b->pred[q]->id < hd->id)
							continue;
						if (bshas(inloop, b->pred[q]->id))
							continue;
						bsset(inloop, b->pred[q]->id);
						vgrow(&stk, nstk + 1);
						stk[nstk++] = b->pred[q];
					}
				}
			}
			pre = 0;
			npre = 0;
			for (k=0; k<hd->npred; k++)
				if (!bshas(inloop, hd->pred[k]->id)) {
					pre = hd->pred[k];
					npre++;
				}
			if (npre != 1 || !hd->s1 || !hd->s2)
				continue;
			for (p=hd->phi; p; p=p->link)
				if (rewrite(fn, hd, pre, p)) {
					filluse(fn);
					if (debug['V'])
						printfn(fn, stderr);
					again = 1;
					break;
				}
		}
	} while (again);
}
