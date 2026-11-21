/* OpenSNES: the exact IR idioms of the G1 lot (I1, I2, I3, I4, I5).
 *
 * Switch: OPENSNES_G1_IDIOMS=1 turns the pass on; it is OFF by default,
 * and off it does not touch the function (the output is byte for byte
 * the one without the pass). Only for a target whose `w` is 16 bits
 * (T.wordsz == 2, w65816): the rule below reads extsh / extsw of a `w`
 * as the sign extension of the whole word, which holds only there.
 * OPENSNES_G1_FAULT is the provoked fault the test must see, one per
 * idiom: =1 (I1) the rewritten value takes a logical shift (`shr`), so
 * its high word is 0 where it should be the sign mask $FFFF; =2 (I2) the
 * entry value is scaled at 16 bits (`extuh (v0 + v0)` in w), the range
 * assumption the rule refuses: bit 16 of 2*v0 is lost when v0 >= $8000;
 * =3 (I3) the trip count is not looked at, so a loop that never runs is
 * rotated and its body runs once; =4 (I4) the trip count of a reversed
 * counter is computed with signed compares read as unsigned. I4 runs
 * before I3, so a counter reversed by I4 is the bare `jnz n` I3 rotates.
 *
 * I1, narrow sign-extension chain. The IR of `(s32)(s >> k)` with s a
 * 16-bit signed value is
 *     %s2 =w sar %s, k
 *     %y  =l extsh %s2
 * and when the same function also has `%e =l extsh %s` (s widened as is)
 * defined in a block that dominates %y's, %y is rewritten
 *     %y  =l sar %e, k
 * The high word of %e is the sign mask of s (0 or $FFFF), built once;
 * a 32-bit `sar` by k of it keeps that mask as its high word and puts
 * floor(s / 2^k) in its low word, which is sar16(s, k). So the value is
 * the one of the original for every s and every k in 1..15: no range
 * fact is used. %y is then itself known as the extension of %s2, so a
 * chain `(s >> a) >> b` takes the rule again.
 *
 * Run once, on SSA, after threadjnz and before reassoc / gvn (the
 * later passes, gcm included, see an ordinary `sar` on an `l`).
 */
#include "all.h"

typedef struct G1ext G1ext;
struct G1ext {
	int w;      /* the 16-bit temp */
	int l;      /* an l temp known equal to its sign extension */
	Blk *b;     /* the block that defines l */
};

static int
g1_isext(Ins *i)
{
	return (i->op == Oextsh || i->op == Oextsw)
		&& i->cls == Kl && rtype(i->arg[0]) == RTmp;
}

/* the temp a chain of copies of w comes from (gvn, which would merge
 * them, runs after this pass) */
static int
g1_root(Fn *fn, int w)
{
	Ins *d;
	int n;

	for (n=0; n<64; n++) {
		d = fn->tmp[w].def;
		if (!d || d->op != Ocopy || rtype(d->arg[0]) != RTmp
		|| fn->tmp[d->arg[0].val].cls != Kw)
			break;
		w = d->arg[0].val;
	}
	return w;
}

static G1ext *
g1_findext(G1ext *v, uint n, int w, Blk *b)
{
	uint k;

	for (k=0; k<n; k++)
		if (v[k].w == w && dom(v[k].b, b))
			return &v[k];
	return 0;
}

static void
g1_i1(Fn *fn, int fault)
{
	Blk *b;
	Ins *i, *d;
	G1ext *v, *e;
	uint n, cap, bi;
	int64_t k;
	int w, nrw;

	cap = 16;
	n = 0;
	v = emalloc(cap * sizeof v[0]);
	nrw = 0;
	/* rpo order: a dominator is met before what it dominates */
	for (bi=0; bi<fn->nblk; bi++) {
		b = fn->rpo[bi];
		for (i=b->ins; i<&b->ins[b->nins]; i++) {
			if (!g1_isext(i) || rtype(i->to) != RTmp)
				continue;
			w = g1_root(fn, i->arg[0].val);
			d = fn->tmp[w].def;
			e = 0;
			if (d && d->op == Osar && d->cls == Kw
			&& rtype(d->arg[0]) == RTmp
			&& isconbits(fn, d->arg[1], &k) && k >= 1 && k <= 15)
				e = g1_findext(v, n, g1_root(fn, d->arg[0].val), b);
			if (e && e->l != i->to.val) {
				i->op = fault ? Oshr : Osar;
				i->arg[0] = TMP(e->l);
				i->arg[1] = getcon(k, fn);
				nrw++;
			}
			if (g1_findext(v, n, w, b))
				continue;
			if (n == cap) {
				cap *= 2;
				v = realloc(v, cap * sizeof v[0]);
				if (!v)
					die("out of memory");
			}
			v[n++] = (G1ext){w, i->to.val, b};
		}
	}
	free(v);
	if (nrw && getenv("OPENSNES_G1_TRACE"))
		fprintf(stderr, "g1idioms: %s: I1 rewrote %d\n", fn->name, nrw);
}

/* I2, scaled induction variable. A loop variable v (a w phi of the
 * header hd, one entry edge from pre, one latch) whose uses inside the
 * loop are only
 *     %e =l extuh v      with every use of %e a `mul %e, 2` (or
 *                        `mul 2, %e`, `shl %e, 1`, `add %e, %e`) in l,
 * and its own step
 *     %t =w add v, s     (or `add s, v`, or `sub v, s`; s a constant or
 *                        a value defined outside the loop)
 *     [%u =w and %t, m]  (m a constant; optional)
 * feeding the latch edge of the phi and nothing else, is carried scaled:
 * an l phi j = 2*extuh(v),
 *     entry   j0 = mul (extuh v0), 2           (the IR's 32-bit rule: up to
 *                                               $1FFFE, no range assumed)
 *     step    jt =l add j, 2*extuh(s)          (sub for sub)
 *             jn =l and jt, 2*m                (2*$FFFF = $1FFFE without
 *                                               a mask: the 16-bit wrap)
 * and each product becomes `copy j`. Exact for every value: for any w
 * operation f in add/sub, 2*extuh(f(v, s) mod 2^16) = (2v f 2s) mod 2^17,
 * and 2*(x & m) = (2x) & (2m); so j is 2*extuh(v) at every point, even,
 * below 2^17. A use of v outside the loop (its value after the loop, a
 * phi of an exit) reads v = `copy (shr j, 1)`, defined at the top of hd:
 * the low 16 bits of j >> 1 are v exactly. What is left of the old
 * variable (copies, extuh) has no use and goes with gcm's sweepdead.
 * Entry and scaled step are computed at the end of pre. Requires rpo,
 * pred, dom, use; maintains them (filluse after each rewrite). */

static BSet g1loop[1];

/* the value a chain of w copies of r comes from */
static Ref
g1_wsrc(Fn *fn, Ref r)
{
	Ins *d;
	int n;

	for (n=0; n<64 && rtype(r) == RTmp; n++) {
		d = fn->tmp[r.val].def;
		if (!d || d->op != Ocopy || d->cls != Kw)
			break;
		if (rtype(d->arg[0]) == RTmp
		&& fn->tmp[d->arg[0].val].cls != Kw)
			break;
		r = d->arg[0];
	}
	return r;
}

/* r, through single-use w copies, collected in cp (to go) */
static Ref
g1_chain(Fn *fn, Ref r, Ins ***cp, uint *ncp)
{
	Ins *d;
	int n;

	for (n=0; n<64 && rtype(r) == RTmp; n++) {
		d = fn->tmp[r.val].def;
		if (!d || d->op != Ocopy || d->cls != Kw
		|| rtype(d->arg[0]) != RTmp
		|| fn->tmp[d->arg[0].val].cls != Kw
		|| fn->tmp[r.val].nuse != 1)
			break;
		vgrow(cp, ++*ncp);
		(*cp)[*ncp-1] = d;
		r = d->arg[0];
	}
	return r;
}

/* m is 2*e in l */
static int
g1_isx2(Fn *fn, Ins *m, Ref e)
{
	int64_t c;

	if (m->cls != Kl)
		return 0;
	if (m->op == Omul)
		return (req(m->arg[0], e) && isconbits(fn, m->arg[1], &c) && c == 2)
			|| (req(m->arg[1], e) && isconbits(fn, m->arg[0], &c) && c == 2);
	if (m->op == Oshl)
		return req(m->arg[0], e) && isconbits(fn, m->arg[1], &c) && c == 1;
	if (m->op == Oadd)
		return req(m->arg[0], e) && req(m->arg[1], e);
	return 0;
}

/* an operand invariant in the loop: a number, or a w value defined
 * outside it; returned through copies */
static int
g1_inv(Fn *fn, Ref r, Ref *out)
{
	int64_t c;

	r = g1_wsrc(fn, r);
	*out = r;
	if (rtype(r) == RCon)
		return isconbits(fn, r, &c);
	return rtype(r) == RTmp && fn->tmp[r.val].cls == Kw
		&& !bshas(g1loop, fn->tmp[r.val].bid);
}

/* 2*extuh(r) at the end of pre (fault16: computed at 16 bits) */
static Ref
g1_x2(Fn *fn, Ref r, int fault16, Ins **pi, uint *npi)
{
	int64_t c;
	Ref e, x;

	if (rtype(r) == RCon) {
		isconbits(fn, r, &c);
		c &= 0xFFFF;
		return getcon(fault16 ? (2*c) & 0xFFFF : 2*c, fn);
	}
	e = newtmp("g1", fault16 ? Kw : Kl, fn);
	x = newtmp("g1", Kl, fn);
	vgrow(pi, *npi + 2);
	if (fault16) {
		(*pi)[(*npi)++] = (Ins){.op = Oadd, .cls = Kw, .to = e, .arg = {r, r}};
		(*pi)[(*npi)++] = (Ins){.op = Oextuh, .cls = Kl, .to = x, .arg = {e, R}};
	} else {
		(*pi)[(*npi)++] = (Ins){.op = Oextuh, .cls = Kl, .to = e, .arg = {r, R}};
		(*pi)[(*npi)++] = (Ins){.op = Omul, .cls = Kl, .to = x,
			.arg = {e, getcon(2, fn)}};
	}
	return x;
}

static int
g1_i2one(Fn *fn, Blk *hd, Blk *pre, Blk *latch, Phi *p, int fault)
{
	Ref v, r, s, mk, j, jt, jn, vl, j0, s2;
	Ins *a, *msk, *i, **cp, **sc, *pi, *nv;
	Use *u, *ue;
	Tmp *te;
	int64_t c, mask;
	uint ipre, ilat, ncp, nsc, npi, k, nm, q, n;
	int *mem;
	Blk *b;

	if (p->narg != 2 || p->cls != Kw)
		return 0;
	if (p->blk[0] == pre && p->blk[1] == latch)
		ipre = 0, ilat = 1;
	else if (p->blk[1] == pre && p->blk[0] == latch)
		ipre = 1, ilat = 0;
	else
		return 0;
	v = p->to;

	/* the step: latch value <- [copies] [and m] <- [copies] <- add/sub */
	cp = vnew(0, sizeof cp[0], PFn);
	ncp = 0;
	r = g1_chain(fn, p->arg[ilat], &cp, &ncp);
	if (rtype(r) != RTmp || fn->tmp[r.val].nuse != 1)
		return 0;
	msk = 0;
	mask = 0xFFFF;
	a = fn->tmp[r.val].def;
	if (a && a->op == Oand && a->cls == Kw) {
		msk = a;
		if (isconbits(fn, a->arg[1], &c))
			r = a->arg[0];
		else if (isconbits(fn, a->arg[0], &c))
			r = a->arg[1];
		else
			return 0;
		mask = c & 0xFFFF;
		r = g1_chain(fn, r, &cp, &ncp);
		if (rtype(r) != RTmp || fn->tmp[r.val].nuse != 1)
			return 0;
		a = fn->tmp[r.val].def;
	}
	if (!a || a->cls != Kw || !bshas(g1loop, fn->tmp[r.val].bid))
		return 0;
	if (a->op == Oadd) {
		if (!(req(g1_wsrc(fn, a->arg[0]), v) && g1_inv(fn, a->arg[1], &s))
		&& !(req(g1_wsrc(fn, a->arg[1]), v) && g1_inv(fn, a->arg[0], &s)))
			return 0;
	} else if (a->op == Osub) {
		if (!req(g1_wsrc(fn, a->arg[0]), v) || !g1_inv(fn, a->arg[1], &s))
			return 0;
	} else
		return 0;

	/* every use of v and of its copies: the products, the step, or
	 * outside the loop */
	mem = vnew(1, sizeof mem[0], PFn);
	mem[0] = v.val;
	nm = 1;
	sc = vnew(0, sizeof sc[0], PFn);
	nsc = 0;
	for (k=0; k<nm; k++) {
		te = &fn->tmp[mem[k]];
		for (u=te->use; u<&te->use[te->nuse]; u++) {
			if (u->type == UIns) {
				i = u->u.ins;
				if (i == a)
					continue;
				if (i->op == Ocopy && i->cls == Kw
				&& rtype(i->to) == RTmp) {
					for (q=0; q<nm; q++)
						if (mem[q] == i->to.val)
							break;
					if (q == nm) {
						vgrow(&mem, ++nm);
						mem[nm-1] = i->to.val;
					}
					continue;
				}
				if (!bshas(g1loop, u->bid))
					continue;
				if (i->op != Oextuh || i->cls != Kl
				|| rtype(i->to) != RTmp)
					return 0;
				for (ue=fn->tmp[i->to.val].use;
				ue<&fn->tmp[i->to.val].use[fn->tmp[i->to.val].nuse];
				ue++) {
					if (ue->type != UIns
					|| !g1_isx2(fn, ue->u.ins, i->to))
						return 0;
					vgrow(&sc, ++nsc);
					sc[nsc-1] = ue->u.ins;
				}
			} else if (bshas(g1loop, u->bid))
				return 0;
		}
	}
	if (nsc == 0)
		return 0;
	r = g1_wsrc(fn, p->arg[ipre]);
	if (rtype(r) == RCon && !isconbits(fn, r, &c))
		return 0;
	if (rtype(r) != RCon)
		r = p->arg[ipre];

	/* the rewrite */
	pi = vnew(0, sizeof pi[0], PFn);
	npi = 0;
	j = newtmp("g1", Kl, fn);
	jt = newtmp("g1", Kl, fn);
	jn = newtmp("g1", Kl, fn);
	vl = newtmp("g1", Kl, fn);
	j0 = g1_x2(fn, r, fault, &pi, &npi);
	s2 = g1_x2(fn, s, 0, &pi, &npi);
	mk = getcon(2 * mask, fn);
	for (k=0; k<nsc; k++)
		*sc[k] = (Ins){.op = Ocopy, .cls = Kl, .to = sc[k]->to, .arg = {j, R}};
	for (k=0; k<ncp; k++)
		*cp[k] = (Ins){.op = Onop};
	*a = (Ins){.op = a->op, .cls = Kl, .to = jt, .arg = {j, s2}};
	if (msk)
		*msk = (Ins){.op = Oand, .cls = Kl, .to = jn, .arg = {jt, mk}};
	else {
		/* the wrap of the 16-bit add, scaled: insert after the step */
		for (b=fn->start; b; b=b->link)
			if (a >= b->ins && a < &b->ins[b->nins])
				break;
		assert(b);
		nv = vnew(b->nins + 1, sizeof nv[0], PFn);
		n = 0;
		for (i=b->ins; i<&b->ins[b->nins]; i++) {
			nv[n++] = *i;
			if (i == a)
				nv[n++] = (Ins){.op = Oand, .cls = Kl, .to = jn,
					.arg = {jt, mk}};
		}
		idup(b, nv, n);
	}
	p->to = j;
	p->cls = Kl;
	p->arg[ipre] = j0;
	p->arg[ilat] = jn;
	/* entry and step at the end of pre */
	if (npi) {
		nv = vnew(pre->nins + npi, sizeof nv[0], PFn);
		icpy(nv, pre->ins, pre->nins);
		icpy(&nv[pre->nins], pi, npi);
		idup(pre, nv, pre->nins + npi);
	}
	/* v = low 16 bits of j >> 1, at the top of hd */
	nv = vnew(hd->nins + 2, sizeof nv[0], PFn);
	nv[0] = (Ins){.op = Oshr, .cls = Kl, .to = vl, .arg = {j, getcon(1, fn)}};
	nv[1] = (Ins){.op = Ocopy, .cls = Kw, .to = v, .arg = {vl, R}};
	icpy(&nv[2], hd->ins, hd->nins);
	idup(hd, nv, hd->nins + 2);
	return 1;
}

static void
g1_i2(Fn *fn, int fault)
{
	Blk *hd, *pre, *latch, *b, **stk;
	Phi *p;
	uint n, k, q, nlatch, npre, nstk;
	int again, nrw;

	nrw = 0;
	filluse(fn);
	bsinit(g1loop, fn->nblk);
	do {
		again = 0;
		for (n=0; n<fn->nblk && !again; n++) {
			hd = fn->rpo[n];
			latch = 0;
			nlatch = 0;
			for (k=0; k<hd->npred; k++)
				if (hd->pred[k]->id >= n) {
					latch = hd->pred[k];
					nlatch++;
				}
			if (nlatch != 1 || !dom(hd, latch))
				continue;
			/* the natural loop: what reaches the latch without hd */
			bszero(g1loop);
			bsset(g1loop, hd->id);
			stk = vnew(1, sizeof stk[0], PFn);
			nstk = 0;
			if (latch != hd) {
				bsset(g1loop, latch->id);
				stk[nstk++] = latch;
			}
			while (nstk) {
				b = stk[--nstk];
				for (q=0; q<b->npred; q++) {
					if (bshas(g1loop, b->pred[q]->id))
						continue;
					bsset(g1loop, b->pred[q]->id);
					vgrow(&stk, nstk + 1);
					stk[nstk++] = b->pred[q];
				}
			}
			pre = 0;
			npre = 0;
			for (k=0; k<hd->npred; k++)
				if (!bshas(g1loop, hd->pred[k]->id)) {
					pre = hd->pred[k];
					npre++;
				}
			if (npre != 1)
				continue;
			for (p=hd->phi; p; p=p->link)
				if (g1_i2one(fn, hd, pre, latch, p, fault)) {
					filluse(fn);
					nrw++;
					again = 1;
					break;
				}
		}
	} while (again);
	if (nrw && getenv("OPENSNES_G1_TRACE"))
		fprintf(stderr, "g1idioms: %s: I2 rewrote %d\n", fn->name, nrw);
}

/* The loops of I3 (and I4): a header hd with one latch (a block that hd
 * dominates and that jumps back to it), one entry edge from pre, the
 * natural loop in g1loop. Requires rpo, pred, dom. */
static int
g1_loop(Blk *hd, Blk **ppre, Blk **platch)
{
	Blk *latch, *pre, *b, **stk;
	uint k, q, nlatch, npre, nstk;

	latch = 0;
	nlatch = 0;
	for (k=0; k<hd->npred; k++)
		if (hd->pred[k]->id >= hd->id) {
			latch = hd->pred[k];
			nlatch++;
		}
	if (nlatch != 1 || !dom(hd, latch))
		return 0;
	bszero(g1loop);
	bsset(g1loop, hd->id);
	stk = vnew(1, sizeof stk[0], PFn);
	nstk = 0;
	if (latch != hd) {
		bsset(g1loop, latch->id);
		stk[nstk++] = latch;
	}
	while (nstk) {
		b = stk[--nstk];
		for (q=0; q<b->npred; q++) {
			if (bshas(g1loop, b->pred[q]->id))
				continue;
			bsset(g1loop, b->pred[q]->id);
			vgrow(&stk, nstk + 1);
			stk[nstk++] = b->pred[q];
		}
	}
	pre = 0;
	npre = 0;
	for (k=0; k<hd->npred; k++)
		if (!bshas(g1loop, hd->pred[k]->id)) {
			pre = hd->pred[k];
			npre++;
		}
	if (npre != 1)
		return 0;
	*ppre = pre;
	*platch = latch;
	return 1;
}

/* r is a constant mod 2^16 (no folding runs before this pass): a number,
 * through w copies, `neg` and the extensions of a number, folded at 16
 * bits (an extension to w of a 16- or 32-bit number keeps its low 16
 * bits; of a byte, the byte's rule) */
static int
g1_kon(Fn *fn, Ref r, int64_t *pc)
{
	Ins *d;
	int64_t c;

	r = g1_wsrc(fn, r);
	if (rtype(r) == RCon) {
		if (!isconbits(fn, r, &c))
			return 0;
		*pc = c & 0xFFFF;
		return 1;
	}
	if (rtype(r) != RTmp)
		return 0;
	d = fn->tmp[r.val].def;
	if (!d || d->cls != Kw || rtype(d->arg[0]) == RTmp
	|| !isconbits(fn, d->arg[0], &c))
		return 0;
	switch (d->op) {
	case Oneg: c = -c; break;
	case Oextsh: case Oextuh: case Oextsw: case Oextuw: break;
	case Oextsb: c = (int64_t)(int8_t)(c & 0xFF); break;
	case Oextub: c &= 0xFF; break;
	default: return 0;
	}
	*pc = c & 0xFFFF;
	return 1;
}

/* p, a w phi of hd with two arguments, is a counter: entry a constant
 * c0, latch value (through w copies) `add p, c`, `add c, p` or
 * `sub p, c`, c a constant. Gives the step instruction, the edge
 * indices, c0 and the step st, both mod 2^16 (sub: st = -c). */
static Ins *
g1_ctr(Fn *fn, Phi *p, Blk *pre, Blk *latch, uint *ipre, uint *ilat,
	int64_t *c0, int64_t *st)
{
	Ref r;
	Ins *a;
	int64_t c;

	if (p->narg != 2 || p->cls != Kw)
		return 0;
	if (p->blk[0] == pre && p->blk[1] == latch)
		*ipre = 0, *ilat = 1;
	else if (p->blk[1] == pre && p->blk[0] == latch)
		*ipre = 1, *ilat = 0;
	else
		return 0;
	if (!g1_kon(fn, p->arg[*ipre], c0))
		return 0;
	r = g1_wsrc(fn, p->arg[*ilat]);
	if (rtype(r) != RTmp)
		return 0;
	a = fn->tmp[r.val].def;
	if (!a || a->cls != Kw)
		return 0;
	if (a->op == Oadd) {
		if (req(g1_wsrc(fn, a->arg[0]), p->to)
		&& g1_kon(fn, a->arg[1], &c))
			;
		else if (req(g1_wsrc(fn, a->arg[1]), p->to)
		&& g1_kon(fn, a->arg[0], &c))
			;
		else
			return 0;
	} else if (a->op == Osub) {
		if (!req(g1_wsrc(fn, a->arg[0]), p->to)
		|| !g1_kon(fn, a->arg[1], &c))
			return 0;
		c = -c;
	} else
		return 0;
	*st = c & 0xFFFF;
	return a;
}

/* The test of hd: its jnz reads either a w compare made in hd of a
 * counter phi with a constant (`cultw %n, 64`, either side), or a
 * counter phi itself (nonzero). Gives the phi, the compare instruction
 * (0 for the bare jnz), the comparison (Ci*), the constant and whether
 * it is the left operand. */
static Phi *
g1_test(Fn *fn, Blk *hd, Ins **pcmp, int *pop, int64_t *pk, int *pswap)
{
	Ref r, x;
	Ins *d;
	Phi *p;
	int k, op;
	int64_t c;

	if (hd->jmp.type != Jjnz || rtype(hd->jmp.arg) != RTmp)
		return 0;
	r = hd->jmp.arg;
	d = fn->tmp[r.val].def;
	*pcmp = 0;
	if (d && fn->tmp[r.val].bid == hd->id && iscmp(d->op, &k, &op)) {
		if (k != Kw || d->cls != Kw)
			return 0;
		if (g1_kon(fn, d->arg[1], &c)) {
			x = g1_wsrc(fn, d->arg[0]);
			*pswap = 0;
		} else if (g1_kon(fn, d->arg[0], &c)) {
			x = g1_wsrc(fn, d->arg[1]);
			*pswap = 1;
		} else
			return 0;
		*pcmp = d;
		*pop = op;
		*pk = c & 0xFFFF;
	} else {
		x = g1_wsrc(fn, r);
		*pop = Cine;
		*pk = 0;
		*pswap = 0;
	}
	for (p=hd->phi; p; p=p->link)
		if (req(p->to, x))
			return p;
	return 0;
}

/* the comparison op of a and b, 16-bit words (fault: signed as unsigned) */
static int
g1_cmp(int op, int64_t a, int64_t b, int fault)
{
	int64_t sa, sb;

	a &= 0xFFFF;
	b &= 0xFFFF;
	sa = fault ? a : (a ^ 0x8000) - 0x8000;
	sb = fault ? b : (b ^ 0x8000) - 0x8000;
	switch (op) {
	case Cieq: return a == b;
	case Cine: return a != b;
	case Cisge: return sa >= sb;
	case Cisgt: return sa > sb;
	case Cisle: return sa <= sb;
	case Cislt: return sa < sb;
	case Ciuge: return a >= b;
	case Ciugt: return a > b;
	case Ciule: return a <= b;
	case Ciult: return a < b;
	}
	die("unreachable");
}

/* The trip count: the number of times the test of hd sends the
 * counter (c0, c0 + st, ... mod 2^16) into the loop (in1: on true)
 * before it leaves; -1 if it never leaves. Exact: the counter takes at
 * most 2^16 values, so a test that holds for the first 2^16 of them
 * holds forever. */
static int64_t
g1_trip(int op, int64_t k, int swap, int in1, int64_t c0, int64_t st,
	int fault)
{
	int64_t t, v;
	int c;

	for (t=0; t<0x10000; t++) {
		v = (c0 + t*st) & 0xFFFF;
		c = swap ? g1_cmp(op, k, v, fault) : g1_cmp(op, v, k, fault);
		if (c != in1)
			return t;
	}
	return -1;
}

/* the instructions a header may hold for I3: no memory, no call, no
 * side effect (they run once less, on entry, after the rotation) */
static int
g1_pure(Ins *i)
{
	int k, c;

	switch (i->op) {
	case Onop: case Ocopy:
	case Oadd: case Osub: case Oneg: case Omul:
	case Oand: case Oor: case Oxor:
	case Osar: case Oshr: case Oshl:
	case Oextsb: case Oextub: case Oextsh: case Oextuh:
	case Oextsw: case Oextuw:
		return 1;
	}
	return iscmp(i->op, &k, &c) && k != Ks && k != Kd;
}

static void
g1_ren(Ref *r, Ref from, Ref to)
{
	if (req(*r, from))
		*r = to;
}

/* I3, loop rotation. A loop (one latch, one entry edge from pre) whose
 * header hd ends with `jnz` on a test of a counter (g1_test, g1_ctr),
 * with a constant trip count of at least 1 (g1_trip), whose only exit
 * is hd's other edge, to a block X that only hd reaches, and whose
 * in-loop successor B of hd has hd as its only predecessor and no phi,
 * is entered at B (the test at the bottom):
 *     pre -> B; hd's phis p become, in B, p' = phi [pre: entry, hd: p],
 *     and in hd `p = copy latch value` (a phi of hd named there is p');
 *     every use of p in a loop block other than hd reads p'.
 * The first test holds (trip count >= 1), so skipping it on entry is
 * exact; hd's other instructions (g1_pure, no side effect) only feed hd
 * itself and what follows the loop, so they need not run on entry. On
 * every later pass the values are the same as before: along latch->hd,
 * p is the latch value; along hd->B, p' is p. X is still reached from
 * hd only, which every path out of the loop crosses, so a value of hd
 * read after the loop still dominates its use. OPENSNES_G1_FAULT=3 is
 * the provoked fault: the trip count is not looked at (a loop that
 * never runs is rotated, its body runs once). Requires rpo, pred, dom,
 * use; maintains them. */
static int
g1_i3one(Fn *fn, Blk *hd, Blk *pre, Blk *latch, int fault)
{
	Blk *b, *bb, *x, *s;
	Phi *pv, *p, *q, *np;
	Ins *cmp, *i, *nv;
	Use *u;
	Tmp *t;
	Ref *map;
	int op, swap, in1;
	int64_t k, c0, st, tc;
	uint ipre, ilat, n, nphi;

	pv = g1_test(fn, hd, &cmp, &op, &k, &swap);
	if (!pv || !g1_ctr(fn, pv, pre, latch, &ipre, &ilat, &c0, &st))
		return 0;
	in1 = bshas(g1loop, hd->s1->id);
	if (in1 == bshas(g1loop, hd->s2->id))
		return 0;
	bb = in1 ? hd->s1 : hd->s2;
	x = in1 ? hd->s2 : hd->s1;
	if (bb == hd || bb->npred != 1 || bb->phi || x->npred != 1)
		return 0;
	tc = g1_trip(op, k, swap, in1, c0, st, 0);
	if (!fault && tc < 1)
		return 0;
	/* the only exit is hd -> x */
	for (b=fn->start; b; b=b->link) {
		if (!bshas(g1loop, b->id) || b == hd)
			continue;
		if ((b->s1 && !bshas(g1loop, b->s1->id))
		|| (b->s2 && !bshas(g1loop, b->s2->id)))
			return 0;
	}
	/* hd's instructions: pure, read in hd or after the loop only */
	for (i=hd->ins; i<&hd->ins[hd->nins]; i++) {
		if (!g1_pure(i))
			return 0;
		if (rtype(i->to) != RTmp)
			continue;
		t = &fn->tmp[i->to.val];
		for (u=t->use; u<&t->use[t->nuse]; u++)
			if (bshas(g1loop, u->bid)
			&& (u->bid != hd->id || u->type == UPhi))
				return 0;
	}

	/* the rewrite */
	nphi = 0;
	for (p=hd->phi; p; p=p->link)
		nphi++;
	map = vnew(nphi, sizeof map[0], PFn);
	n = 0;
	for (p=hd->phi; p; p=p->link)
		map[n++] = newtmp("g1", p->cls, fn);
	/* uses of p in the loop below hd read p' */
	n = 0;
	for (p=hd->phi; p; p=p->link, n++) {
		t = &fn->tmp[p->to.val];
		for (u=t->use; u<&t->use[t->nuse]; u++) {
			if (!bshas(g1loop, u->bid) || u->bid == hd->id)
				continue;
			if (u->type == UIns) {
				g1_ren(&u->u.ins->arg[0], p->to, map[n]);
				g1_ren(&u->u.ins->arg[1], p->to, map[n]);
			} else if (u->type == UJmp) {
				g1_ren(&fn->rpo[u->bid]->jmp.arg, p->to, map[n]);
			} else if (u->type == UPhi) {
				for (q=u->u.phi, ipre=0; ipre<q->narg; ipre++)
					g1_ren(&q->arg[ipre], p->to, map[n]);
			}
		}
	}
	/* hd: p = copy (latch value, a phi of hd read as p'); B: p' */
	nv = vnew(nphi + hd->nins, sizeof nv[0], PFn);
	n = 0;
	for (p=hd->phi; p; p=p->link, n++) {
		ilat = p->blk[0] == latch ? 0 : 1;
		ipre = 1 - ilat;
		assert(p->narg == 2 && p->blk[ilat] == latch
			&& p->blk[ipre] == pre);
		nv[n] = (Ins){.op = Ocopy, .cls = p->cls, .to = p->to,
			.arg = {p->arg[ilat], R}};
		for (q=hd->phi, nphi=0; q; q=q->link, nphi++)
			g1_ren(&nv[n].arg[0], q->to, map[nphi]);
		np = alloc(sizeof *np);
		np->cls = p->cls;
		np->to = map[n];
		np->narg = 2;
		np->arg = vnew(2, sizeof np->arg[0], PFn);
		np->blk = vnew(2, sizeof np->blk[0], PFn);
		np->arg[0] = p->arg[ipre];
		np->blk[0] = pre;
		np->arg[1] = p->to;
		np->blk[1] = hd;
		np->link = bb->phi;
		bb->phi = np;
	}
	icpy(&nv[n], hd->ins, hd->nins);
	idup(hd, nv, n + hd->nins);
	hd->phi = 0;
	/* enter at B */
	s = pre;
	if (s->s1 == hd)
		s->s1 = bb;
	if (s->s2 == hd)
		s->s2 = bb;
	return 1;
}

static void
g1_i3(Fn *fn, int fault)
{
	Blk *hd, *pre, *latch;
	uint n;
	int again, nrw;

	nrw = 0;
	filluse(fn);
	do {
		again = 0;
		bsinit(g1loop, fn->nblk);
		for (n=0; n<fn->nblk && !again; n++) {
			hd = fn->rpo[n];
			if (!g1_loop(hd, &pre, &latch))
				continue;
			if (g1_i3one(fn, hd, pre, latch, fault)) {
				fillcfg(fn);
				filldom(fn);
				filluse(fn);
				nrw++;
				again = 1;
			}
		}
	} while (again);
	if (nrw && getenv("OPENSNES_G1_TRACE"))
		fprintf(stderr, "g1idioms: %s: I3 rewrote %d\n", fn->name, nrw);
}

/* I4, induction reversal. A counter n of a loop (one latch, one entry
 * edge; g1_test, g1_ctr: entry c0, step st, the header's test against a
 * constant, one of the header's edges in the loop and one out of it)
 * whose only uses, through w copies, are its own step (whose result,
 * through single-use copies, feeds the latch edge of n's phi and nothing
 * else) and the header's test (a compare read by the jnz alone, or the
 * jnz itself), with a trip count T (g1_trip: 0 to 65535, the loop
 * leaving), is counted down to zero:
 *     n = phi [pre: T, latch: n'],  n' = sub n, 1,  jnz n (true: the loop)
 * Exact: on the k-th arrival at the header (k = 0, 1, ...) the old test
 * holds for k < T and fails at k = T, the new n is T - k, nonzero for
 * k < T and zero at k = T; nothing else read n, so nothing else sees the
 * change (other exits of the loop leave it as before). OPENSNES_G1_FAULT=4
 * is the provoked fault: the trip count computed with the signed
 * compares read as unsigned (a counter from -5 to 5 counts 0 trips).
 * Requires rpo, pred, dom, use; maintains them. */
static int
g1_i4one(Fn *fn, Blk *hd, Blk *pre, Blk *latch, int fault)
{
	Phi *pv;
	Ins *cmp, *a, *i, **cp;
	Use *u;
	Tmp *te;
	Blk *t;
	Ref r;
	int op, swap, in1, *mem, sawa, sawt;
	int64_t k, c0, st, tc;
	uint ipre, ilat, ncp, nm, m, q;

	pv = g1_test(fn, hd, &cmp, &op, &k, &swap);
	if (!pv)
		return 0;
	a = g1_ctr(fn, pv, pre, latch, &ipre, &ilat, &c0, &st);
	if (!a)
		return 0;
	in1 = bshas(g1loop, hd->s1->id);
	if (in1 == bshas(g1loop, hd->s2->id))
		return 0;
	if (!cmp && in1 && st == 0xFFFF)
		return 0;       /* counted down to zero already */
	if (cmp && fn->tmp[cmp->to.val].nuse != 1)
		return 0;
	/* the step feeds the latch edge alone */
	cp = vnew(0, sizeof cp[0], PFn);
	ncp = 0;
	r = g1_chain(fn, pv->arg[ilat], &cp, &ncp);
	if (!req(r, a->to) || fn->tmp[a->to.val].nuse != 1)
		return 0;
	/* n and its copies: read by copies, the step and the test only */
	mem = vnew(1, sizeof mem[0], PFn);
	mem[0] = pv->to.val;
	nm = 1;
	sawa = sawt = 0;
	for (m=0; m<nm; m++) {
		te = &fn->tmp[mem[m]];
		for (u=te->use; u<&te->use[te->nuse]; u++) {
			if (u->type == UJmp) {
				if (cmp || u->bid != hd->id)
					return 0;
				sawt = 1;
				continue;
			}
			if (u->type != UIns)
				return 0;
			i = u->u.ins;
			if (i == a) {
				sawa = 1;
				continue;
			}
			if (i == cmp) {
				sawt = 1;
				continue;
			}
			if (i->op != Ocopy || i->cls != Kw
			|| rtype(i->to) != RTmp)
				return 0;
			for (q=0; q<nm; q++)
				if (mem[q] == i->to.val)
					break;
			if (q == nm) {
				vgrow(&mem, ++nm);
				mem[nm-1] = i->to.val;
			}
		}
	}
	if (!sawa || !sawt)
		return 0;
	tc = g1_trip(op, k, swap, in1, c0, st, fault);
	if (tc < 0)
		return 0;

	/* the rewrite */
	for (q=0; q<ncp; q++)
		*cp[q] = (Ins){.op = Onop};
	*a = (Ins){.op = Osub, .cls = Kw, .to = a->to,
		.arg = {pv->to, getcon(1, fn)}};
	pv->arg[ipre] = getcon(tc, fn);
	pv->arg[ilat] = a->to;
	if (cmp)
		*cmp = (Ins){.op = Onop};
	hd->jmp.arg = pv->to;
	if (!in1) {
		t = hd->s1;
		hd->s1 = hd->s2;
		hd->s2 = t;
	}
	return 1;
}

static void
g1_i4(Fn *fn, int fault)
{
	Blk *hd, *pre, *latch;
	uint n;
	int again, nrw;

	nrw = 0;
	filluse(fn);
	do {
		again = 0;
		bsinit(g1loop, fn->nblk);
		for (n=0; n<fn->nblk && !again; n++) {
			hd = fn->rpo[n];
			if (!g1_loop(hd, &pre, &latch))
				continue;
			if (g1_i4one(fn, hd, pre, latch, fault)) {
				fillcfg(fn);
				filldom(fn);
				filluse(fn);
				nrw++;
				again = 1;
			}
		}
	} while (again);
	if (nrw && getenv("OPENSNES_G1_TRACE"))
		fprintf(stderr, "g1idioms: %s: I4 rewrote %d\n", fn->name, nrw);
}

/* I5, the two-entry loop. A loop (one latch, one entry edge from pre,
 * header hd with phis) whose iteration starts in a block B (hd itself,
 * or hd's in-loop successor when hd only tests a counter) with pure
 * instructions (g1_pure) and then a load of
 *     %a =l add $S, %r      (either order, $S a symbol, any offset)
 * a far access (the IR's `cst` or `farram` keyword: a const or FAR
 * object, whose access the backend makes bank-honouring, `lda.l S,x`),
 * where %r is, through l copies, a phi P of hd with
 *     entry   j0 = 2*extuh(%w)      (`mul e, 2`, `mul 2, e`, `shl e, 1`
 *                                    or `add e, e`, e = `extuh %w`, %w a
 *                                    w temp: up to $1FFFE, 17 bits)
 *     latch   jl = and %x, K        (K a constant, 0..$FFFF)
 * is entered after that load: the first load is peeled onto the entry
 * edge, in two forms chosen by the carry out of the scaling, bit 15 of
 * %w (the 17th bit of j0):
 *     E:  lo =w add %w, %w;  xe =l extuh lo;  c =w csltw %w, 0;
 *         jnz c, F2, F1
 *     F1: s1 = load ($S + xe)            F2: s2 = load ($S+$10000 + xe)
 *     both jump to R, B after its load, whose phis are each phi p of hd
 *     as p' = [F1, F2: entry, T: p] and s = [F1: s1, F2: s2, T: s'];
 *     T, the new block on hd's way into B, holds the instructions of B
 *     the load needs (read by B's prefix and the load alone) and the load
 *     (its result s'); B's other prefix instructions (pure, read after
 *     the load) stay in B, after it, and read p'; hd keeps
 *     `p = copy latch value` (a phi of hd named there is p') and its test.
 *     A pure instruction nothing reads (what I2 leaves of the old
 *     variable, which gcm's sweepdead would remove) goes, and its reads
 *     do not count.
 * Exact for every value: $S + j0 = $S + $10000*bit15(%w) + lo, the sum
 * the IR says, and either form adds a 16-bit index that does not
 * overflow (the backend's `lda.l S,x` adds X as an unsigned word); the
 * load T makes on every later pass reads P after hd, a latch value
 * `and %x, K`, below $10000 by the IR's own `and`. No range fact is
 * used: the IR alone says j0 is 17 bits and jl 16. When B is hd's
 * successor, the test the entry skips holds (the trip count of the
 * counter, g1_trip, at least 1, as I3); hd's instructions are pure and
 * read only in hd, in what moves to T (after hd) or after the loop,
 * whose only exit is hd's; uses of p in the loop below hd
 * read p'. When B is hd, hd is first split into its phis (H0, jumping to
 * B) and B: nothing to skip, and every use of p but those moved to T
 * (after H0) reads p'. The volatile load runs once per
 * pass as before: on the entry edge it comes before pure instructions
 * only. OPENSNES_G1_FAULT=5 is the provoked fault: the second form reads
 * $S, not $S+$10000 (the carry dropped: the range assumption refused,
 * wrong when %w >= $8000). Requires rpo, pred, dom, use; maintains them
 * (fillcfg, filldom, filluse after each rewrite). Not for a plain object
 * (neither keyword): its access is `lda.w S,x` under the data bank, a
 * 16-bit operand that has no second form (S+$10000 does not assemble),
 * and the backend reads a plain address in bank $00. */

static int g1_i5seq;

/* r through l copies */
static Ref
g1_lsrc(Fn *fn, Ref r)
{
	Ins *d;
	int n;

	for (n=0; n<64 && rtype(r) == RTmp; n++) {
		d = fn->tmp[r.val].def;
		if (!d || d->op != Ocopy || d->cls != Kl
		|| rtype(d->arg[0]) != RTmp
		|| fn->tmp[d->arg[0].val].cls != Kl)
			break;
		r = d->arg[0];
	}
	return r;
}

/* r is 2*extuh(w), w a w temp: gives w */
static int
g1_x2of(Fn *fn, Ref r, Ref *pw)
{
	Ins *m, *e;
	uint k;

	r = g1_lsrc(fn, r);
	if (rtype(r) != RTmp || !(m = fn->tmp[r.val].def))
		return 0;
	for (k=0; k<2; k++) {
		if (rtype(m->arg[k]) != RTmp || !g1_isx2(fn, m, m->arg[k]))
			continue;
		e = fn->tmp[m->arg[k].val].def;
		if (e && e->op == Oextuh && e->cls == Kl
		&& rtype(e->arg[0]) == RTmp
		&& fn->tmp[e->arg[0].val].cls == Kw) {
			*pw = e->arg[0];
			return 1;
		}
	}
	return 0;
}

/* r is `and x, K` in l, 0 <= K <= $FFFF */
static int
g1_and16(Fn *fn, Ref r)
{
	Ins *a;
	int64_t c;

	r = g1_lsrc(fn, r);
	if (rtype(r) != RTmp || !(a = fn->tmp[r.val].def)
	|| a->op != Oand || a->cls != Kl)
		return 0;
	return (isconbits(fn, a->arg[1], &c) || isconbits(fn, a->arg[0], &c))
		&& c >= 0 && c <= 0xFFFF;
}

/* a pure instruction nothing reads (what I2 leaves of the old
 * variable, which gcm's sweepdead would remove): it goes with the
 * rewrite, and its reads do not count */
static int
g1_dead(Fn *fn, Ins *i)
{
	return g1_pure(i) && i->op != Onop && rtype(i->to) == RTmp
		&& fn->tmp[i->to.val].nuse == 0;
}

static Blk *
g1_blk(Fn *fn, char *tag, Blk *after)
{
	Blk *b;

	b = newblk();
	snprintf(b->name, NString, "g1i5_%s%d", tag, g1_i5seq);
	b->link = after->link;
	after->link = b;
	(void)fn;
	return b;
}

static Phi *
g1_phi(int cls, Ref to, uint n)
{
	Phi *p;

	p = alloc(sizeof *p);
	p->cls = cls;
	p->to = to;
	p->narg = n;
	p->arg = vnew(n, sizeof p->arg[0], PFn);
	p->blk = vnew(n, sizeof p->blk[0], PFn);
	return p;
}

static void
g1_redir(Blk *b, Blk *from, Blk *to)
{
	if (b->s1 == from)
		b->s1 = to;
	if (b->s2 == from)
		b->s2 = to;
}

static int
g1_i5one(Fn *fn, Blk *hd, Blk *pre, Blk *latch, int fault)
{
	Blk *bb, *x, *b, *h0, *e, *f1, *f2, *t;
	Phi *p, *pv, *q, *np, *ps;
	Ins *i, *l, *ad, *cmp, *nv, li, **dead;
	Use *u;
	Tmp *tt;
	Ref r, base, w, lo, xe, c, a1, a2, s1, s2, st, *map;
	Con k2;
	int op, swap, in1, rot, keep, *inT;
	int64_t k, c0, step, tc;
	uint ipre, ilat, n, nphi, nl, m, ndead;

	if (!hd->phi || (pre->s1 == hd && pre->s2 == hd))
		return 0;
	dead = vnew(0, sizeof dead[0], PFn);
	ndead = 0;
	/* B: hd itself when its pure instructions end on a load, else the
	 * in-loop successor of hd's counter test */
	for (i=hd->ins; i<&hd->ins[hd->nins]; i++)
		if (!g1_pure(i))
			break;
	rot = i < &hd->ins[hd->nins];
	bb = hd;
	if (!rot) {
		if (hd->jmp.type != Jjnz)
			return 0;
		pv = g1_test(fn, hd, &cmp, &op, &k, &swap);
		if (!pv || !g1_ctr(fn, pv, pre, latch, &ipre, &ilat, &c0, &step))
			return 0;
		in1 = bshas(g1loop, hd->s1->id);
		if (in1 == bshas(g1loop, hd->s2->id))
			return 0;
		bb = in1 ? hd->s1 : hd->s2;
		x = in1 ? hd->s2 : hd->s1;
		if (bb == hd || bb->npred != 1 || bb->phi || x->npred != 1)
			return 0;
		tc = g1_trip(op, k, swap, in1, c0, step, 0);
		if (tc < 1)
			return 0;
		for (b=fn->start; b; b=b->link) {
			if (!bshas(g1loop, b->id) || b == hd)
				continue;
			if ((b->s1 && !bshas(g1loop, b->s1->id))
			|| (b->s2 && !bshas(g1loop, b->s2->id)))
				return 0;
		}
	}
	/* B's pure prefix, then the load */
	for (l=bb->ins; l<&bb->ins[bb->nins]; l++)
		if (!g1_pure(l))
			break;
	if (l == &bb->ins[bb->nins] || !isload(l->op)
	|| rtype(l->arg[0]) != RTmp || rtype(l->to) != RTmp
	|| (l->volat & 6) == 0)
		return 0;
	nl = l - bb->ins;
	ad = fn->tmp[l->arg[0].val].def;
	if (!ad || ad->op != Oadd || ad->cls != Kl
	|| ad < bb->ins || ad >= l)
		return 0;
	if (rtype(ad->arg[0]) == RCon && fn->con[ad->arg[0].val].type == CAddr)
		base = ad->arg[0], r = ad->arg[1];
	else if (rtype(ad->arg[1]) == RCon
	&& fn->con[ad->arg[1].val].type == CAddr)
		base = ad->arg[1], r = ad->arg[0];
	else
		return 0;
	r = g1_lsrc(fn, r);
	for (pv=hd->phi; pv; pv=pv->link)
		if (req(pv->to, r))
			break;
	if (!pv || pv->cls != Kl || pv->narg != 2)
		return 0;
	ipre = pv->blk[0] == pre ? 0 : 1;
	ilat = 1 - ipre;
	if (pv->blk[ipre] != pre || pv->blk[ilat] != latch)
		return 0;
	if (!g1_x2of(fn, pv->arg[ipre], &w) || !g1_and16(fn, pv->arg[ilat]))
		return 0;
	/* the prefix splits: what the load needs (read by the prefix and
	 * the load only, and not by what stays) goes to T; the rest (pure,
	 * read after the load) stays in B after it, where it runs once a
	 * pass as before */
	inT = vnew(nl + 1, sizeof inT[0], PFn);
	for (n=nl+1; n-- > 0;) {
		i = &bb->ins[n];
		inT[n] = 1;
		if (n == nl || rtype(i->to) != RTmp)
			continue;
		tt = &fn->tmp[i->to.val];
		for (u=tt->use; u<&tt->use[tt->nuse]; u++) {
			if (u->type == UIns && u->bid == bb->id
			&& u->u.ins > i && u->u.ins <= l)
				inT[n] &= inT[u->u.ins - bb->ins];
			else if (u->type == UIns && g1_dead(fn, u->u.ins)) {
				vgrow(&dead, ++ndead);
				dead[ndead-1] = u->u.ins;
			} else
				inT[n] = 0;
		}
	}
	/* what goes to T reads nothing that stays */
	for (n=0; n<=nl; n++) {
		if (!inT[n])
			continue;
		for (m=0; m<2; m++) {
			r = bb->ins[n].arg[m];
			if (rtype(r) != RTmp)
				continue;
			i = fn->tmp[r.val].def;
			if (i && i >= bb->ins && i < l && !inT[i - bb->ins])
				return 0;
		}
	}
	if (!inT[ad - bb->ins])
		return 0;
	/* hd's instructions (not rot): read in hd, in the prefix, or after
	 * the loop */
	if (!rot)
		for (i=hd->ins; i<&hd->ins[hd->nins]; i++) {
			if (rtype(i->to) != RTmp)
				continue;
			tt = &fn->tmp[i->to.val];
			for (u=tt->use; u<&tt->use[tt->nuse]; u++) {
				if (!bshas(g1loop, u->bid))
					continue;
				if (u->type == UPhi)
					return 0;
				if (u->bid == hd->id)
					continue;
				if (u->type == UIns && u->bid == bb->id
				&& u->u.ins >= bb->ins && u->u.ins <= l
				&& inT[u->u.ins - bb->ins])
					continue;
				/* a pure instruction nothing reads (what I2
				 * leaves of the old variable): goes */
				if (u->type == UIns && g1_dead(fn, u->u.ins)) {
					vgrow(&dead, ++ndead);
					dead[ndead-1] = u->u.ins;
					continue;
				}
				return 0;
			}
		}
	/* the rewrite. 1: uses of p read p' */
	for (n=0; n<ndead; n++)
		*dead[n] = (Ins){.op = Onop};
	g1_i5seq++;
	nphi = 0;
	for (p=hd->phi; p; p=p->link)
		nphi++;
	map = vnew(nphi, sizeof map[0], PFn);
	for (p=hd->phi, n=0; p; p=p->link, n++)
		map[n] = newtmp("g1", p->cls, fn);
	for (p=hd->phi, n=0; p; p=p->link, n++) {
		tt = &fn->tmp[p->to.val];
		for (u=tt->use; u<&tt->use[tt->nuse]; u++) {
			if (u->type == UPhi) {
				for (q=hd->phi; q; q=q->link)
					if (q == u->u.phi)
						break;
				if (q)
					continue;       /* hd's own: below */
			}
			keep = 0;
			if (!rot && u->bid == hd->id)
				keep = 1;
			if (u->type == UIns && u->bid == bb->id
			&& u->u.ins >= bb->ins && u->u.ins <= l
			&& inT[u->u.ins - bb->ins])
				keep = 1;
			if (keep || (!rot && !bshas(g1loop, u->bid)))
				continue;
			if (u->type == UIns) {
				g1_ren(&u->u.ins->arg[0], p->to, map[n]);
				g1_ren(&u->u.ins->arg[1], p->to, map[n]);
			} else if (u->type == UJmp) {
				g1_ren(&fn->rpo[u->bid]->jmp.arg, p->to, map[n]);
			} else if (u->type == UPhi) {
				for (q=u->u.phi, m=0; m<q->narg; m++)
					g1_ren(&q->arg[m], p->to, map[n]);
			}
		}
	}
	/* 2: B = hd: hd's phis go to H0, which jumps to B */
	if (rot) {
		h0 = g1_blk(fn, "h", pre);
		h0->phi = hd->phi;
		hd->phi = 0;
		h0->jmp.type = Jjmp;
		h0->s1 = hd;
		g1_redir(pre, hd, h0);
		g1_redir(latch, hd, h0);        /* latch may be hd (= B) */
		hd = h0;
	}
	/* 3: E, F1, F2 on the entry edge */
	lo = newtmp("g1", Kw, fn);
	xe = newtmp("g1", Kl, fn);
	c = newtmp("g1", Kw, fn);
	a1 = newtmp("g1", Kl, fn);
	a2 = newtmp("g1", Kl, fn);
	s1 = newtmp("g1", l->cls, fn);
	s2 = newtmp("g1", l->cls, fn);
	st = newtmp("g1", l->cls, fn);
	e = g1_blk(fn, "e", pre);
	f1 = g1_blk(fn, "f", e);
	f2 = g1_blk(fn, "g", f1);
	nv = vnew(3, sizeof nv[0], PFn);
	nv[0] = (Ins){.op = Oadd, .cls = Kw, .to = lo, .arg = {w, w}};
	nv[1] = (Ins){.op = Oextuh, .cls = Kl, .to = xe, .arg = {lo, R}};
	nv[2] = (Ins){.op = Ocsltw, .cls = Kw, .to = c,
		.arg = {w, getcon(0, fn)}};
	idup(e, nv, 3);
	e->jmp.type = Jjnz;
	e->jmp.arg = c;
	e->s1 = f2;
	e->s2 = f1;
	k2 = fn->con[base.val];
	if (!fault)
		k2.bits.i += 0x10000;
	li = *l;
	nv = vnew(2, sizeof nv[0], PFn);
	nv[0] = (Ins){.op = Oadd, .cls = Kl, .to = a1, .arg = {base, xe}};
	nv[1] = li;
	nv[1].to = s1;
	nv[1].arg[0] = a1;
	idup(f1, nv, 2);
	nv[0] = (Ins){.op = Oadd, .cls = Kl, .to = a2,
		.arg = {newcon(&k2, fn), xe}};
	nv[1] = li;
	nv[1].to = s2;
	nv[1].arg[0] = a2;
	idup(f2, nv, 2);
	f1->jmp.type = f2->jmp.type = Jjmp;
	f1->s1 = f2->s1 = bb;
	g1_redir(pre, hd, e);
	/* 4: T, hd's way into B, with what the load needs and the load */
	t = g1_blk(fn, "t", hd);
	nv = vnew(nl + 1, sizeof nv[0], PFn);
	for (n=0, m=0; n<=nl; n++)
		if (inT[n])
			nv[m++] = bb->ins[n];
	nv[m-1].to = st;
	idup(t, nv, m);
	t->jmp.type = Jjmp;
	t->s1 = bb;
	g1_redir(hd, bb, t);
	nv = vnew(bb->nins, sizeof nv[0], PFn);
	for (n=0, m=0; n<bb->nins; n++)
		if (n > nl || !inT[n])
			nv[m++] = bb->ins[n];
	idup(bb, nv, m);
	/* 5: B's phis p' and s; hd: p = copy latch value */
	ps = g1_phi(li.cls, li.to, 3);
	ps->arg[0] = s1; ps->blk[0] = f1;
	ps->arg[1] = s2; ps->blk[1] = f2;
	ps->arg[2] = st; ps->blk[2] = t;
	bb->phi = ps;
	nv = vnew(nphi + hd->nins, sizeof nv[0], PFn);
	for (p=hd->phi, n=0; p; p=p->link, n++) {
		ilat = p->blk[0] == latch ? 0 : 1;
		ipre = 1 - ilat;
		assert(p->narg == 2 && p->blk[ilat] == latch
			&& p->blk[ipre] == pre);
		nv[n] = (Ins){.op = Ocopy, .cls = p->cls, .to = p->to,
			.arg = {p->arg[ilat], R}};
		for (q=hd->phi, m=0; q; q=q->link, m++)
			g1_ren(&nv[n].arg[0], q->to, map[m]);
		np = g1_phi(p->cls, map[n], 3);
		np->arg[0] = np->arg[1] = p->arg[ipre];
		np->blk[0] = f1;
		np->blk[1] = f2;
		np->arg[2] = p->to;
		np->blk[2] = t;
		np->link = bb->phi;
		bb->phi = np;
	}
	icpy(&nv[n], hd->ins, hd->nins);
	idup(hd, nv, n + hd->nins);
	hd->phi = 0;
	return 1;
}

static void
g1_i5(Fn *fn, int fault)
{
	Blk *hd, *pre, *latch;
	uint n;
	int again, nrw;

	nrw = 0;
	filluse(fn);
	do {
		again = 0;
		bsinit(g1loop, fn->nblk);
		for (n=0; n<fn->nblk && !again; n++) {
			hd = fn->rpo[n];
			if (!g1_loop(hd, &pre, &latch))
				continue;
			if (g1_i5one(fn, hd, pre, latch, fault)) {
				fillcfg(fn);
				filldom(fn);
				filluse(fn);
				nrw++;
				again = 1;
			}
		}
	} while (again);
	if (nrw && getenv("OPENSNES_G1_TRACE"))
		fprintf(stderr, "g1idioms: %s: I5 rewrote %d\n", fn->name, nrw);
}

void
g1idioms(Fn *fn)
{
	static int on = -1, fault;

	if (on < 0) {
		char *s = getenv("OPENSNES_G1_IDIOMS");
		char *f = getenv("OPENSNES_G1_FAULT");
		on = s && s[0] == '1' && T.wordsz == 2;
		fault = f ? f[0] - '0' : 0;
	}
	if (!on)
		return;
	g1_i1(fn, fault == 1);
	g1_i2(fn, fault == 2);
	g1_i4(fn, fault == 4);
	g1_i3(fn, fault == 3);
	g1_i5(fn, fault == 5);
}
