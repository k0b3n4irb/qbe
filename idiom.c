/* OpenSNES: the exact IR idioms of the G1 lot (I1, I2).
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
 * assumption the rule refuses: bit 16 of 2*v0 is lost when v0 >= $8000.
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
}
