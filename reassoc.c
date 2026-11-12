#include "all.h"

/* OpenSNES (2026-10-16): the constant of a bounded index folded into the
 * symbol it indexes.
 *
 * `buf[i + 8]` with `buf` an array of 16-bit words reaches here as
 *
 *     %a =w add %x, 8
 *     %e =l extsh %a          (or extuh)
 *     %m =l mul %e, 2         (or shl %e, 1)
 *     %p =l add $buf, %m
 *
 * next to `buf[i]`'s `%e0 =l extuh %x / %m0 =l mul %e0, 2`. The backend
 * then computes (x + 8) * 2 into X for the second access, where
 * `lda.l buf+16,x` on the first access's X would do. When x is known to
 * lie in [0, B] and B + 8 <= 0x7FFF, x + 8 cannot wrap at 16 bits nor
 * read as negative, so ext(x + 8) * k == ext(x) * k + 8 * k exactly, and
 * the chain is rewritten in place:
 *
 *     %e =l extuh %x          (extub %b when x is `extub %b`)
 *     %m =l mul %e, 2
 *     %p =l add $buf+16, %m
 *
 * gvn, which runs next, makes %e and %m the first access's. The symbol
 * may come through `%t =l add $buf, c` (a union member, `u.w[i + 8]`). The bound is
 * PROVED from x's definition, through copies — `extub` (255), `and x,
 * M` (M), `shr x, s` (0xFFFF >> s) — or, since 2026-11-11 (A6a R4), from
 * a test `cultw x, N` that dominates the add (testbound: a loop's counter
 * in its body): the compiler does not reason from an out-of-bounds
 * access being undefined (owner decision, 2026-10-16). %e and %m must have
 * no other use; the old add goes when %e was its only use (else it is
 * left to gcm, which drops it when unused).
 * QBE_NO_REASSOC=1 switches the pass off, QBE_NO_REASSOC_LOOP=1 the loop
 * bound alone. requires use, dom; maintains use. */

static int
cbits(Fn *fn, Ref r, int64_t *v)
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

static Ins *
def1(Fn *fn, Ref r)
{
	Tmp *t;

	if (rtype(r) != RTmp || r.val < Tmp0)
		return 0;
	t = &fn->tmp[r.val];
	if (t->ndef != 1 || !t->def)
		return 0;
	return t->def;
}

/* an upper bound of the 16-bit value r, proved from its definition; -1 */
static int64_t
bound(Fn *fn, Ref r)
{
	Ins *d;
	int64_t v;

	d = def1(fn, r);
	for (int n = 0; d && d->op == Ocopy && n < 8; n++)
		d = def1(fn, d->arg[0]);      /* `%i =w copy %m` */
	if (!d || d->cls != Kw)
		return -1;
	switch (d->op) {
	case Oextub:
		return 0xFF;
	case Oand:
		if (cbits(fn, d->arg[1], &v) || cbits(fn, d->arg[0], &v))
			return v >= 0 ? (v & 0xFFFF) : -1;
		return -1;
	case Oshr:
		if (cbits(fn, d->arg[1], &v) && v > 0 && v < 16)
			return 0xFFFF >> v;
		return -1;
	default:
		return -1;
	}
}

/* r through its copies */
static Ref
rootof(Fn *fn, Ref r)
{
	Ins *d;

	for (int n = 0; n < 8 && (d = def1(fn, r)) && d->op == Ocopy; n++)
		r = d->arg[0];
	return r;
}

/* A6a R4: an upper bound of the 16-bit value r at block `at`, proved by a
 * test that dominates it: a block h ending in `jnz %c, @s, ...` with
 * %c = `cultw r, N` (or `culew r, N`), @s entered from h alone and
 * dominating `at`. Every way to `at` then took the true edge of that test,
 * and r, an SSA value, is the one tested: r <= N - 1 (N). A loop's counter
 * in its body, the shape of `for (i = 0; i < N; i += 8) buf[i + 1] = v`.
 * Like bound(), nothing is inferred from an access being out of bounds.
 * -1 when no such test. requires dom. */
static int64_t
testbound(Fn *fn, Ref r, Blk *at)
{
	Blk *h, *s;
	Ins *d;
	int64_t n, v, best;

	if (getenv("QBE_NO_REASSOC_LOOP"))
		return -1;
	r = rootof(fn, r);
	best = -1;
	for (h = fn->start; h; h = h->link) {
		if (h->jmp.type != Jjnz || rtype(h->jmp.arg) != RTmp)
			continue;
		d = def1(fn, rootof(fn, h->jmp.arg));
		if (!d || (d->op != Ocultw && d->op != Oculew)
		    || !req(rootof(fn, d->arg[0]), r) || !cbits(fn, d->arg[1], &n))
			continue;
		s = h->s1;
		if (!s || s == h->s2 || s->npred != 1 || s->pred[0] != h || !dom(s, at))
			continue;
		v = d->op == Ocultw ? n - 1 : n;
		if (v < 0 || v > 0xFFFF)
			continue;
		if (best < 0 || v < best)
			best = v;
	}
	return best;
}

/* the address r stands for, when it is a symbol plus a constant: the
 * symbol itself, or `%t =l add $sym, c` (a union member, `u.w[x]`) */
static int
symof(Fn *fn, Ref r, Con *out)
{
	Ins *d;
	int64_t v;
	int j;

	if (rtype(r) == RCon && fn->con[r.val].type == CAddr) {
		*out = fn->con[r.val];
		return 1;
	}
	d = def1(fn, r);
	if (!d || d->op != Oadd || d->cls != Kl)
		return 0;
	for (j = 0; j < 2; j++)
		if (rtype(d->arg[j]) == RCon && fn->con[d->arg[j].val].type == CAddr
		    && cbits(fn, d->arg[1 - j], &v)) {
			*out = fn->con[d->arg[j].val];
			out->bits.i += v;
			return 1;
		}
	return 0;
}

void
reassoc(Fn *fn)
{
	Blk *b;
	Ins *i, *m, *e, *a, *xd;
	Ref mt, x;
	Con c;
	int64_t k, cst, bx;
	int s, done;

	if (getenv("QBE_NO_REASSOC"))
		return;
	done = 0;
	for (b = fn->start; b; b = b->link)
		for (i = b->ins; i < &b->ins[b->nins]; i++) {
			if (i->op != Oadd || i->cls != Kl)
				continue;
			s = symof(fn, i->arg[0], &c) ? 0 : symof(fn, i->arg[1], &c) ? 1 : -1;
			if (s < 0)
				continue;
			mt = i->arg[1 - s];
			m = def1(fn, mt);
			if (!m || m->cls != Kl || fn->tmp[mt.val].nuse != 1)
				continue;
			if (m->op == Omul && cbits(fn, m->arg[1], &k))
				;
			else if (m->op == Oshl && cbits(fn, m->arg[1], &k) && k >= 0 && k < 8)
				k = (int64_t)1 << k;
			else
				continue;
			if (k <= 0 || k > 16)
				continue;
			e = def1(fn, m->arg[0]);
			if (!e || e->cls != Kl || fn->tmp[m->arg[0].val].nuse != 1)
				continue;
			if (e->op != Oextsh && e->op != Oextuh && e->op != Oextsw && e->op != Oextuw)
				continue;
			a = def1(fn, e->arg[0]);
			if (!a || a->op != Oadd || a->cls != Kw)
				continue;
			if (cbits(fn, a->arg[1], &cst))
				x = a->arg[0];
			else if (cbits(fn, a->arg[0], &cst))
				x = a->arg[1];
			else
				continue;
			if (cst <= 0 || cst > 0x7FFF || rtype(x) != RTmp)
				continue;
			bx = bound(fn, x);
			if (bx < 0 && fn->tmp[e->arg[0].val].bid < fn->nblk)
				bx = testbound(fn, x, fn->rpo[fn->tmp[e->arg[0].val].bid]);
			if (bx < 0 || bx + cst > 0x7FFF)
				continue;
			/* rewrite: e = ext(x) (extub b when x is extub b), sym += cst * k.
			 * The old add goes now when e was its only use (A6a R4): left
			 * for gcm, it was still a use of x when ivar looked, and a
			 * loop counter with one more use is not rewritten. */
			if (fn->tmp[e->arg[0].val].nuse == 1)
				*a = (Ins){.op = Onop};
			xd = def1(fn, x);
			if (xd && xd->op == Oextub) {
				e->op = Oextub;
				e->arg[0] = xd->arg[0];
			} else {
				e->op = Oextuh;
				e->arg[0] = x;
			}
			c.bits.i += cst * k;
			i->arg[s] = newcon(&c, fn);
			done++;
		}
	if (done)
		filluse(fn);
}
