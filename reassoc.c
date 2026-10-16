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
 * gvn, which runs next, makes %e and %m the first access's. The bound is
 * PROVED from x's definition only, through copies — `extub` (255), `and x,
 * M` (M), `shr x, s` (0xFFFF >> s): the compiler does not reason from an out-of-bounds
 * access being undefined (owner decision, 2026-10-16). %e and %m must have
 * no other use; the old add is left to gcm, which drops it when unused.
 * QBE_NO_REASSOC=1 switches the pass off. requires use; maintains it. */

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

void
reassoc(Fn *fn)
{
	Blk *b;
	Ins *i, *m, *e, *a, *xd;
	Ref sym, mt, x;
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
			s = rtype(i->arg[0]) == RCon && fn->con[i->arg[0].val].type == CAddr ? 0
			  : rtype(i->arg[1]) == RCon && fn->con[i->arg[1].val].type == CAddr ? 1 : -1;
			if (s < 0)
				continue;
			sym = i->arg[s];
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
			if (bx < 0 || bx + cst > 0x7FFF)
				continue;
			/* rewrite: e = ext(x) (extub b when x is extub b), sym += cst * k */
			xd = def1(fn, x);
			if (xd && xd->op == Oextub) {
				e->op = Oextub;
				e->arg[0] = xd->arg[0];
			} else {
				e->op = Oextuh;
				e->arg[0] = x;
			}
			c = fn->con[sym.val];
			c.bits.i += cst * k;
			i->arg[s] = newcon(&c, fn);
			done++;
		}
	if (done)
		filluse(fn);
}
