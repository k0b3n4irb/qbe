/* OpenSNES: the exact IR idioms of the G1 lot (I1 so far).
 *
 * Switch: OPENSNES_G1_IDIOMS=1 turns the pass on; it is OFF by default,
 * and off it does not touch the function (the output is byte for byte
 * the one without the pass). Only for a target whose `w` is 16 bits
 * (T.wordsz == 2, w65816): the rule below reads extsh / extsw of a `w`
 * as the sign extension of the whole word, which holds only there.
 * OPENSNES_G1_FAULT=1 is the provoked fault the test must see: the
 * rewritten value takes a logical shift (`shr`), so its high word is 0
 * where it should be the sign mask $FFFF.
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

void
g1idioms(Fn *fn)
{
	static int on = -1, fault;
	Blk *b;
	Ins *i, *d;
	G1ext *v, *e;
	uint n, cap, bi;
	int64_t k;
	int w, nrw;

	if (on < 0) {
		char *s = getenv("OPENSNES_G1_IDIOMS");
		char *f = getenv("OPENSNES_G1_FAULT");
		on = s && s[0] == '1' && T.wordsz == 2;
		fault = f && f[0] == '1';
	}
	if (!on)
		return;
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
