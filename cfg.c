#include "all.h"

Blk *
newblk(void)
{
	static Blk z;
	Blk *b;

	b = alloc(sizeof *b);
	*b = z;
	b->ins = vnew(0, sizeof b->ins[0], PFn);
	b->pred = vnew(0, sizeof b->pred[0], PFn);
	return b;
}

static void
fixphis(Fn *f)
{
	Blk *b;
	Phi *p;
	uint n, n0;

	for (b=f->start; b; b=b->link) {
		assert(b->id < f->nblk);
		for (p=b->phi; p; p=p->link) {
			for (n=n0=0; n<p->narg; n++)
				if (p->blk[n]->id != -1u) {
					p->blk[n0] = p->blk[n];
					p->arg[n0] = p->arg[n];
					n0++;
				}
			assert(n0 > 0);
			p->narg = n0;
		}
	}
}

static void
addpred(Blk *bp, Blk *b)
{
	vgrow(&b->pred, ++b->npred);
	b->pred[b->npred-1] = bp;
}

void
fillpreds(Fn *f)
{
	Blk *b;

	for (b=f->start; b; b=b->link)
		b->npred = 0;
	for (b=f->start; b; b=b->link) {
		if (b->s1)
			addpred(b, b->s1);
		if (b->s2 && b->s2 != b->s1)
			addpred(b, b->s2);
	}
}

static void
porec(Blk *b, uint *npo)
{
	Blk *s1, *s2;

	if (!b || b->id != -1u)
		return;
	b->id = 0; /* marker */
	s1 = b->s1;
	s2 = b->s2;
	if (s1 && s2 && s1->loop > s2->loop) {
		s1 = b->s2;
		s2 = b->s1;
	}
	porec(s1, npo);
	porec(s2, npo);
	b->id = (*npo)++;
}

static void
fillrpo(Fn *f)
{
	Blk *b, **p;

	for (b=f->start; b; b=b->link)
		b->id = -1u;
	f->nblk = 0;
	porec(f->start, &f->nblk);
	vgrow(&f->rpo, f->nblk);
	for (p=&f->start; (b=*p);) {
		if (b->id == -1u) {
			*p = b->link;
		} else {
			b->id = f->nblk-b->id-1;
			f->rpo[b->id] = b;
			p = &b->link;
		}
	}
}

/* fill rpo, preds; prune dead blks */
void
fillcfg(Fn *f)
{
	fillrpo(f);
	fillpreds(f);
	fixphis(f);
}

/* for dominators computation, read
 * "A Simple, Fast Dominance Algorithm"
 * by K. Cooper, T. Harvey, and K. Kennedy.
 */

static Blk *
inter(Blk *b1, Blk *b2)
{
	Blk *bt;

	if (b1 == 0)
		return b2;
	while (b1 != b2) {
		if (b1->id < b2->id) {
			bt = b1;
			b1 = b2;
			b2 = bt;
		}
		while (b1->id > b2->id) {
			b1 = b1->idom;
			assert(b1);
		}
	}
	return b1;
}

void
filldom(Fn *fn)
{
	Blk *b, *d;
	int ch;
	uint n, p;

	for (b=fn->start; b; b=b->link) {
		b->idom = 0;
		b->dom = 0;
		b->dlink = 0;
	}
	do {
		ch = 0;
		for (n=1; n<fn->nblk; n++) {
			b = fn->rpo[n];
			d = 0;
			for (p=0; p<b->npred; p++)
				if (b->pred[p]->idom
				||  b->pred[p] == fn->start)
					d = inter(d, b->pred[p]);
			if (d != b->idom) {
				ch++;
				b->idom = d;
			}
		}
	} while (ch);
	for (b=fn->start; b; b=b->link)
		if ((d=b->idom)) {
			assert(d != b);
			b->dlink = d->dom;
			d->dom = b;
		}
}

int
sdom(Blk *b1, Blk *b2)
{
	assert(b1 && b2);
	if (b1 == b2)
		return 0;
	while (b2->id > b1->id)
		b2 = b2->idom;
	return b1 == b2;
}

int
dom(Blk *b1, Blk *b2)
{
	return b1 == b2 || sdom(b1, b2);
}

static void
addfron(Blk *a, Blk *b)
{
	uint n;

	for (n=0; n<a->nfron; n++)
		if (a->fron[n] == b)
			return;
	if (!a->nfron)
		a->fron = vnew(++a->nfron, sizeof a->fron[0], PFn);
	else
		vgrow(&a->fron, ++a->nfron);
	a->fron[a->nfron-1] = b;
}

/* fill the dominance frontier */
void
fillfron(Fn *fn)
{
	Blk *a, *b;

	for (b=fn->start; b; b=b->link)
		b->nfron = 0;
	for (b=fn->start; b; b=b->link) {
		if (b->s1)
			for (a=b; !sdom(a, b->s1); a=a->idom)
				addfron(a, b->s1);
		if (b->s2)
			for (a=b; !sdom(a, b->s2); a=a->idom)
				addfron(a, b->s2);
	}
}

static void
loopmark(Blk *hd, Blk *b, void f(Blk *, Blk *))
{
	uint p;

	if (b->id < hd->id || b->visit == hd->id)
		return;
	b->visit = hd->id;
	f(hd, b);
	for (p=0; p<b->npred; ++p)
		loopmark(hd, b->pred[p], f);
}

void
loopiter(Fn *fn, void f(Blk *, Blk *))
{
	uint n, p;
	Blk *b;

	for (b=fn->start; b; b=b->link)
		b->visit = -1u;
	for (n=0; n<fn->nblk; ++n) {
		b = fn->rpo[n];
		for (p=0; p<b->npred; ++p)
			if (b->pred[p]->id >= n)
				loopmark(b, b->pred[p], f);
	}
}

/* dominator tree depth */
void
filldepth(Fn *fn)
{
	Blk *b, *d;
	int depth;

	for (b=fn->start; b; b=b->link)
		b->depth = -1;

	fn->start->depth = 0;

	for (b=fn->start; b; b=b->link) {
		if (b->depth != -1)
			continue;
		depth = 1;
		for (d=b->idom; d->depth==-1; d=d->idom)
			depth++;
		depth += d->depth;
		b->depth = depth;
		for (d=b->idom; d->depth==-1; d=d->idom)
			d->depth = --depth;
	}
}

/* least common ancestor in dom tree */
Blk *
lca(Blk *b1, Blk *b2)
{
	if (!b1)
		return b2;
	if (!b2)
		return b1;
	while (b1->depth > b2->depth)
		b1 = b1->idom;
	while (b2->depth > b1->depth)
		b2 = b2->idom;
	while (b1 != b2) {
		b1 = b1->idom;
		b2 = b2->idom;
	}
	return b1;
}

void
multloop(Blk *hd, Blk *b)
{
	(void)hd;
	b->loop *= 10;
}

void
fillloop(Fn *fn)
{
	Blk *b;

	for (b=fn->start; b; b=b->link)
		b->loop = 1;
	loopiter(fn, multloop);
}

static void
uffind(Blk **pb, Blk **uf)
{
	Blk **pb1;

	pb1 = &uf[(*pb)->id];
	if (*pb1) {
		uffind(pb1, uf);
		*pb = *pb1;
	}
}

/* requires rpo and no phis, breaks cfg */
void
simpljmp(Fn *fn)
{

	Blk **uf; /* union-find */
	Blk **p, *b, *ret;

	ret = newblk();
	ret->id = fn->nblk++;
	ret->jmp.type = Jret0;
	uf = emalloc(fn->nblk * sizeof uf[0]);
	for (b=fn->start; b; b=b->link) {
		assert(!b->phi);
		if (b->jmp.type == Jret0) {
			b->jmp.type = Jjmp;
			b->s1 = ret;
		}
		if (b->nins == 0)
		if (b->jmp.type == Jjmp) {
			uffind(&b->s1, uf);
			if (b->s1 != b)
				uf[b->id] = b->s1;
		}
	}
	for (p=&fn->start; (b=*p); p=&b->link) {
		if (b->s1)
			uffind(&b->s1, uf);
		if (b->s2)
			uffind(&b->s2, uf);
		if (b->s1 && b->s1 == b->s2) {
			b->jmp.type = Jjmp;
			b->s2 = 0;
		}
	}
	*p = ret;
	free(uf);
}

static int
reachrec(Blk *b, Blk *to)
{
	if (b == to)
		return 1;
	if (!b || b->visit)
		return 0;

	b->visit = 1;
	if (reachrec(b->s1, to))
		return 1;
	if (reachrec(b->s2, to))
		return 1;

	return 0;
}

/* Blk.visit needs to be clear at entry */
int
reaches(Fn *fn, Blk *b, Blk *to)
{
	int r;

	assert(to);
	r = reachrec(b, to);
	for (b=fn->start; b; b=b->link)
		b->visit = 0;
	return r;
}

/* can b reach 'to' not through excl
 * Blk.visit needs to be clear at entry */
int
reachesnotvia(Fn *fn, Blk *b, Blk *to, Blk *excl)
{
	excl->visit = 1;
	return reaches(fn, b, to);
}

/* OpenSNES (2026-10-08): thread jumps through a block that only chooses.
 *
 * `a && b` in a condition reaches us as
 *
 *     @p1   jnz %a, @p2, @j
 *     @p2   %c =w cugtw ...        ; b
 *           jmp @j
 *     @j    %p =w phi @p1 0, @p2 %c
 *           jnz %p, @then, @else
 *
 * A target with a register allocator pays little for %p. One without
 * (w65816: every temp is a stack slot) stores 0 or 1 in each predecessor
 * and tests it again in @j, and %c can no longer be fused with a branch.
 * When @j holds nothing but that phi and the jnz on it, each predecessor
 * can take @j's decision itself: a constant argument goes straight to
 * @then or @else, a temp argument of a predecessor that ends in `jmp @j`
 * becomes that predecessor's own `jnz`. @j disappears when every
 * predecessor is gone (fillcfg prunes it).
 *
 * The phis of @then / @else gain, for each new predecessor, the value they
 * had for @j — which dominates the predecessor because it dominates @j and
 * is not defined in it.
 *
 * requires rpo pred use; breaks them (call fillcfg and filluse after) */
static Ref
phival(Phi *p, Blk *b)
{
	uint n;

	for (n=0; n<p->narg; n++)
		if (p->blk[n] == b)
			return p->arg[n];
	return R;
}

static void
phiadd(Phi *p, Blk *b, Ref r)
{
	p->narg++;
	vgrow(&p->arg, p->narg);
	vgrow(&p->blk, p->narg);
	p->arg[p->narg-1] = r;
	p->blk[p->narg-1] = b;
}

/* can the edge b->j be replaced by b->t (t a successor of j)? */
static int
canretarget(Blk *b, Blk *j, Blk *t)
{
	Phi *p;

	if (b->s1 != t && b->s2 != t)
		return 1;
	/* b already reaches t: one phi argument per predecessor, so the
	 * value arriving through j must be the one arriving directly */
	for (p=t->phi; p; p=p->link)
		if (!req(phival(p, b), phival(p, j)))
			return 0;
	return 1;
}

static void
retarget(Blk *b, Blk *j, Blk *t)
{
	Phi *p;

	if (b->s1 != t && b->s2 != t)
		for (p=t->phi; p; p=p->link)
			phiadd(p, b, phival(p, j));
	if (b->s1 == j)
		b->s1 = t;
	if (b->s2 == j)
		b->s2 = t;
	if (b->jmp.type == Jjnz && b->s1 == b->s2) {
		b->jmp.type = Jjmp;
		b->jmp.arg = R;
		b->s2 = 0;
	}
}

/* A phi whose arguments are all the same value (or itself) is that
 * value: remove it and rename its uses. Promotion leaves such phis at
 * the join of an `a || b` for every variable live across it, and one of
 * them is enough to hide a block that only chooses from threadjnz.
 * (gvn would remove them, but it runs after.) Returns the number removed. */
static void
rename1(Fn *fn, Ref from, Ref to)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint n;

	for (b=fn->start; b; b=b->link) {
		for (p=b->phi; p; p=p->link)
			for (n=0; n<p->narg; n++)
				if (req(p->arg[n], from))
					p->arg[n] = to;
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			for (n=0; n<2; n++)
				if (req(i->arg[n], from))
					i->arg[n] = to;
		if (req(b->jmp.arg, from))
			b->jmp.arg = to;
	}
}

static int
trivphis(Fn *fn)
{
	Blk *b;
	Phi *p, **pp;
	Ref r;
	uint n;
	int removed;

	removed = 0;
	for (b=fn->start; b; b=b->link)
		for (pp=&b->phi; (p=*pp);) {
			r = R;
			for (n=0; n<p->narg; n++) {
				if (req(p->arg[n], p->to))
					continue;
				if (req(r, R))
					r = p->arg[n];
				else if (!req(r, p->arg[n]))
					break;
			}
			if (n < p->narg || req(r, R)) {
				pp = &p->link;
				continue;
			}
			*pp = p->link;
			rename1(fn, p->to, r);
			removed++;
		}
	return removed;
}

/* The decision of j, when j only chooses: its one phi, and the two
 * targets for phi != 0 and phi == 0. j may hold the front end's boolean
 * conversion of the phi (`%c =w cnew %p, 0` or `ceqw`, then `jnz %c`). */
static Phi *
chooser(Fn *fn, Blk *j, Blk **pnz, Blk **pz)
{
	Phi *p;
	Ins *i, *d;
	Con *c;

	p = j->phi;
	if (!p || p->link || p->cls != Kw
	|| j->jmp.type != Jjnz || rtype(j->jmp.arg) != RTmp
	|| j->s1 == j || j->s2 == j || j->s1 == j->s2)
		return 0;
	d = 0;
	for (i=j->ins; i<&j->ins[j->nins]; i++) {
		if (i->op == Onop)
			continue;
		if (d)
			return 0;
		d = i;
	}
	if (!d) {
		if (!req(j->jmp.arg, p->to)
		|| fn->tmp[p->to.val].nuse != 1)
			return 0;
		*pnz = j->s1;
		*pz = j->s2;
		return p;
	}
	if ((d->op != Ocnew && d->op != Oceqw)
	|| !req(d->to, j->jmp.arg)
	|| !req(d->arg[0], p->to)
	|| rtype(d->arg[1]) != RCon
	|| fn->tmp[p->to.val].nuse != 1
	|| fn->tmp[d->to.val].nuse != 1)
		return 0;
	c = &fn->con[d->arg[1].val];
	if (c->type != CBits
	|| (T.wordsz == 2 ? (uint16_t)c->bits.i : (uint32_t)c->bits.i) != 0)
		return 0;
	*pnz = d->op == Ocnew ? j->s1 : j->s2;
	*pz = d->op == Ocnew ? j->s2 : j->s1;
	return p;
}

/* Runs on plain SSA, before gvn: after gvn a definition may sit in a
 * block this pass cuts off while its uses live on (gcm then asserts). */
int
threadjnz(Fn *fn)
{
	Blk *j, *b, *t, *snz, *sz;
	Phi *p, *q;
	Con *c;
	Ref a;
	uint n;
	int64_t v;
	int changed, done;

	if (trivphis(fn))
		return 1; /* use counts changed: come back */
	changed = 0;
	for (j=fn->start; j; j=j->link) {
		p = chooser(fn, j, &snz, &sz);
		if (getenv("QBE_DBG_THREAD") && j->phi && j->jmp.type == Jjnz)
			fprintf(stderr, "THREAD %s @%s: %s (phis=%d nins=%u nuse=%u)\n",
				fn->name, j->name, p ? "chooser" : "no",
				j->phi->link ? 2 : 1, j->nins,
				fn->tmp[j->phi->to.val].nuse);
		if (!p)
			continue;
		for (n=0; n<p->narg;) {
			b = p->blk[n];
			a = p->arg[n];
			done = 0;
			if (b == j || (b->s1 == j && b->s2 == j)) {
				n++;
				continue;
			}
			if (rtype(a) == RCon) {
				c = &fn->con[a.val];
				if (c->type == CBits) {
					v = c->bits.i;
					/* jnz tests a word */
					if (T.wordsz == 2)
						v = (uint16_t)v;
					else
						v = (uint32_t)v;
					t = v ? snz : sz;
					if (canretarget(b, j, t)) {
						retarget(b, j, t);
						done = 1;
					}
				}
			}
			else if (rtype(a) == RTmp
			&& b->jmp.type == Jjmp && b->s1 == j) {
				for (q=snz->phi; q; q=q->link)
					phiadd(q, b, phival(q, j));
				for (q=sz->phi; q; q=q->link)
					phiadd(q, b, phival(q, j));
				b->jmp.type = Jjnz;
				b->jmp.arg = a;
				b->s1 = snz;
				b->s2 = sz;
				done = 1;
			}
			if (done) {
				p->narg--;
				p->arg[n] = p->arg[p->narg];
				p->blk[n] = p->blk[p->narg];
				changed = 1;
			} else
				n++;
		}
		/* use counts are stale once a phi gained an argument */
		if (changed)
			return 1;
	}
	return 0;
}

/* OpenSNES (2026-10-08): `jnz (x != 0)` is `jnz x`, `jnz (x == 0)` is
 * `jnz x` with the targets swapped. The front end converts a condition to
 * a boolean before it branches on it; where the conversion survives, a
 * target without registers materialises 0 or 1 and tests it again.
 * Word compares only: jnz tests a word. The compare, left without a use,
 * is removed by gcm's sweep. requires use (for Tmp.def). */
void
simpljnz(Fn *fn)
{
	Blk *b, *t;
	Tmp *tmp;
	Ins *d;
	Con *c;

	for (b=fn->start; b; b=b->link) {
		for (;;) {
			if (b->jmp.type != Jjnz || rtype(b->jmp.arg) != RTmp)
				break;
			tmp = &fn->tmp[b->jmp.arg.val];
			d = tmp->def;
			if (!d || (d->op != Ocnew && d->op != Oceqw)
			|| rtype(d->arg[1]) != RCon)
				break;
			c = &fn->con[d->arg[1].val];
			if (c->type != CBits
			|| (T.wordsz == 2 ? (uint16_t)c->bits.i : (uint32_t)c->bits.i) != 0)
				break;
			if (rtype(d->arg[0]) != RTmp
			|| fn->tmp[d->arg[0].val].cls != Kw)
				break;
			b->jmp.arg = d->arg[0];
			if (d->op == Oceqw) {
				t = b->s1;
				b->s1 = b->s2;
				b->s2 = t;
			}
		}
	}
}

/* OpenSNES (2026-10-08): the compare a block branches on goes last in it.
 * gcm schedules a block by dependencies only, so the compare may be
 * followed by instructions hoisted from the successors; a backend that
 * fuses compare and branch only when they are adjacent (w65816) then
 * materialises 0 or 1 and tests it. Moving an instruction down its own
 * block is always valid when its one use is the block's jump.
 * requires use. */
void
cmplast(Fn *fn)
{
	Blk *b;
	Ins *i, *e, c;
	int x;

	for (b=fn->start; b; b=b->link) {
		if (b->jmp.type != Jjnz || rtype(b->jmp.arg) != RTmp
		|| b->nins < 2
		|| fn->tmp[b->jmp.arg.val].nuse != 1)
			continue;
		e = &b->ins[b->nins];
		for (i=b->ins; i<e-1; i++)
			if (req(i->to, b->jmp.arg))
				break;
		if (i == e-1 || !iscmp(i->op, &x, &x))
			continue;
		c = *i;
		memmove(i, i+1, (e-1-i) * sizeof *i);
		e[-1] = c;
	}
}
