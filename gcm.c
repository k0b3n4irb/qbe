#include "all.h"

#define NOBID (-1u)

static int
isdivwl(Ins *i)
{
	switch (i->op) {
	case Odiv:
	case Orem:
	case Oudiv:
	case Ourem:
		return KBASE(i->cls) == 0;
	default:
		return 0;
	}
}

int
pinned(Ins *i)
{
	return optab[i->op].pinned || isdivwl(i);
}

/* pinned ins that can be eliminated if unused */
static int
canelim(Ins *i)
{
	/* OpenSNES patch (chantier A2, 2026-05-09): a volatile load is
	 * a SIDE EFFECT (the C standard says each access reads memory
	 * even if the result is unused) and cannot be elided here even
	 * when its result temp has no users. Stay pinned-in-place. */
	if (i->volat & 1)
		return 0;
	return isload(i->op) || isalloc(i->op) || isdivwl(i);
}

static uint earlyins(Fn *, Blk *, Ins *);

static uint
schedearly(Fn *fn, Ref r)
{
	Tmp *t;
	Blk *b;

	if (rtype(r) != RTmp)
		return 0;

	t = &fn->tmp[r.val];
	if (t->gcmbid != NOBID)
		return t->gcmbid;

	b = fn->rpo[t->bid];
	if (t->def) {
		assert(b->ins <= t->def && t->def < &b->ins[b->nins]);
		t->gcmbid = 0;  /* mark as visiting */
		t->gcmbid = earlyins(fn, b, t->def);
	} else {
		/* phis do not move */
		t->gcmbid = t->bid;
	}

	return t->gcmbid;
}

static uint
earlyins(Fn *fn, Blk *b, Ins *i)
{
	uint b0, b1;

	b0 = schedearly(fn, i->arg[0]);
	assert(b0 != NOBID);
	b1 = schedearly(fn, i->arg[1]);
	assert(b1 != NOBID);
	if (fn->rpo[b0]->depth < fn->rpo[b1]->depth) {
		assert(dom(fn->rpo[b0], fn->rpo[b1]));
		b0 = b1;
	}
	return pinned(i) ? b->id : b0;
}

static void
earlyblk(Fn *fn, uint bid)
{
	Blk *b;
	Phi *p;
	Ins *i;
	uint n;

	b = fn->rpo[bid];
	for (p=b->phi; p; p=p->link)
		for (n=0; n<p->narg; n++)
			schedearly(fn, p->arg[n]);
	for (i=b->ins; i<&b->ins[b->nins]; i++)
		if (pinned(i)) {
			schedearly(fn, i->arg[0]);
			schedearly(fn, i->arg[1]);
		}
	schedearly(fn, b->jmp.arg);
}

/* least common ancestor in dom tree */
static uint
lcabid(Fn *fn, uint bid1, uint bid2)
{
	Blk *b;

	if (bid1 == NOBID)
		return bid2;
	if (bid2 == NOBID)
		return bid1;

	b = lca(fn->rpo[bid1], fn->rpo[bid2]);
	assert(b);
	return b->id;
}

static uint
bestbid(Fn *fn, uint earlybid, uint latebid)
{
	Blk *curb, *earlyb, *bestb;

	if (latebid == NOBID)
		return NOBID; /* unused */

	assert(earlybid != NOBID);

	earlyb = fn->rpo[earlybid];
	bestb = curb = fn->rpo[latebid];
	assert(dom(earlyb, curb));

	while (curb != earlyb) {
		curb = curb->idom;
		if (curb->loop < bestb->loop)
			bestb = curb;
	}
	return bestb->id;
}

static uint lateins(Fn *, Blk *, Ins *, Ref r);
static uint latephi(Fn *, Phi *, Ref r);
static uint latejmp(Blk *, Ref r);

/* return lca bid of ref uses */
static uint
schedlate(Fn *fn, Ref r)
{
	Tmp *t;
	Blk *b;
	Use *u;
	uint earlybid;
	uint latebid;
	uint uselatebid;

	if (rtype(r) != RTmp)
		return NOBID;

	t = &fn->tmp[r.val];
	if (t->visit)
		return t->gcmbid;

	t->visit = 1;
	earlybid = t->gcmbid;
	if (earlybid == NOBID)
		return NOBID; /* not used */

	/* reuse gcmbid for late bid */
	t->gcmbid = t->bid;
	latebid = NOBID;
	for (u=t->use; u<&t->use[t->nuse]; u++) {
		assert(u->bid < fn->nblk);
		b = fn->rpo[u->bid];
		switch (u->type) {
		case UXXX:
			die("unreachable");
			break;
		case UPhi:
			uselatebid = latephi(fn, u->u.phi, r);
			break;
		case UIns:
			uselatebid = lateins(fn, b, u->u.ins, r);
			break;
		case UJmp:
			uselatebid = latejmp(b, r);
			break;
		}
		latebid = lcabid(fn, latebid, uselatebid);
	}
	/* latebid may be NOBID if the temp is used
	 * in fixed instructions that may be eliminated
	 * and are themselves unused transitively */

	if (t->def && !pinned(t->def))
		t->gcmbid = bestbid(fn, earlybid, latebid);
	/* else, keep the early one */

	/* now, gcmbid is the best bid */
	return t->gcmbid;
}

/* returns lca bid of uses or NOBID if
 * the definition can be eliminated */
static uint
lateins(Fn *fn, Blk *b, Ins *i, Ref r)
{
	uint latebid;

	assert(b->ins <= i && i < &b->ins[b->nins]);
	assert(req(i->arg[0], r) || req(i->arg[1], r));

	latebid = schedlate(fn, i->to);
	if (pinned(i)) {
		if (latebid == NOBID)
		if (canelim(i))
			return NOBID;
		return b->id;
	}

	return latebid;
}

static uint
latephi(Fn *fn, Phi *p, Ref r)
{
	uint n;
	uint latebid;

	if (!p->narg)
		return NOBID; /* marked as unused */

	latebid = NOBID;
	for (n = 0; n < p->narg; n++)
		if (req(p->arg[n], r))
			latebid = lcabid(fn, latebid, p->blk[n]->id);

	assert(latebid != NOBID);
	return latebid;
}

static uint
latejmp(Blk *b, Ref r)
{
	if (req(b->jmp.arg, R))
		return NOBID;
	else {
		assert(req(b->jmp.arg, r));
		return b->id;
	}
}

static void
lateblk(Fn *fn, uint bid)
{
	Blk *b;
	Phi **pp;
	Ins *i;

	b = fn->rpo[bid];
	for (pp=&b->phi; *(pp);)
		if (schedlate(fn, (*pp)->to) == NOBID) {
			(*pp)->narg = 0; /* mark unused */
			*pp = (*pp)->link; /* remove phi */
		} else
			pp = &(*pp)->link;

	for (i=b->ins; i<&b->ins[b->nins]; i++)
		if (pinned(i))
			schedlate(fn, i->to);
}

static void
addgcmins(Fn *fn, Ins *vins, uint nins)
{
	Ins *i;
	Tmp *t;
	Blk *b;

	for (i=vins; i<&vins[nins]; i++) {
		assert(rtype(i->to) == RTmp);
		t = &fn->tmp[i->to.val];
		b = fn->rpo[t->gcmbid];
		addins(&b->ins, &b->nins, i);
	}
}

/* move live instructions to the
 * end of their target block; use-
 * before-def errors are fixed by
 * schedblk */
static void
gcmmove(Fn *fn)
{
	Tmp *t;
	Ins *vins, *i;
	uint nins;

	nins = 0;
	vins = vnew(nins, sizeof vins[0], PFn);

	for (t=fn->tmp; t<&fn->tmp[fn->ntmp]; t++) {
		if (t->def == 0)
			continue;
		if (t->bid == t->gcmbid)
			continue;
		i = t->def;
		if (pinned(i) && !canelim(i))
			continue;
		assert(rtype(i->to) == RTmp);
		assert(t == &fn->tmp[i->to.val]);
		if (t->gcmbid != NOBID)
			addins(&vins, &nins, i);
		*i = (Ins){.op = Onop};
	}
	addgcmins(fn, vins, nins);
}

/* dfs ordering */
static Ins *
schedins(Fn *fn, Blk *b, Ins *i, Ins **pvins, uint *pnins)
{
	Ins *i0, *i1;
	Tmp *t;
	uint n;

	igroup(b, i, &i0, &i1);
	for (i=i0; i<i1; i++)
		for (n=0; n<2; n++) {
			if (rtype(i->arg[n]) != RTmp)
				continue;
			t = &fn->tmp[i->arg[n].val];
			if (t->bid != b->id || !t->def)
				continue;
			schedins(fn, b, t->def, pvins, pnins);
		}
	for (i=i0; i<i1; i++) {
		addins(pvins, pnins, i);
		*i = (Ins){.op = Onop};
	}
	return i1;
}

/* order ins within a block */
static void
schedblk(Fn *fn)
{
	Blk *b;
	Ins *i, *vins;
	uint nins;

	vins = vnew(0, sizeof vins[0], PHeap);
	for (b=fn->start; b; b=b->link) {
		nins = 0;
		for (i=b->ins; i<&b->ins[b->nins];)
			i = schedins(fn, b, i, &vins, &nins);
		idup(b, vins, nins);
	}
	vfree(vins);
}

static int
cheap(Ins *i)
{
	int x;

	if (KBASE(i->cls) != 0)
		return 0;
	switch (i->op) {
	case Oneg:
	case Oadd:
	case Osub:
	case Omul:
	case Oand:
	case Oor:
	case Oxor:
	case Osar:
	case Oshr:
	case Oshl:
		return 1;
	default:
		return iscmp(i->op, &x, &x);
	}
}

static void
sinkref(Fn *fn, Blk *b, Ref *pr)
{
	Ins i, *e;
	Tmp *t;
	Ref r;

	if (rtype(*pr) != RTmp)
		return;
	t = &fn->tmp[pr->val];
	if (!t->def
	|| t->bid == b->id
	|| pinned(t->def)
	|| !cheap(t->def))
		return;

	/* sink t->def to b */
	i = *t->def;
	r = newtmp("snk", t->cls, fn);
	t = 0;  /* invalidated */
	*pr = r;
	i.to = r;
	fn->tmp[r.val].gcmbid = b->id;
	emiti(i);
	/* The operands are NOT sunk with it, unless QBE_SINK_DEEP is set.
	 * Upstream recurses on the local `i`, so its emitted copy keeps the
	 * original operands anyway and the sunk operands are referenced by
	 * nothing. On 2026-10-08 this fork made the recursion real (on the
	 * emitted copy, curi): the dead copies went, but every load and store
	 * of `a[i]` then carried its own `mul i, 2`, which this target emits
	 * as `lda i / asl a / tax` in front of each access — a dozen times
	 * per iteration in a loop over parallel arrays. Sinking the address
	 * alone keeps it next to its access (where the emitter folds it into
	 * the addressing mode) and leaves the index computed once. Measured
	 * on a game's match logic: 75.1 M master cycles to 68.3 M over the
	 * same 340 ticks. (OpenSNES, 2026-10-10) */
	if (getenv("QBE_SINK_DEEP")) {
		e = curi;
		sinkref(fn, b, &e->arg[0]);
		sinkref(fn, b, &e->arg[1]);
	}
}

/* Remove what sink() left without a use: the original of an instruction
 * whose every use was sunk, and anything that only fed it. Upstream leaves
 * them to spill/rega; a target that skips those passes would emit them —
 * on w65816, 258 of the 496 cycles of an insertion sort's inner iteration
 * were 32-bit index and address values stored and never read. Same
 * predicate as gcmmove() for what may go (a call, a store, a volatile load
 * stay). An alloc stays too: the w65816 abi0 pass has already recorded its
 * size. requires use; maintains use. (OpenSNES, 2026-10-08) */
static void
sweepdead(Fn *fn)
{
	Blk *b;
	Ins *i;
	int again;

	do {
		again = 0;
		for (b=fn->start; b; b=b->link)
			for (i=b->ins; i<&b->ins[b->nins]; i++) {
				if (i->op == Onop || rtype(i->to) != RTmp)
					continue;
				if (fn->tmp[i->to.val].nuse != 0)
					continue;
				if (isalloc(i->op) || (pinned(i) && !canelim(i)))
					continue;
				*i = (Ins){.op = Onop};
				again = 1;
			}
		if (again)
			filluse(fn);
	} while (again);
}

/* redistribute trivial ops to point of
 * use to reduce register pressure
 * requires rpo, use; breaks use
 */
static void
sink(Fn *fn)
{
	Blk *b;
	Ins *i;

	for (b=fn->start; b; b=b->link) {
		for (i=b->ins; i<&b->ins[b->nins]; i++)
			if (isload(i->op))
				sinkref(fn, b, &i->arg[0]);
			else if (isstore(i->op))
				sinkref(fn, b, &i->arg[1]);
		sinkref(fn, b, &b->jmp.arg);
	}
	addgcmins(fn, curi, &insb[NIns] - curi);
}

/* requires use dom
 * maintains rpo pred dom
 * breaks use
 */
void
gcm(Fn *fn)
{
	Tmp *t;
	uint bid;

	filldepth(fn);
	fillloop(fn);

	for (t=fn->tmp; t<&fn->tmp[fn->ntmp]; t++) {
		t->visit = 0;
		t->gcmbid = NOBID;
	}
	for (bid=0; bid<fn->nblk; bid++)
		earlyblk(fn, bid);
	for (bid=0; bid<fn->nblk; bid++)
		lateblk(fn, bid);

	gcmmove(fn);
	filluse(fn);
	curi = &insb[NIns];
	sink(fn);
	filluse(fn);
	sweepdead(fn);
	schedblk(fn);
	
	if (debug['G']) {
		fprintf(stderr, "\n> After GCM:\n");
		printfn(fn, stderr);
	}
}
