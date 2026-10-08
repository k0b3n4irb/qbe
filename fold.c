#include "all.h"

/* boring folding code */

int
foldint(Con *res, int op, int w, Con *cl, Con *cr)
{
	union {
		int64_t s;
		uint64_t u;
		float fs;
		double fd;
	} l, r;
	uint64_t x;
	Sym sym;
	int typ, bits;

	memset(&sym, 0, sizeof sym);
	typ = CBits;
	l.s = cl->bits.i;
	r.s = cr->bits.i;
	/* w65816 fork: on this target Kw is 16 bits wide and Kl 32 (T.wordsz;
	 * it has no 64-bit integer), while a constant sits in 64 bits in no single form: the
	 * front end prints a negative int sign-extended to 64 bits, an
	 * unsigned long zero-extended, and `neg`, `sub` or `mul` folded below
	 * leave whatever the 64-bit operation gave. Upstream divided, shifted
	 * right and compared at 32 and 64 bits, so -(4294967291UL) == 5 folded
	 * to 0 (the 64-bit value is -4294967291). Every operation whose result
	 * depends on the bits above the class width takes its operands through
	 * SX / ZX at the real width. (difftest, 2026-10-08) */
	bits = (w ? 16 : 8) * T.wordsz;   /* 16 and 32 here, 32 and 64 elsewhere */
#define SX(v, b) ((int64_t)((uint64_t)(v) << (64 - (b))) >> (64 - (b)))
#define ZX(v, b) ((b) >= 64 ? (uint64_t)(v) : (uint64_t)(v) & (((uint64_t)1 << (b)) - 1))
	if (op == Oadd) {
		if (cl->type == CAddr) {
			if (cr->type == CAddr)
				return 1;
			typ = CAddr;
			sym = cl->sym;
		}
		else if (cr->type == CAddr) {
			typ = CAddr;
			sym = cr->sym;
		}
	}
	else if (op == Osub) {
		if (cl->type == CAddr) {
			if (cr->type != CAddr) {
				typ = CAddr;
				sym = cl->sym;
			} else if (!symeq(cl->sym, cr->sym))
				return 1;
		}
		else if (cr->type == CAddr)
			return 1;
	}
	else if (cl->type == CAddr || cr->type == CAddr)
		return 1;
	if (op == Odiv || op == Orem || op == Oudiv || op == Ourem) {
		if (cr->type == CBits && ZX(r.u, bits) == 0)
			return 1;
		if (op == Odiv || op == Orem) {
			if (cr->type == CBits && SX(r.u, bits) == -1)
			if (cl->type == CBits
			&& SX(l.u, bits) == SX((uint64_t)1 << (bits - 1), bits))
				return 1;
		}
	}
	switch (op) {
	case Oadd:  x = l.u + r.u; break;
	case Osub:  x = l.u - r.u; break;
	case Oneg:  x = -l.u; break;
	case Odiv:  x = SX(l.u, bits) / SX(r.u, bits); break;
	case Orem:  x = SX(l.u, bits) % SX(r.u, bits); break;
	case Oudiv: x = ZX(l.u, bits) / ZX(r.u, bits); break;
	case Ourem: x = ZX(l.u, bits) % ZX(r.u, bits); break;
	case Omul:  x = l.u * r.u; break;
	case Oand:  x = l.u & r.u; break;
	case Oor:   x = l.u | r.u; break;
	case Oxor:  x = l.u ^ r.u; break;
	case Osar:  x = SX(l.u, bits) >> (r.u & (bits - 1)); break;
	case Oshr:  x = ZX(l.u, bits) >> (r.u & (bits - 1)); break;
	case Oshl:  x = l.u << (r.u & (bits - 1)); break;
	case Oextsb: x = (int8_t)l.u;   break;
	case Oextub: x = (uint8_t)l.u;  break;
	case Oextsh: x = (int16_t)l.u;  break;
	case Oextuh: x = (uint16_t)l.u; break;
	case Oextsw: x = (int32_t)l.u;  break;
	case Oextuw: x = (uint32_t)l.u; break;
	case Ostosi: x = w ? (int64_t)cl->bits.s : (int32_t)cl->bits.s; break;
	case Ostoui: x = w ? (uint64_t)cl->bits.s : (uint32_t)cl->bits.s; break;
	case Odtosi: x = w ? (int64_t)cl->bits.d : (int32_t)cl->bits.d; break;
	case Odtoui: x = w ? (uint64_t)cl->bits.d : (uint32_t)cl->bits.d; break;
	case Ocast:
		x = l.u;
		if (cl->type == CAddr) {
			typ = CAddr;
			sym = cl->sym;
		}
		break;
	default:
		if (Ocmpw <= op && op <= Ocmpl1) {
			if (op <= Ocmpw1)
				bits = 8 * T.wordsz;
			else {
				bits = 16 * T.wordsz;
				op -= Ocmpl - Ocmpw;
			}
			/* signed compares read .s, unsigned ones .u */
			switch (op - Ocmpw) {
			case Cisle: case Cislt: case Cisgt: case Cisge:
				l.s = SX(l.u, bits);
				r.s = SX(r.u, bits);
				break;
			default:
				l.u = ZX(l.u, bits);
				r.u = ZX(r.u, bits);
			}
			switch (op - Ocmpw) {
			case Ciule: x = l.u <= r.u; break;
			case Ciult: x = l.u < r.u;  break;
			case Cisle: x = l.s <= r.s; break;
			case Cislt: x = l.s < r.s;  break;
			case Cisgt: x = l.s > r.s;  break;
			case Cisge: x = l.s >= r.s; break;
			case Ciugt: x = l.u > r.u;  break;
			case Ciuge: x = l.u >= r.u; break;
			case Cieq:  x = l.u == r.u; break;
			case Cine:  x = l.u != r.u; break;
			default: die("unreachable");
			}
		}
		else if (Ocmps <= op && op <= Ocmps1) {
			switch (op - Ocmps) {
			case Cfle: x = l.fs <= r.fs; break;
			case Cflt: x = l.fs < r.fs;  break;
			case Cfgt: x = l.fs > r.fs;  break;
			case Cfge: x = l.fs >= r.fs; break;
			case Cfne: x = l.fs != r.fs; break;
			case Cfeq: x = l.fs == r.fs; break;
			case Cfo: x = l.fs < r.fs || l.fs >= r.fs; break;
			case Cfuo: x = !(l.fs < r.fs || l.fs >= r.fs); break;
			default: die("unreachable");
			}
		}
		else if (Ocmpd <= op && op <= Ocmpd1) {
			switch (op - Ocmpd) {
			case Cfle: x = l.fd <= r.fd; break;
			case Cflt: x = l.fd < r.fd;  break;
			case Cfgt: x = l.fd > r.fd;  break;
			case Cfge: x = l.fd >= r.fd; break;
			case Cfne: x = l.fd != r.fd; break;
			case Cfeq: x = l.fd == r.fd; break;
			case Cfo: x = l.fd < r.fd || l.fd >= r.fd; break;
			case Cfuo: x = !(l.fd < r.fd || l.fd >= r.fd); break;
			default: die("unreachable");
			}
		}
		else
			die("unreachable");
	}
	/* ...and the result is cut to its class: `-22016 * 256U` is 0 in 16
	 * bits, and left as 0xAA0000 a `jnz` on it took the true branch. An
	 * address keeps its (possibly negative) offset. */
	if (typ == CBits)
		x = ZX(x, (w ? 16 : 8) * T.wordsz);
#undef SX
#undef ZX
	*res = (Con){.type=typ, .sym=sym, .bits={.i=x}};
	return 0;
}

static void
foldflt(Con *res, int op, int w, Con *cl, Con *cr)
{
	float xs, ls, rs;
	double xd, ld, rd;

	if (cl->type != CBits || cr->type != CBits)
		err("invalid address operand for '%s'", optab[op].name);
	*res = (Con){.type = CBits};
	memset(&res->bits, 0, sizeof(res->bits));
	if (w)  {
		ld = cl->bits.d;
		rd = cr->bits.d;
		switch (op) {
		case Oadd: xd = ld + rd; break;
		case Osub: xd = ld - rd; break;
		case Oneg: xd = -ld; break;
		case Odiv: xd = ld / rd; break;
		case Omul: xd = ld * rd; break;
		case Oswtof: xd = (int32_t)cl->bits.i; break;
		case Ouwtof: xd = (uint32_t)cl->bits.i; break;
		case Osltof: xd = (int64_t)cl->bits.i; break;
		case Oultof: xd = (uint64_t)cl->bits.i; break;
		case Oexts: xd = cl->bits.s; break;
		case Ocast: xd = ld; break;
		default: die("unreachable");
		}
		res->bits.d = xd;
		res->flt = 2;
	} else {
		ls = cl->bits.s;
		rs = cr->bits.s;
		switch (op) {
		case Oadd: xs = ls + rs; break;
		case Osub: xs = ls - rs; break;
		case Oneg: xs = -ls; break;
		case Odiv: xs = ls / rs; break;
		case Omul: xs = ls * rs; break;
		case Oswtof: xs = (int32_t)cl->bits.i; break;
		case Ouwtof: xs = (uint32_t)cl->bits.i; break;
		case Osltof: xs = (int64_t)cl->bits.i; break;
		case Oultof: xs = (uint64_t)cl->bits.i; break;
		case Otruncd: xs = cl->bits.d; break;
		case Ocast: xs = ls; break;
		default: die("unreachable");
		}
		res->bits.s = xs;
		res->flt = 1;
	}
}

static Ref
opfold(int op, int cls, Con *cl, Con *cr, Fn *fn)
{
	Ref r;
	Con c;

	if (cls == Kw || cls == Kl) {
		if (foldint(&c, op, cls == Kl, cl, cr))
			return R;
	} else
		foldflt(&c, op, cls == Kd, cl, cr);
	if (!KWIDE(cls))
		c.bits.i &= 0xffffffff;
	r = newcon(&c, fn);
	assert(!(cls == Ks || cls == Kd) || c.flt);
	return r;
}

/* used by GVN */
Ref
foldref(Fn *fn, Ins *i)
{
	Ref rr;
	Con *cl, *cr;

	if (rtype(i->to) != RTmp)
		return R;
	if (optab[i->op].canfold) {
		if (rtype(i->arg[0]) != RCon)
			return R;
		cl = &fn->con[i->arg[0].val];
		rr = i->arg[1];
		if (req(rr, R))
		    rr = CON_Z;
		if (rtype(rr) != RCon)
			return R;
		cr = &fn->con[rr.val];

		return opfold(i->op, i->cls, cl, cr, fn);
	}
	return R;
}
