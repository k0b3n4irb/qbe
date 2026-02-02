#include "all.h"

enum {
	SecText,
	SecData,
	SecBss,
};

static int datasec_counter = 0;

/*
 * Initialized data handling for SNES (w65816):
 *
 * Problem: Initialized static variables like "static u8 x = 5;" need to
 * live in RAM (writable), but their initial values must come from ROM.
 *
 * Solution:
 * 1. Variable storage goes in RAMSECTION (RAM)
 * 2. Initial values go in a ROM section with format:
 *    [ram_addr:2][size:2][data:N]
 * 3. crt0 copies init data from ROM to RAM at startup
 *
 * We buffer data items and emit both sections at DEnd when we know the size.
 */

/* Buffer for accumulating initialized data */
#define MAX_INIT_ITEMS 1024
static struct {
	int type;       /* DB, DH, DW, DL, or -1 for string, -2 for ref, -3 for zero-fill */
	int64_t num;    /* numeric value or size for zero-fill */
	char *str;      /* string value (if type == -1) */
	struct {        /* reference (if type == -2) */
		char *name;
		int64_t off;
	} ref;
	int reftype;    /* original type for references (DB, DH, etc.) */
} init_items[MAX_INIT_ITEMS];
static int init_count;
static int64_t init_total_size;

/* Current data definition info */
static char *cur_data_name;
static Lnk *cur_data_lnk;
static int has_nonzero_data;

/* Size of each data type in bytes */
static int dtype_size[] = {
	[DB] = 1,
	[DH] = 2,
	[DW] = 4,
	[DL] = 8
};

/* Emit a string with proper handling of null terminators for WLA-DX.
 * WLA-DX doesn't support \000 escape sequences, so we convert them to:
 * "Hello\000World" -> "Hello", 0, "World"
 */
static void
emit_wladx_string(char *str, FILE *f)
{
	char *p = str;
	int in_string = 0;
	int need_comma = 0;

	/* Skip opening quote if present */
	if (*p == '"')
		p++;

	fputs("\t.ASC ", f);

	while (*p) {
		/* Check for \000 escape sequence (4 chars) */
		if (p[0] == '\\' && p[1] == '0' && p[2] == '0' && p[3] == '0') {
			if (in_string) {
				fputc('"', f);
				in_string = 0;
			}
			if (need_comma)
				fputs(", ", f);
			fputs("0", f);
			need_comma = 1;
			p += 4;
			continue;
		}
		/* Check for closing quote */
		if (*p == '"' && p[1] == '\0') {
			if (in_string)
				fputc('"', f);
			break;
		}
		/* Regular character */
		if (!in_string) {
			if (need_comma)
				fputs(", ", f);
			fputc('"', f);
			in_string = 1;
			need_comma = 1;
		}
		fputc(*p, f);
		p++;
	}

	/* Close string if still open and not ended by closing quote */
	if (in_string && *p != '"')
		fputc('"', f);

	fputc('\n', f);
}

/* Calculate string length including null terminators */
static int64_t
string_length(char *str)
{
	char *p = str;
	int64_t len = 0;

	if (*p == '"')
		p++;

	while (*p) {
		if (p[0] == '\\' && p[1] == '0' && p[2] == '0' && p[3] == '0') {
			len++;  /* null byte */
			p += 4;
		} else if (*p == '"' && p[1] == '\0') {
			break;
		} else {
			len++;
			p++;
		}
	}
	return len;
}

void
emitlnk(char *n, Lnk *l, int s, FILE *f)
{
	char *pfx;
	char *name;
	(void)l;  /* WLA-DX doesn't use linkage flags */

	pfx = n[0] == '"' ? "" : T.assym;

	/* WLA-DX: strip .L prefix from local labels (e.g., .Lstring.1 -> string.1)
	 * WLA-DX doesn't support labels starting with '.' */
	name = n;
	if (n[0] == '.' && n[1] == 'L')
		name = n + 2;

	/* WLA-DX compatible section syntax for w65816 */
	switch (s) {
	case SecText:
		fprintf(f, ".SECTION \".text.%s\" SUPERFREE\n", name);
		break;
	case SecData:
		fprintf(f, ".SECTION \".rodata.%d\" SUPERFREE\n", ++datasec_counter);
		break;
	case SecBss:
		/* BSS goes to RAM (SLOT 1) not ROM */
		fprintf(f, ".RAMSECTION \".bss.%d\" BANK 0 SLOT 1\n", ++datasec_counter);
		break;
	}
	/* WLA-DX: skip .globl since we compile as single file.
	 * Symbols are visible within the same compilation unit. */
	fprintf(f, "%s%s:\n", pfx, name);
}

void
emitfnlnk(char *n, Lnk *l, FILE *f)
{
	emitlnk(n, l, SecText, f);
}

/* Emit buffered init data items to file */
static void
emit_init_data(FILE *f)
{
	static char *dtoa[] = {
		[DB] = "\t.db",
		[DH] = "\t.dw",
		[DW] = "\t.dl",
		[DL] = "\t.dl"
	};
	char *p;
	int i;

	for (i = 0; i < init_count; i++) {
		if (init_items[i].type == -1) {
			/* String */
			emit_wladx_string(init_items[i].str, f);
		} else if (init_items[i].type == -2) {
			/* Reference */
			char *refname = init_items[i].ref.name;
			if (refname[0] == '.' && refname[1] == 'L')
				refname = refname + 2;
			p = refname[0] == '"' ? "" : T.assym;
			fprintf(f, "%s %s%s%+"PRId64"\n",
				dtoa[init_items[i].reftype], p, refname,
				init_items[i].ref.off);
		} else if (init_items[i].type == -3) {
			/* Zero-fill */
			fprintf(f, "\t.dsb %"PRId64", 0\n", init_items[i].num);
		} else {
			/* Numeric value */
			fprintf(f, "%s %"PRId64"\n",
				dtoa[init_items[i].type], init_items[i].num);
		}
	}
}

void
emitdat(Dat *d, FILE *f)
{
	char *p;
	char *name;
	int sec_id;

	switch (d->type) {
	case DStart:
		/* Reset state for new data definition */
		init_count = 0;
		init_total_size = 0;
		cur_data_name = d->name;
		cur_data_lnk = d->lnk;
		has_nonzero_data = 0;
		break;

	case DEnd:
		if (d->lnk->common) {
			/* Common data - not supported for now */
			if (has_nonzero_data)
				die("initialized common data not supported");
			p = d->name[0] == '"' ? "" : T.assym;
			fprintf(f, ".comm %s%s,%"PRId64,
				p, d->name, init_total_size);
			if (d->lnk->align)
				fprintf(f, ",%d", d->lnk->align);
			fputc('\n', f);
		}
		else if (!has_nonzero_data) {
			/* Pure BSS - only zeros, emit RAMSECTION */
			sec_id = ++datasec_counter;
			fprintf(f, ".RAMSECTION \".bss.%d\" BANK 0 SLOT 1\n", sec_id);
			p = cur_data_name[0] == '"' ? "" : T.assym;
			name = cur_data_name;
			if (name[0] == '.' && name[1] == 'L')
				name = name + 2;
			fprintf(f, "%s%s:\n", p, name);
			fprintf(f, "\tdsb %"PRId64"\n", init_total_size);
			fputs(".ENDS\n", f);
		}
		else {
			/* Initialized data - emit RAM section + ROM init record */
			sec_id = ++datasec_counter;
			p = cur_data_name[0] == '"' ? "" : T.assym;
			name = cur_data_name;
			if (name[0] == '.' && name[1] == 'L')
				name = name + 2;

			/* 1. Emit RAMSECTION for the variable */
			fprintf(f, ".RAMSECTION \".data.%d\" BANK 0 SLOT 1\n", sec_id);
			fprintf(f, "%s%s:\n", p, name);
			fprintf(f, "\tdsb %"PRId64"\n", init_total_size);
			fputs(".ENDS\n\n", f);

			/* 2. Emit ROM section with init data */
			/* Format: [target_addr:2][size:2][data:N] */
			/* Use APPENDTO to add to the existing .data_init section */
			fprintf(f, ".SECTION \".data_init.%d\" SEMIFREE APPENDTO \".data_init\"\n", sec_id);
			fprintf(f, "\t.dw %s%s\n", p, name);  /* RAM target address */
			fprintf(f, "\t.dw %"PRId64"\n", init_total_size);  /* Size */
			emit_init_data(f);  /* Actual data bytes */
			fputs(".ENDS\n", f);
		}
		break;

	case DZ:
		/* Zero-fill - accumulate size */
		if (has_nonzero_data) {
			/* Already have non-zero data, buffer this zero-fill */
			if (init_count >= MAX_INIT_ITEMS)
				die("too many data items");
			init_items[init_count].type = -3;
			init_items[init_count].num = d->u.num;
			init_count++;
		}
		init_total_size += d->u.num;
		break;

	default:
		/* DB, DH, DW, DL - actual data */
		has_nonzero_data = 1;

		if (init_count >= MAX_INIT_ITEMS)
			die("too many data items");

		if (d->isstr) {
			if (d->type != DB)
				err("strings only supported for 'b' currently");
			init_items[init_count].type = -1;
			init_items[init_count].str = d->u.str;
			init_total_size += string_length(d->u.str);
		}
		else if (d->isref) {
			init_items[init_count].type = -2;
			init_items[init_count].ref.name = d->u.ref.name;
			init_items[init_count].ref.off = d->u.ref.off;
			init_items[init_count].reftype = d->type;
			init_total_size += dtype_size[d->type];
		}
		else {
			init_items[init_count].type = d->type;
			init_items[init_count].num = d->u.num;
			init_total_size += dtype_size[d->type];
		}
		init_count++;
		break;
	}
}

typedef struct Asmbits Asmbits;

struct Asmbits {
	bits n;
	int size;
	Asmbits *link;
};

static Asmbits *stash;

int
stashbits(bits n, int size)
{
	Asmbits **pb, *b;
	int i;

	assert(size == 4 || size == 8 || size == 16);
	for (pb=&stash, i=0; (b=*pb); pb=&b->link, i++)
		if (size <= b->size && b->n == n)
			return i;
	b = emalloc(sizeof *b);
	b->n = n;
	b->size = size;
	b->link = 0;
	*pb = b;
	return i;
}

static void
emitfin(FILE *f, char *sec[3])
{
	Asmbits *b;
	int lg, i;
	union { int32_t i; float f; } u;

	if (!stash)
		return;
	fprintf(f, "/* floating point constants */\n");
	for (lg=4; lg>=2; lg--)
		for (b=stash, i=0; b; b=b->link, i++) {
			if (b->size == (1<<lg)) {
				fprintf(f,
					".section %s\n"
					".p2align %d\n"
					"%sfp%d:",
					sec[lg-2], lg, T.asloc, i
				);
				if (lg == 4)
					fprintf(f,
						"\n\t.quad %"PRId64
						"\n\t.quad 0\n\n",
						(int64_t)b->n);
				else if (lg == 3)
					fprintf(f,
						"\n\t.quad %"PRId64
						" /* %f */\n\n",
						(int64_t)b->n,
						*(double *)&b->n);
				else if (lg == 2) {
					u.i = b->n;
					fprintf(f,
						"\n\t.int %"PRId32
						" /* %f */\n\n",
						u.i, (double)u.f);
				}
			}
		}
	while ((b=stash)) {
		stash = b->link;
		free(b);
	}
}

void
elf_emitfin(FILE *f)
{
	static char *sec[3] = { ".rodata", ".rodata", ".rodata" };

	emitfin(f ,sec);
	fprintf(f, ".section .note.GNU-stack,\"\",@progbits\n");
}

void
elf_emitfnfin(char *fn, FILE *f)
{
	fprintf(f, ".type %s, @function\n", fn);
	fprintf(f, ".size %s, .-%s\n", fn, fn);
}

void
macho_emitfin(FILE *f)
{
	static char *sec[3] = {
		"__TEXT,__literal4,4byte_literals",
		"__TEXT,__literal8,8byte_literals",
		".abort \"unreachable\"",
	};

	emitfin(f, sec);
}

static uint32_t *file;
static uint nfile;
static uint curfile;

void
emitdbgfile(char *fn, FILE *f)
{
	uint32_t id;
	uint n;

	id = intern(fn);
	for (n=0; n<nfile; n++)
		if (file[n] == id) {
			/* gas requires positive
			 * file numbers */
			curfile = n + 1;
			return;
		}
	if (!file)
		file = vnew(0, sizeof *file, PHeap);
	vgrow(&file, ++nfile);
	file[nfile-1] = id;
	curfile = nfile;
	fprintf(f, ".file %u %s\n", curfile, fn);
}

void
emitdbgloc(uint line, uint col, FILE *f)
{
	if (col != 0)
		fprintf(f, "\t.loc %u %u %u\n", curfile, line, col);
	else
		fprintf(f, "\t.loc %u %u\n", curfile, line);
}
