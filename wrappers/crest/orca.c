#define _POSIX_C_SOURCE 200809L
#include "../common/fastuma_protocol.h"
#include <time.h>

#define ANGSTROM_TO_BOHR 1.8897259886

typedef struct {
    int charge;
    int multiplicity;
    uint32_t natoms;
    uint32_t capacity;
    uint16_t *atomic_numbers;
    float *positions_angstrom;
} Structure;

static char *lstrip(char *s) {
    while (isspace((unsigned char)*s)) {
        s++;
    }
    return s;
}

static void reserve_atoms(Structure *st, uint32_t needed) {
    if (needed <= st->capacity) {
        return;
    }
    uint32_t next = st->capacity ? st->capacity : 32u;
    while (next < needed) {
        if (next > FASTUMA_MAX_ATOMS / 2u) {
            next = FASTUMA_MAX_ATOMS;
            break;
        }
        next *= 2u;
    }
    uint16_t *new_numbers = (uint16_t *)realloc(st->atomic_numbers, (size_t)next * sizeof(uint16_t));
    if (!new_numbers) {
        die_errno("realloc atomic numbers failed");
    }
    st->atomic_numbers = new_numbers;

    float *new_positions = (float *)realloc(st->positions_angstrom, (size_t)next * 3u * sizeof(float));
    if (!new_positions) {
        die_errno("realloc positions failed");
    }
    st->positions_angstrom = new_positions;
    st->capacity = next;
}

static void append_atom(Structure *st, const char *sym, double x, double y, double zcoord) {
    if (st->natoms >= FASTUMA_MAX_ATOMS) {
        die("too many atoms");
    }
    uint16_t znum = atomic_number_from_symbol(sym);
    if (!znum) {
        die("unsupported element symbol");
    }
    reserve_atoms(st, st->natoms + 1u);
    st->atomic_numbers[st->natoms] = znum;
    st->positions_angstrom[3u * st->natoms + 0u] = (float)x;
    st->positions_angstrom[3u * st->natoms + 1u] = (float)y;
    st->positions_angstrom[3u * st->natoms + 2u] = (float)zcoord;
    st->natoms++;
}

static void copy_input_dir(const char *input_path, char *out, size_t out_size) {
    const char *slash = strrchr(input_path, '/');
    if (!slash) {
        snprintf(out, out_size, ".");
        return;
    }
    size_t n = (size_t)(slash - input_path);
    if (n == 0) {
        n = 1;
    }
    if (n >= out_size) {
        die("input directory path too long");
    }
    memcpy(out, input_path, n);
    out[n] = '\0';
}

static void resolve_relative_to_input(const char *input_path, const char *raw, char *out, size_t out_size) {
    if (raw[0] == '/') {
        int n = snprintf(out, out_size, "%s", raw);
        if (n < 0 || (size_t)n >= out_size) {
            die("path too long");
        }
        return;
    }
    char dir[4096];
    copy_input_dir(input_path, dir, sizeof(dir));
    int n = strcmp(dir, ".") == 0
        ? snprintf(out, out_size, "%s", raw)
        : snprintf(out, out_size, "%s/%s", dir, raw);
    if (n < 0 || (size_t)n >= out_size) {
        die("resolved path too long");
    }
}

static void read_xyz_file(const char *path, Structure *st) {
    FILE *f = fopen(path, "r");
    if (!f) {
        die_errno("open xyz failed");
    }
    char line[4096];
    if (!fgets(line, sizeof(line), f)) {
        die("empty xyz file");
    }
    long nat = strtol(line, NULL, 10);
    if (nat <= 0 || nat > (long)FASTUMA_MAX_ATOMS) {
        die("bad xyz atom count");
    }
    reserve_atoms(st, (uint32_t)nat);
    if (!fgets(line, sizeof(line), f)) {
        die("missing xyz comment line");
    }
    for (long i = 0; i < nat; i++) {
        if (!fgets(line, sizeof(line), f)) {
            die("short xyz file");
        }
        char sym[16];
        double x, y, zcoord;
        if (sscanf(line, "%15s %lf %lf %lf", sym, &x, &y, &zcoord) != 4) {
            die("bad xyz atom line");
        }
        append_atom(st, sym, x, y, zcoord);
    }
    if (fclose(f) != 0) {
        die_errno("close xyz failed");
    }
}

static int parse_xyzfile_line(char *line, const char *input_path, Structure *st) {
    char *p = lstrip(line);
    if (*p == '*') {
        p++;
        p = lstrip(p);
    }
    if (strncasecmp(p, "xyzfile", 7) != 0 || (p[7] && !isspace((unsigned char)p[7]))) {
        return 0;
    }
    p += 7;
    char *end = NULL;
    errno = 0;
    long charge = strtol(p, &end, 10);
    if (end == p || errno || charge < INT_MIN || charge > INT_MAX) {
        die("bad charge in *xyzfile line");
    }
    p = end;
    errno = 0;
    long mult = strtol(p, &end, 10);
    if (end == p || errno || mult < 1 || mult > INT_MAX) {
        die("bad multiplicity in *xyzfile line");
    }
    p = lstrip(end);
    if (!*p) {
        die("missing xyz filename in *xyzfile line");
    }
    char raw[4096];
    char quote = 0;
    if (*p == '"' || *p == '\'') {
        quote = *p++;
    }
    size_t i = 0;
    while (*p) {
        if ((quote && *p == quote) || (!quote && isspace((unsigned char)*p))) {
            break;
        }
        if (i + 1 >= sizeof(raw)) {
            die("xyz filename too long");
        }
        raw[i++] = *p++;
    }
    raw[i] = '\0';

    char resolved[4096];
    st->charge = (int)charge;
    st->multiplicity = (int)mult;
    resolve_relative_to_input(input_path, raw, resolved, sizeof(resolved));
    read_xyz_file(resolved, st);
    return 1;
}

static int parse_xyz_header(char *line, Structure *st) {
    char *p = lstrip(line);
    if (*p == '*') {
        p++;
        p = lstrip(p);
    }
    if (strncasecmp(p, "xyz", 3) != 0 || (p[3] && !isspace((unsigned char)p[3]))) {
        return 0;
    }
    p += 3;
    char *end = NULL;
    errno = 0;
    long charge = strtol(p, &end, 10);
    if (end == p || errno || charge < INT_MIN || charge > INT_MAX) {
        die("bad charge in *xyz line");
    }
    p = end;
    errno = 0;
    long mult = strtol(p, &end, 10);
    if (end == p || errno || mult < 1 || mult > INT_MAX) {
        die("bad multiplicity in *xyz line");
    }
    st->charge = (int)charge;
    st->multiplicity = (int)mult;
    return 1;
}

static Structure read_orca_coordinates(const char *input_path) {
    FILE *f = fopen(input_path, "r");
    if (!f) {
        die_errno("open ORCA input failed");
    }
    Structure st;
    memset(&st, 0, sizeof(st));
    st.multiplicity = 1;

    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *clean = strip_comment(line);
        if (!*clean) {
            continue;
        }
        if (*clean != '*') {
            continue;
        }
        if (parse_xyzfile_line(clean, input_path, &st)) {
            if (fclose(f) != 0) die_errno("close file failed");
            return st;
        }
        if (parse_xyz_header(clean, &st)) {
            while (fgets(line, sizeof(line), f)) {
                clean = strip_comment(line);
                if (!*clean) {
                    continue;
                }
                if (*clean == '*') {
                    if (st.natoms == 0) {
                        die("empty *xyz block");
                    }
                    if (fclose(f) != 0) die_errno("close file failed");
                    return st;
                }
                char sym[16];
                double x, y, zcoord;
                if (sscanf(clean, "%15s %lf %lf %lf", sym, &x, &y, &zcoord) != 4) {
                    die("bad inline xyz atom line");
                }
                append_atom(&st, sym, x, y, zcoord);
            }
            die("unterminated *xyz block");
        }
    }
    if (fclose(f) != 0) {
        die_errno("close ORCA input failed");
    }
    die("ORCA input does not contain *xyz or *xyzfile coordinates");
    return st;
}

static void engrad_path_from_input(const char *input_path, char *out, size_t out_size) {
    int n = snprintf(out, out_size, "%s", input_path);
    if (n < 0 || (size_t)n >= out_size) {
        die("engrad path too long");
    }
    char *slash = strrchr(out, '/');
    char *base = slash ? slash + 1 : out;
    char *dot = strrchr(base, '.');
    if (dot) {
        *dot = '\0';
    }
    if (strlen(out) + strlen(".engrad") >= out_size) {
        die("engrad path too long");
    }
    strcat(out, ".engrad");
}

static void log_call(const char *input_path, const char *engrad_path, const char *event, const char *extra) {
    const char *log_path = getenv("CREST_ORCA_LOG");
    if (!log_path || !*log_path) {
        return;
    }
    FILE *f = fopen(log_path, "a");
    if (!f) {
        return;
    }
    fprintf(f, "%ld\t%s\tpid=%ld\tcwd=", (long)time(NULL), event, (long)getpid());
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd))) {
        fprintf(f, "%s", cwd);
    }
    fprintf(f, "\tinput=%s\toutput=%s", input_path, engrad_path);
    if (extra && *extra) {
        fprintf(f, "\t%s", extra);
    }
    fprintf(f, "\n");
    if (fclose(f) != 0) die_errno("close file failed");
}

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && *value && strcmp(value, "0") != 0;
}

static FastUMAResult call_fastuma(const Structure *st) {
    int fd = connect_fastuma();
    send_fastuma_request(fd, st->atomic_numbers, st->positions_angstrom,
                         st->natoms, st->charge, st->multiplicity, "force");
    FastUMAResult result = recv_fastuma_response(fd);
    close(fd);
    return result;
}

static void write_engrad(const char *path, const Structure *st, const FastUMAResult *res) {
    if (res->grad_len != 3u * st->natoms) {
        die("gradient length mismatch");
    }
    char tmp[4096];
    int n = snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", path, (long)getpid());
    if (n < 0 || (size_t)n >= sizeof(tmp)) {
        die("temporary engrad path too long");
    }
    FILE *f = fopen(tmp, "w");
    if (!f) {
        die_errno("write temporary engrad failed");
    }
    fprintf(f, "#\n# Number of atoms\n#\n");
    fprintf(f, "%12u\n", st->natoms);
    fprintf(f, "#\n# The current total energy in Eh\n#\n");
    fprintf(f, "%22.12f\n", res->energy);
    fprintf(f, "#\n# The current gradient in Eh/bohr\n#\n");
    for (uint32_t i = 0; i < res->grad_len; i++) {
        fprintf(f, "%22.12f\n", res->gradient[i]);
    }
    fprintf(f, "#\n# The atomic numbers and current coordinates in Bohr\n#\n");
    for (uint32_t i = 0; i < st->natoms; i++) {
        const float *pos = &st->positions_angstrom[3u * i];
        fprintf(
            f,
            "%6u%18.10f%18.10f%18.10f\n",
            st->atomic_numbers[i],
            (double)pos[0] * ANGSTROM_TO_BOHR,
            (double)pos[1] * ANGSTROM_TO_BOHR,
            (double)pos[2] * ANGSTROM_TO_BOHR
        );
    }
    if (env_enabled("CREST_ORCA_FSYNC")) {
        if (fflush(f) != 0 || fsync(fileno(f)) != 0) die_errno("sync engrad failed");
    }
    if (fclose(f) != 0) die_errno("close file failed");
    if (rename(tmp, path) != 0) {
        die_errno("replace engrad failed");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: orca ORCA_INPUT_FILE\n");
        return 2;
    }
    const char *input_path = argv[1];
    char engrad_path[4096];
    engrad_path_from_input(input_path, engrad_path, sizeof(engrad_path));
    log_call(input_path, engrad_path, "start", "");

    Structure st = read_orca_coordinates(input_path);
    char parsed[128];
    snprintf(parsed, sizeof(parsed), "charge=%d\tmult=%d\tnatoms=%u", st.charge, st.multiplicity, st.natoms);
    log_call(input_path, engrad_path, "parsed", parsed);

    FastUMAResult res = call_fastuma(&st);
    write_engrad(engrad_path, &st, &res);
    log_call(input_path, engrad_path, "wrote_engrad", "");

    printf("UMA-CBridge CREST adapter\n");
    printf("Input file: %s\n", input_path);
    printf("Number of atoms: %u\n", st.natoms);
    printf("Gradient file: %s\n", engrad_path);
    printf("FINAL SINGLE POINT ENERGY     %.15f\n", res.energy);
    printf("****ORCA TERMINATED NORMALLY****\n");

    free(st.atomic_numbers);
    free(st.positions_angstrom);
    free_fastuma_result(&res);
    return 0;
}
