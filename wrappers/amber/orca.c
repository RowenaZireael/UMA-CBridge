#include "../common/fastuma_protocol.h"

#define ANGSTROM_TO_BOHR 1.8897259886

typedef struct {
    uint32_t natoms;
    uint16_t *atomic_numbers;
    float *positions_angstrom;
} XyzData;

static void rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) {
        s[--n] = '\0';
    }
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

static void resolve_relative_to_input(
    const char *input_path,
    const char *maybe_relative,
    char *out,
    size_t out_size
) {
    if (maybe_relative[0] == '/') {
        snprintf(out, out_size, "%s", maybe_relative);
        return;
    }

    char dir[4096];
    copy_input_dir(input_path, dir, sizeof(dir));
    if (strcmp(dir, ".") == 0) {
        snprintf(out, out_size, "%s", maybe_relative);
    } else {
        int n = snprintf(out, out_size, "%s/%s", dir, maybe_relative);
        if (n < 0 || (size_t)n >= out_size) {
            die("resolved xyz path too long");
        }
    }
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

static int parse_xyzfile_line(const char *line, char *xyz_path, size_t xyz_path_size, int *charge, int *mult) {
    const char *p = line;
    while (isspace((unsigned char)*p)) {
        p++;
    }
    if (*p != '*') {
        return 0;
    }
    p++;
    while (isspace((unsigned char)*p)) {
        p++;
    }
    if (strncasecmp(p, "xyzfile", 7) != 0) {
        return 0;
    }
    p += 7;
    if (*p && !isspace((unsigned char)*p)) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    long q = strtol(p, &end, 10);
    if (end == p || errno != 0 || q < INT_MIN || q > INT_MAX) {
        die("bad charge in *xyzfile line");
    }
    p = end;
    errno = 0;
    long m = strtol(p, &end, 10);
    if (end == p || errno != 0 || m < 1 || m > INT_MAX) {
        die("bad multiplicity in *xyzfile line");
    }
    p = end;
    while (isspace((unsigned char)*p)) {
        p++;
    }
    if (*p == '\0') {
        die("missing xyz filename in *xyzfile line");
    }

    char quote = 0;
    if (*p == '"' || *p == '\'') {
        quote = *p++;
    }
    size_t i = 0;
    while (*p) {
        if ((quote && *p == quote) || (!quote && isspace((unsigned char)*p))) {
            break;
        }
        if (*p == '#') {
            break;
        }
        if (i + 1 >= xyz_path_size) {
            die("xyz filename too long");
        }
        xyz_path[i++] = *p++;
    }
    xyz_path[i] = '\0';
    if (i == 0) {
        die("empty xyz filename in *xyzfile line");
    }

    *charge = (int)q;
    *mult = (int)m;
    return 1;
}

static void read_orca_input(const char *input_path, char *xyz_path, size_t xyz_path_size, int *charge, int *mult) {
    FILE *f = fopen(input_path, "r");
    if (!f) {
        die_errno("open ORCA input failed");
    }

    char line[4096];
    char raw_xyz[4096] = "";
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        rstrip(line);
        if (parse_xyzfile_line(line, raw_xyz, sizeof(raw_xyz), charge, mult)) {
            found = 1;
            break;
        }
    }
    if (fclose(f) != 0) die_errno("close file failed");
    if (!found) {
        die("ORCA input does not contain a *xyzfile charge multiplicity filename line");
    }
    resolve_relative_to_input(input_path, raw_xyz, xyz_path, xyz_path_size);
}

static void read_xyz(const char *path, XyzData *xyz) {
    FILE *f = fopen(path, "r");
    if (!f) {
        die_errno("open xyz failed");
    }

    char line[4096];
    if (!fgets(line, sizeof(line), f)) {
        die("empty xyz");
    }
    int n = atoi(line);
    if (n <= 0 || (uint32_t)n > FASTUMA_MAX_ATOMS) {
        die("bad atom count in xyz");
    }
    if (!fgets(line, sizeof(line), f)) {
        die("missing xyz comment line");
    }

    uint16_t *atomic_numbers = (uint16_t *)calloc((size_t)n, sizeof(uint16_t));
    float *positions = (float *)calloc((size_t)n * 3u, sizeof(float));
    if (!atomic_numbers || !positions) {
        die_errno("calloc xyz arrays failed");
    }

    for (int i = 0; i < n; i++) {
        char sym[16];
        double x, y, z;
        if (!fgets(line, sizeof(line), f)) {
            die("short xyz file");
        }
        if (sscanf(line, "%15s %lf %lf %lf", sym, &x, &y, &z) != 4) {
            die("bad xyz atom line");
        }
        int atomic_number = atomic_number_from_symbol(sym);
        if (!atomic_number) {
            die("unsupported element symbol in xyz");
        }
        atomic_numbers[i] = (uint16_t)atomic_number;
        positions[3 * i + 0] = (float)x;
        positions[3 * i + 1] = (float)y;
        positions[3 * i + 2] = (float)z;
    }
    if (fclose(f) != 0) die_errno("close file failed");

    xyz->natoms = (uint32_t)n;
    xyz->atomic_numbers = atomic_numbers;
    xyz->positions_angstrom = positions;
}

static void write_engrad(const char *path, const XyzData *xyz, double energy, const double *gradient, uint32_t grad_len) {
    if (grad_len != 3u * xyz->natoms) {
        die("gradient length mismatch");
    }

    FILE *f = fopen(path, "w");
    if (!f) {
        die_errno("write engrad failed");
    }

    fprintf(f, "#\n# Number of atoms\n#\n");
    fprintf(f, " %u\n", xyz->natoms);
    fprintf(f, "#\n# The current total energy in Eh\n#\n");
    fprintf(f, "%20.12f\n", energy);
    fprintf(f, "#\n# The current gradient in Eh/bohr\n#\n");
    for (uint32_t i = 0; i < grad_len; i++) {
        fprintf(f, "%20.12f\n", gradient[i]);
    }
    fprintf(f, "#\n# The atomic numbers and current coordinates in Bohr\n#\n");
    for (uint32_t i = 0; i < xyz->natoms; i++) {
        double x = (double)xyz->positions_angstrom[3 * i + 0] * ANGSTROM_TO_BOHR;
        double y = (double)xyz->positions_angstrom[3 * i + 1] * ANGSTROM_TO_BOHR;
        double z = (double)xyz->positions_angstrom[3 * i + 2] * ANGSTROM_TO_BOHR;
        fprintf(f, "%4u %13.7f %13.7f %13.7f\n", xyz->atomic_numbers[i], x, y, z);
    }
    if (fclose(f) != 0) die_errno("close file failed");
}

static void free_xyz(XyzData *xyz) {
    free(xyz->atomic_numbers);
    free(xyz->positions_angstrom);
    memset(xyz, 0, sizeof(*xyz));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: orca ORCA_INPUT_FILE\n");
        return 2;
    }

    const char *input_path = argv[1];
    char xyz_path[4096];
    char engrad_path[4096];
    int charge = 0;
    int mult = 1;

    read_orca_input(input_path, xyz_path, sizeof(xyz_path), &charge, &mult);
    engrad_path_from_input(input_path, engrad_path, sizeof(engrad_path));

    XyzData xyz;
    memset(&xyz, 0, sizeof(xyz));
    read_xyz(xyz_path, &xyz);

    int fd = connect_fastuma();
    send_fastuma_request(fd, xyz.atomic_numbers, xyz.positions_angstrom, xyz.natoms, charge, mult, "force");
    FastUMAResult res = recv_fastuma_response(fd);
    close(fd);

    write_engrad(engrad_path, &xyz, res.energy, res.gradient, res.grad_len);
    printf("UMA-CBridge AMBER adapter\n");
    printf("Input coordinates: %s\n", xyz_path);
    printf("Gradient file: %s\n", engrad_path);
    printf("FINAL SINGLE POINT ENERGY     %.15g\n", res.energy);
    printf("****ORCA TERMINATED NORMALLY****\n");

    free_fastuma_result(&res);
    free_xyz(&xyz);
    return 0;
}
