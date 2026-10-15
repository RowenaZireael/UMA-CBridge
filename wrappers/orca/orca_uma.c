#include "../common/fastuma_protocol.h"

static void read_orca_input(const char *path, char *xyz_path, size_t xyz_path_size, int *charge, int *mult, int *want_gradient) {
    FILE *f = fopen(path, "r");
    if (!f) die_errno("open ORCA wrapper input failed");
    char line[4096];

    if (!fgets(line, sizeof(line), f)) die("missing xyz filename");
    snprintf(xyz_path, xyz_path_size, "%s", strip_comment(line));
    if (!fgets(line, sizeof(line), f)) die("missing charge");
    *charge = atoi(strip_comment(line));
    if (!fgets(line, sizeof(line), f)) die("missing multiplicity");
    *mult = atoi(strip_comment(line));
    if (!fgets(line, sizeof(line), f)) die("missing ncores");
    if (!fgets(line, sizeof(line), f)) die("missing gradient flag");
    *want_gradient = atoi(strip_comment(line));
    if (fclose(f) != 0) die_errno("close file failed");
}

static void read_xyz(const char *path, uint32_t *natoms, uint16_t **atomic_numbers, float **positions) {
    FILE *f = fopen(path, "r");
    if (!f) die_errno("open xyz failed");
    char line[4096];
    if (!fgets(line, sizeof(line), f)) die("empty xyz");
    int n = atoi(line);
    if (n <= 0 || (uint32_t)n > FASTUMA_MAX_ATOMS) die("bad atom count in xyz");
    if (!fgets(line, sizeof(line), f)) die("missing xyz comment");

    uint16_t *atomic_nums = (uint16_t *)calloc((size_t)n, sizeof(uint16_t));
    float *pos = (float *)calloc((size_t)n * 3u, sizeof(float));
    if (!atomic_nums || !pos) die_errno("calloc xyz arrays failed");

    for (int i = 0; i < n; i++) {
        char sym[16];
        double x, y, coord_z;
        if (!fgets(line, sizeof(line), f)) die("short xyz");
        if (sscanf(line, "%15s %lf %lf %lf", sym, &x, &y, &coord_z) != 4) die("bad xyz atom line");
        int atomic_number = atomic_number_from_symbol(sym);
        if (!atomic_number) die("unsupported element symbol in xyz");
        atomic_nums[i] = (uint16_t)atomic_number;
        pos[3 * i + 0] = (float)x;
        pos[3 * i + 1] = (float)y;
        pos[3 * i + 2] = (float)coord_z;
    }
    if (fclose(f) != 0) die_errno("close file failed");
    *natoms = (uint32_t)n;
    *atomic_numbers = atomic_nums;
    *positions = pos;
}

static void engrad_name(const char *xyz_path, char *out, size_t out_size) {
    const char *base = strrchr(xyz_path, '/');
    base = base ? base + 1 : xyz_path;
    int copied = snprintf(out, out_size, "%s", base);
    if (copied < 0 || (size_t)copied >= out_size) die("engrad filename too long");
    char *dot = strrchr(out, '.');
    if (dot) *dot = '\0';
    size_t n = strlen(out);
    if (n + 8 >= out_size) die("engrad filename too long");
    strcat(out, ".engrad");
}

static void write_engrad(const char *path, uint32_t natoms, double energy, int want_gradient, const double *gradient, uint32_t grad_len) {
    FILE *f = fopen(path, "w");
    if (!f) die_errno("write engrad failed");
    fprintf(f, "#\n# Number of atoms\n#\n%u\n", natoms);
    fprintf(f, "#\n# Total energy [Eh]\n#\n%.12e\n", energy);
    if (want_gradient) {
        if (grad_len != 3u * natoms) die("gradient length mismatch");
        fprintf(f, "#\n# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...\n#\n");
        for (uint32_t i = 0; i < grad_len; i++) {
            fprintf(f, " %.12e\n", gradient[i]);
        }
    }
    if (fclose(f) != 0) die_errno("close file failed");
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: orca_uma ORCA_INPUT_FILE");

    char xyz_path[4096];
    int charge = 0, mult = 1, want_gradient = 0;
    read_orca_input(argv[1], xyz_path, sizeof(xyz_path), &charge, &mult, &want_gradient);

    uint32_t natoms = 0;
    uint16_t *atomic_numbers = NULL;
    float *positions = NULL;
    read_xyz(xyz_path, &natoms, &atomic_numbers, &positions);

    int fd = connect_fastuma();
    send_fastuma_request(
        fd, atomic_numbers, positions, natoms, charge, mult,
        want_gradient ? "force" : "energy"
    );
    FastUMAResult res = recv_fastuma_response(fd);
    close(fd);

    char out_path[4096];
    engrad_name(xyz_path, out_path, sizeof(out_path));
    write_engrad(out_path, natoms, res.energy, want_gradient, res.gradient, res.grad_len);
    printf("Wrote %s\n", out_path);
    printf("SCF Done:  E(UMA) =  %.15g     A.U.\n", res.energy);

    free(atomic_numbers);
    free(positions);
    free_fastuma_result(&res);
    return 0;
}
