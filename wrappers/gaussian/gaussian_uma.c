#include "../common/fastuma_protocol.h"

#define BOHR2ANGSTROM 0.529177249

static void write_gaussian_output(
    const char *path,
    uint32_t natoms,
    int calc_type_code,
    const FastUMAResult *res
) {
    FILE *f = fopen(path, "w");
    if (!f) die_errno("write Gaussian external output failed");

    fprintf(f, "%20.12e%20.12e%20.12e%20.12e\n", res->energy, 0.0, 0.0, 0.0);

    if (calc_type_code >= 1) {
        if (res->grad_len != 3u * natoms) die("gradient length mismatch");
        for (uint32_t i = 0; i < natoms; i++) {
            fprintf(
                f,
                "%20.12E%20.12E%20.12E\n",
                res->gradient[3 * i + 0],
                res->gradient[3 * i + 1],
                res->gradient[3 * i + 2]
            );
        }
    }

    if (calc_type_code == 2) {
        for (int i = 0; i < 2; i++) {
            fprintf(f, "%20.12E%20.12E%20.12E\n", 0.0, 0.0, 0.0);
        }
        for (uint32_t i = 0; i < 3u * natoms; i++) {
            fprintf(f, "%20.12E%20.12E%20.12E\n", 0.0, 0.0, 0.0);
        }

        uint32_t ncoords = 3u * natoms;
        uint64_t expected = (uint64_t)ncoords * (ncoords + 1u) / 2u;
        if (res->hess_len != expected) die("hessian lower-triangle length mismatch");
        for (uint32_t i = 0; i < res->hess_len; i += 3) {
            double a = res->hessian_lower[i];
            double b = (i + 1 < res->hess_len) ? res->hessian_lower[i + 1] : 0.0;
            double c = (i + 2 < res->hess_len) ? res->hessian_lower[i + 2] : 0.0;
            fprintf(f, "%20.12E%20.12E%20.12E\n", a, b, c);
        }
    }

    if (fclose(f) != 0) die_errno("close file failed");
}

int main(int argc, char **argv) {
    if (argc < 4) die("usage: gaussian_uma <ignored> GAUSSIAN_INPUT GAUSSIAN_OUTPUT");

    const char *input_path = argv[2];
    const char *output_path = argv[3];

    FILE *f = fopen(input_path, "r");
    if (!f) die_errno("open Gaussian external input failed");

    char line[4096];
    if (!fgets(line, sizeof(line), f)) die("empty Gaussian external input");
    int natoms = 0, calc_type_code = 0, charge = 0, mult = 1;
    if (sscanf(line, "%d %d %d %d", &natoms, &calc_type_code, &charge, &mult) != 4) die("bad Gaussian header");
    if (natoms <= 0 || (uint32_t)natoms > FASTUMA_MAX_ATOMS) die("bad atom count");
    if (calc_type_code < 0 || calc_type_code > 2) die("unsupported derivative order");

    uint16_t *atomic_numbers = (uint16_t *)calloc((size_t)natoms, sizeof(uint16_t));
    float *positions = (float *)calloc((size_t)natoms * 3u, sizeof(float));
    if (!atomic_numbers || !positions) die_errno("calloc Gaussian arrays failed");

    for (int i = 0; i < natoms; i++) {
        int atomic_number = 0;
        double x, y, coord_z;
        if (!fgets(line, sizeof(line), f)) die("short Gaussian external input");
        if (sscanf(line, "%d %lf %lf %lf", &atomic_number, &x, &y, &coord_z) != 4) die("bad Gaussian atom line");
        if (atomic_number <= 0 || atomic_number > 118) die("unsupported atomic number");
        atomic_numbers[i] = (uint16_t)atomic_number;
        positions[3 * i + 0] = (float)(x * BOHR2ANGSTROM);
        positions[3 * i + 1] = (float)(y * BOHR2ANGSTROM);
        positions[3 * i + 2] = (float)(coord_z * BOHR2ANGSTROM);
    }
    if (fclose(f) != 0) die_errno("close file failed");

    const char *calc_type = "energy";
    if (calc_type_code == 1) {
        calc_type = "force";
    } else if (calc_type_code == 2) {
        calc_type = "hessian";
    }

    int fd = connect_fastuma();
    send_fastuma_request(
        fd,
        atomic_numbers,
        positions,
        (uint32_t)natoms,
        charge,
        mult,
        calc_type
    );
    FastUMAResult res = recv_fastuma_response(fd);
    close(fd);

    write_gaussian_output(output_path, (uint32_t)natoms, calc_type_code, &res);
    printf("SCF Done:  E(UMA) =  %.15g     A.U.\n", res.energy);

    free(atomic_numbers);
    free(positions);
    free_fastuma_result(&res);
    return 0;
}
