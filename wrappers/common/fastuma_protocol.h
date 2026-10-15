#ifndef FASTUMA_PROTOCOL_H
#define FASTUMA_PROTOCOL_H

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <strings.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/tcp.h>
#include <unistd.h>

#define FASTUMA_PORT 50423
#define FASTUMA_HOST "127.0.0.1"
#define FASTUMA_MAX_ATOMS 1000000u
#define FASTUMA_MAX_RESPONSE_BYTES (512ull * 1024ull * 1024ull)

#if defined(__GNUC__)
#define FASTUMA_UNUSED __attribute__((unused))
#else
#define FASTUMA_UNUSED
#endif

typedef struct {
    double energy;
    uint32_t grad_len;
    double *gradient;
    uint32_t hess_len;
    double *hessian_lower;
} FastUMAResult;

static void die(const char *msg) {
    fprintf(stderr, "UMA-CBridge error: %s\n", msg);
    exit(2);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "UMA-CBridge error: %s: %s\n", msg, strerror(errno));
    exit(2);
}

static FASTUMA_UNUSED char *strip_comment(char *s) {
    char *p = strchr(s, '#');
    if (p) *p = '\0';
    while (isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    return s;
}

static FASTUMA_UNUSED int atomic_number_from_symbol(const char *sym) {
    static const char *table[] = {
        "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
        "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
        "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
        "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
        "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
    };
    for (int i = 0; i < 118; i++) {
        if (strcasecmp(sym, table[i]) == 0) return i + 1;
    }
    return 0;
}

static int connect_fastuma(void) {
    const uint16_t endian = 1;
    if (*(const unsigned char *)&endian != 1) die("raw protocol requires a little-endian host");
    const char *host = getenv("FASTUMA_HOST");
    const char *port_env = getenv("FASTUMA_PORT");
    if (!host || !*host) host = FASTUMA_HOST;
    long port = FASTUMA_PORT;
    if (port_env && *port_env) {
        char *end = NULL;
        errno = 0;
        port = strtol(port_env, &end, 10);
        if (end == port_env || *end || errno || port < 1 || port > 65535)
            die("FASTUMA_PORT must be between 1 and 65535");
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) die_errno("socket failed");
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) die("bad UMA-CBridge host");
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) die_errno("connect failed");
    return fd;
}

static void writev_all(int fd, struct iovec *iov, int iovcnt) {
    while (iovcnt > 0) {
        ssize_t written = writev(fd, iov, iovcnt);
        if (written < 0) {
            if (errno == EINTR) continue;
            die_errno("writev failed");
        }
        if (written == 0) die("writev failed: connection closed");

        size_t sent = (size_t)written;
        while (iovcnt > 0 && sent >= iov[0].iov_len) {
            sent -= iov[0].iov_len;
            iov++;
            iovcnt--;
        }
        if (iovcnt > 0 && sent > 0) {
            iov[0].iov_base = (char *)iov[0].iov_base + sent;
            iov[0].iov_len -= sent;
        }
    }
}

static void read_all(int fd, void *buf, size_t n) {
    unsigned char *p = (unsigned char *)buf;
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            die_errno("recv failed");
        }
        if (r == 0) die("recv failed: connection closed");
        p += (size_t)r;
        n -= (size_t)r;
    }
}

static void send_fastuma_request(
    int fd,
    const uint16_t *atomic_numbers,
    const float *positions_angstrom,
    uint32_t natoms,
    int charge,
    int mult,
    const char *calc_type
) {
    char header[512];
    if (natoms == 0 || natoms > FASTUMA_MAX_ATOMS) die("bad atom count");
    int hlen = snprintf(
        header, sizeof(header),
        "{\"charge\":%d,\"multiplicity\":%d,\"calc_type\":\"%s\","
        "\"payload_format\":\"fastuma_raw_v1\"}",
        charge, mult, calc_type
    );
    if (hlen <= 0 || (size_t)hlen >= sizeof(header)) die("header too large");

    uint64_t header_len = (uint64_t)hlen;
    size_t payload_len_size = 4u + 14u * (size_t)natoms;
    uint64_t payload_len = (uint64_t)payload_len_size;

    struct iovec iov[6];
    iov[0].iov_base = &header_len;
    iov[0].iov_len = sizeof(header_len);
    iov[1].iov_base = header;
    iov[1].iov_len = (size_t)header_len;
    iov[2].iov_base = &payload_len;
    iov[2].iov_len = sizeof(payload_len);
    iov[3].iov_base = &natoms;
    iov[3].iov_len = sizeof(natoms);
    iov[4].iov_base = (void *)atomic_numbers;
    iov[4].iov_len = 2u * (size_t)natoms;
    iov[5].iov_base = (void *)positions_angstrom;
    iov[5].iov_len = 12u * (size_t)natoms;
    writev_all(fd, iov, 6);
}

static FastUMAResult recv_fastuma_response(int fd) {
    uint64_t msg_len = 0;
    read_all(fd, &msg_len, 8);
    if (msg_len == 0 || msg_len > FASTUMA_MAX_RESPONSE_BYTES || msg_len > (uint64_t)SIZE_MAX) {
        die("bad response length");
    }

    if (msg_len < 4) die("bad response");
    uint64_t remaining = msg_len;
    uint32_t status;
    read_all(fd, &status, 4);
    remaining -= 4u;
    if (status != 0) {
        uint32_t elen = 0;
        if (remaining < 4) die("short error response");
        read_all(fd, &elen, 4);
        remaining -= 4u;
        if ((uint64_t)elen > remaining || elen > (uint32_t)INT_MAX) die("bad error response length");
        char *errmsg = (char *)malloc((size_t)elen + 1u);
        if (!errmsg) die_errno("malloc error message failed");
        read_all(fd, errmsg, (size_t)elen);
        errmsg[elen] = '\0';
        fprintf(stderr, "Server error: %s\n", errmsg);
        free(errmsg);
        exit(3);
    }

    FastUMAResult res;
    memset(&res, 0, sizeof(res));
    if (remaining < 8u + 4u) die("short success response");
    read_all(fd, &res.energy, 8);
    remaining -= 8u;
    read_all(fd, &res.grad_len, 4);
    remaining -= 4u;
    size_t grad_bytes = (size_t)res.grad_len * sizeof(double);
    if ((uint64_t)grad_bytes > remaining || remaining - (uint64_t)grad_bytes < 4u) {
        die("short gradient response");
    }
    if (res.grad_len) {
        res.gradient = (double *)malloc(grad_bytes);
        if (!res.gradient) die_errno("malloc gradient failed");
        read_all(fd, res.gradient, grad_bytes);
        remaining -= (uint64_t)grad_bytes;
    }
    read_all(fd, &res.hess_len, 4);
    remaining -= 4u;
    size_t hess_bytes = (size_t)res.hess_len * sizeof(double);
    if ((uint64_t)hess_bytes > remaining) die("short hessian response");
    if (res.hess_len) {
        res.hessian_lower = (double *)malloc(hess_bytes);
        if (!res.hessian_lower) die_errno("malloc hessian failed");
        read_all(fd, res.hessian_lower, hess_bytes);
        remaining -= (uint64_t)hess_bytes;
    }
    if (remaining != 0) die("unexpected trailing response bytes");
    return res;
}

static void free_fastuma_result(FastUMAResult *res) {
    free(res->gradient);
    free(res->hessian_lower);
}

#endif
