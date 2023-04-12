/*
 * mtxreusedist
 *
 * Copyright (C) 2023 James D. Trotter
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see
 * <https://www.gnu.org/licenses/>.
 *
 * Compute reference distance for sparse matrix-vector multiply.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *
 * History:
 *
 *  1.0 — 2023-04-12:
 *
 *   - initial version
 */

#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_LIBZ
#include <zlib.h>
#endif

#include <unistd.h>

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef IDXTYPEWIDTH
typedef int idx_t;
#define PRIdx "d"
#define IDX_T_MIN INT_MIN
#define IDX_T_MAX INT_MAX
#define parse_idx_t parse_int
#elif IDXTYPEWIDTH == 32
typedef int32_t idx_t;
#define PRIdx PRId32
#define IDX_T_MIN INT32_MIN
#define IDX_T_MAX INT32_MAX
#define parse_idx_t parse_int32_t
#elif IDXTYPEWIDTH == 64
typedef int64_t idx_t;
#define PRIdx PRId64
#define IDX_T_MIN INT64_MIN
#define IDX_T_MAX INT64_MAX
#define parse_idx_t parse_int64_t
#endif

const char * program_name = "mtxreusedist";
const char * program_version = "1.0";
const char * program_copyright =
    "Copyright (C) 2023 James D. Trotter";
const char * program_license =
    "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>\n"
    "This is free software: you are free to change and redistribute it.\n"
    "There is NO WARRANTY, to the extent permitted by law.";
const char * program_invocation_name;
const char * program_invocation_short_name;

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    char * Apath;
#ifdef HAVE_LIBZ
    int gzip;
#endif
    bool separate_diagonal;
    bool bandwidth;
    bool profile;
    int nblockssize;
    int * nblocks;
    int verbose;
    int quiet;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->Apath = NULL;
#ifdef HAVE_LIBZ
    args->gzip = 0;
#endif
    args->separate_diagonal = false;
    args->bandwidth = false;
    args->profile = false;
    args->nblockssize = 0;
    args->nblocks = NULL;
    args->quiet = 0;
    args->verbose = 0;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->Apath) free(args->Apath);
    if (args->nblocks) free(args->nblocks);
}

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Compute reference distance for sparse matrix-vector multiply.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments are:\n");
    fprintf(f, "  A        path to Matrix Market file for the matrix A\n");
    fprintf(f, "\n");
    fprintf(f, " Other options are:\n");
#ifdef HAVE_LIBZ
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    /* fprintf(f, "  --separate-diagonal    store diagonal nonzeros separately\n"); */
    fprintf(f, "  -q, --quiet            do not print Matrix Market output\n");
    fprintf(f, "  -v, --verbose          be more verbose\n");
    fprintf(f, "\n");
    fprintf(f, "  -h, --help             display this help and exit\n");
    fprintf(f, "  --version              display version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s\n", program_name, program_version);
    fprintf(f, "row/column offsets: %ld-bit\n", sizeof(idx_t)*CHAR_BIT);
#ifdef _OPENMP
    fprintf(f, "OpenMP: yes (%d)\n", _OPENMP);
#else
    fprintf(f, "OpenMP: no\n");
#endif
#ifdef HAVE_LIBZ
    fprintf(f, "zlib: yes ("ZLIB_VERSION")\n");
#else
    fprintf(f, "zlib: no\n");
#endif
    fprintf(f, "\n");
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    const char * s,
    char ** outendptr,
    int base,
    long long int * out_number,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *out_number = number;
    return 0;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int32_t()’ parses a string to produce a number that may be
 * represented as a signed, 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int32_t(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int64_t()’ parses a string to produce a number that may be
 * represented as a signed, 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int64_t(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a double, ‘ERANGE’ is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) { return errno; }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

/**
 * ‘parse_ints()’ parses a string of comma-separated numbers to
 * produce an array of integers.
 *
 * Numbers are parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that each number is
 * parsed correctly. The number of items parsed is stored in ‘N’. If
 * ‘x’ is not ‘NULL’, then it must point to an array of length
 * ‘xsize’, and the integers resulting from parsing are stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_ints(
    int * N,
    int xsize,
    int * x,
    const char * s,
    char ** outendptr,
    int64_t * outbytes_read)
{
    *N = 0;
    while (true) {
        char * endptr;
        int64_t bytes_read = outbytes_read ? *outbytes_read : 0;
        long long int y;
        int err = parse_long_long_int(s, &endptr, 10, &y, &bytes_read);
        if (outendptr) *outendptr = endptr;
        if (outbytes_read) *outbytes_read = bytes_read;
        if (err) return err;
        if (y < INT_MIN || y > INT_MAX) return ERANGE;
        if (x && *N >= xsize) return ENOMEM;
        else if (x) x[*N] = y;
        (*N)++;
        s += bytes_read;
        if (*s == '\0') break;
        if (*s == ',') s++;
        else break;
    }
    return 0;
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs)
{
    int err;
    *nargs = 0;
    (*nargs)++; argv++;

    /* Set default program options. */
    err = program_options_init(args);
    if (err) return err;

    /* Parse program options. */
    int num_positional_arguments_consumed = 0;
    while (*nargs < argc) {
#ifdef HAVE_LIBZ
        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = 1;
            (*nargs)++; argv++; continue;
        }
#endif

        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = 1;
            (*nargs)++; argv++; continue;
        }

        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }

        /* If requested, print program help text. */
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            program_options_free(args);
            program_options_print_help(stdout);
            exit(EXIT_SUCCESS);
        }

        /* If requested, print program version information. */
        if (strcmp(argv[0], "--version") == 0) {
            program_options_free(args);
            program_options_print_version(stdout);
            exit(EXIT_SUCCESS);
        }

        /* Stop parsing options after '--'.  */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /*
         * Parse positional arguments.
         */
        if (num_positional_arguments_consumed == 0) {
            args->Apath = strdup(argv[0]);
            if (!args->Apath) { program_options_free(args); return errno; }
        } else { program_options_free(args); return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }

    if (num_positional_arguments_consumed < 1) {
        program_options_free(args);
        program_options_print_usage(stdout);
        exit(EXIT_FAILURE);
    }
    return 0;
}

/**
 * `timespec_duration()` is the duration, in seconds, elapsed between
 * two given time points.
 */
static double timespec_duration(
    struct timespec t0,
    struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

enum streamtype
{
    stream_stdio,
#ifdef HAVE_LIBZ
    stream_zlib,
#endif
};

union stream
{
    FILE * f;
#ifdef HAVE_LIBZ
    gzFile gzf;
#endif
};

void stream_close(
    enum streamtype streamtype,
    union stream s)
{
    if (streamtype == stream_stdio) {
        fclose(s.f);
#ifdef HAVE_LIBZ
    } else if (streamtype == stream_zlib) {
        gzclose(s.gzf);
#endif
    }
}

/**
 * ‘freadline()’ reads a single line from a stream.
 */
static int freadline(
    char * linebuf,
    size_t line_max,
    enum streamtype streamtype,
    union stream stream)
{
    if (streamtype == stream_stdio) {
        char * s = fgets(linebuf, line_max+1, stream.f);
        if (!s && feof(stream.f)) return -1;
        else if (!s) return errno;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
        return 0;
#ifdef HAVE_LIBZ
    } else if (streamtype == stream_zlib) {
        char * s = gzgets(stream.gzf, linebuf, line_max+1);
        if (!s && gzeof(stream.gzf)) return -1;
        else if (!s) return errno;
        int n = strlen(s);
        if (n > 0 && n == line_max && s[n-1] != '\n') return EOVERFLOW;
        return 0;
#endif
    } else { return EINVAL; }
}

enum mtxobject
{
    mtxmatrix,
    mtxvector,
};

enum mtxformat
{
    mtxarray,
    mtxcoordinate,
};

enum mtxfield
{
    mtxreal,
    mtxinteger,
    mtxpattern,
};

enum mtxsymmetry
{
    mtxgeneral,
    mtxsymmetric,
};

static int mtxfile_fread_header(
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxfield * field,
    enum mtxsymmetry * symmetry,
    idx_t * num_rows,
    idx_t * num_columns,
    int64_t * num_nonzeros,
    enum streamtype streamtype,
    union stream stream,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;

    /* read and parse header line */
    int err = freadline(linebuf, line_max, streamtype, stream);
    if (err) { free(linebuf); return err; }
    char * s = linebuf;
    char * t = s;
    if (strncmp("%%MatrixMarket ", t, strlen("%%MatrixMarket ")) == 0) {
        t += strlen("%%MatrixMarket ");
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        t += strlen("matrix ");
        *object = mtxmatrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        t += strlen("vector ");
        *object = mtxvector;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("array ", t, strlen("array ")) == 0) {
        t += strlen("array ");
        *format = mtxarray;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
        *format = mtxcoordinate;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
        *field = mtxreal;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        t += strlen("integer ");
        *field = mtxinteger;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        t += strlen("pattern ");
        *field = mtxpattern;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxgeneral;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxsymmetric;
    } else { free(linebuf); return EINVAL; }
    if (bytes_read) *bytes_read += t-s;
    s = t;

    /* skip lines starting with '%' */
    do {
        if (lines_read) (*lines_read)++;
        err = freadline(linebuf, line_max, streamtype, stream);
        if (err) { free(linebuf); return err; }
        s = t = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_idx_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_idx_t(num_columns, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
        if (bytes_read) (*bytes_read)++;
        s = t+1;
        err = parse_int64_t(num_nonzeros, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_idx_t(num_rows, s, &t, bytes_read);
        if (err) { free(linebuf); return err; }
        if (s == t) { free(linebuf); return EINVAL; }
        if (lines_read) (*lines_read)++;
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int mtxfile_fread_matrix_coordinate(
    enum mtxfield field,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    idx_t * rowidx,
    idx_t * colidx,
    double * a,
    enum streamtype streamtype,
    union stream stream,
    int64_t * lines_read,
    int64_t * bytes_read)
{
    int line_max = sysconf(_SC_LINE_MAX);
    char * linebuf = malloc(line_max+1);
    if (!linebuf) return errno;
    if (field == mtxreal || field == mtxinteger) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_double(&a[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            if (lines_read) (*lines_read)++;
        }
    } else if (field == mtxinteger) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            int x;
            err = parse_int(&x, s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            a[i] = x;
            if (lines_read) (*lines_read)++;
        }
    } else if (field == mtxpattern) {
        for (int64_t i = 0; i < num_nonzeros; i++) {
            int err = freadline(linebuf, line_max, streamtype, stream);
            if (err) { free(linebuf); return err; }
            char * s = linebuf;
            char * t = s;
            err = parse_idx_t(&rowidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t || *t != ' ') { free(linebuf); return EINVAL; }
            if (bytes_read) (*bytes_read)++;
            s = t+1;
            err = parse_idx_t(&colidx[i], s, &t, bytes_read);
            if (err) { free(linebuf); return err; }
            if (s == t) { free(linebuf); return EINVAL; }
            a[i] = 1;
            if (lines_read) (*lines_read)++;
        }
    } else { free(linebuf); return EINVAL; }
    free(linebuf);
    return 0;
}

static int csr_from_coo_size(
    enum mtxsymmetry symmetry,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * __restrict rowidx,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    int64_t * __restrict rowptr,
    int64_t * __restrict csrsize,
    idx_t * __restrict rowsizemin,
    idx_t * __restrict rowsizemax,
    idx_t * __restrict diagsize,
    bool separate_diagonal)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (idx_t i = 0; i < num_rows; i++) rowptr[i] = 0;
    rowptr[num_rows] = 0;
    if (num_rows == num_columns && symmetry == mtxsymmetric && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) { rowptr[rowidx[k]]++; rowptr[colidx[k]]++; }
        }
    } else if (num_rows == num_columns && symmetry == mtxsymmetric && !separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) { rowptr[rowidx[k]]++; rowptr[colidx[k]]++; }
            else { rowptr[rowidx[k]]++; }
        }
    } else if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] != colidx[k]) rowptr[rowidx[k]]++;
        }
    } else { for (int64_t k = 0; k < num_nonzeros; k++) rowptr[rowidx[k]]++; }
    idx_t rowmin = num_rows > 0 ? rowptr[1] : 0;
    idx_t rowmax = 0;
    for (idx_t i = 1; i <= num_rows; i++) {
        rowmin = rowmin <= rowptr[i] ? rowmin : rowptr[i];
        rowmax = rowmax >= rowptr[i] ? rowmax : rowptr[i];
        rowptr[i] += rowptr[i-1];
    }
    if (num_rows == num_columns && separate_diagonal) { rowmin++; rowmax++; }
    *rowsizemin = rowmin;
    *rowsizemax = rowmax;
    *csrsize = rowptr[num_rows];
    *diagsize = (num_rows == num_columns && separate_diagonal) ? num_rows : 0;
    return 0;
}

static int csr_from_coo(
    enum mtxsymmetry symmetry,
    idx_t num_rows,
    idx_t num_columns,
    int64_t num_nonzeros,
    const idx_t * __restrict rowidx,
    const idx_t * __restrict colidx,
    const double * __restrict a,
    int64_t * __restrict rowptr,
    int64_t csrsize,
    idx_t rowsizemin,
    idx_t rowsizemax,
    idx_t * __restrict csrcolidx,
    double * __restrict csra,
    double * __restrict csrad,
    bool separate_diagonal)
{
    if (num_rows == num_columns && symmetry == mtxsymmetric && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            if (rowidx[k] == colidx[k]) { csrad[rowidx[k]-1] += a[k]; }
            else {
                idx_t i = rowidx[k]-1, j = colidx[k]-1;
                csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++;
                csrcolidx[rowptr[j]] = i; csra[rowptr[j]] = a[k]; rowptr[j]++;
            }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else if (num_rows == num_columns && symmetry == mtxsymmetric && !separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1, j = colidx[k]-1;
            csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++;
            if (i != j) { csrcolidx[rowptr[j]] = i; csra[rowptr[j]] = a[k]; rowptr[j]++; }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else if (num_rows == num_columns && separate_diagonal) {
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1, j = colidx[k]-1;
            if (i == j) { csrad[i] += a[k]; }
            else { csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++; }
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
    } else {
        /* simpler, serial version: */
        /* for (int64_t k = 0; k < num_nonzeros; k++) { */
        /*     idx_t i = rowidx[k]-1, j = colidx[k]-1; */
        /*     csrcolidx[rowptr[i]] = j; csra[rowptr[i]] = a[k]; rowptr[i]++; */
        /* } */
        /* for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1]; */
        /* rowptr[0] = 0; */

        int64_t * __restrict perm = malloc(num_nonzeros * sizeof(int64_t));
        if (!perm) { return errno; }
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t k = 0; k < num_nonzeros; k++) perm[k] = 0;
        for (int64_t k = 0; k < num_nonzeros; k++) {
            idx_t i = rowidx[k]-1;
            perm[rowptr[i]++] = k;
        }
        for (idx_t i = num_rows; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t k = 0; k < num_nonzeros; k++) {
            csrcolidx[k] = colidx[perm[k]]-1;
            csra[k] = a[perm[k]];
        }
        free(perm);
    }
    return 0;
}

static int csrgemvreferencedist(
    int64_t * referencedist,
    idx_t num_rows,
    idx_t num_columns,
    int64_t csrsize,
    const int64_t * __restrict rowptr,
    const idx_t * __restrict colidx)
{
    /* sort nonzeros by their column offsets in ascending order using
     * a (stable) counting sort */
    int64_t * colptr = malloc((num_columns+1) * sizeof(int64_t));
    if (!colptr) return errno;
    int64_t * perm = malloc(csrsize * sizeof(int64_t));
    if (!perm) { free(colptr); return errno; }

    /* count the number of nonzeros in each column and offsets to
     * first nonzero in each column */
    for (idx_t i = 0; i < num_columns; i++) colptr[i] = 0;
    colptr[num_columns] = 0;
    for (int64_t k = 0; k < csrsize; k++) colptr[colidx[k]+1]++;
    for (idx_t i = 1; i <= num_columns; i++) colptr[i] += colptr[i-1];

    /* find the sorting permutation to place nonzeros in ascending
     * order by column offset */
    for (int64_t k = 0; k < csrsize; k++) perm[k] = 0;
    for (idx_t i = 0; i < num_rows; i++) {
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
            idx_t j = colidx[k];
            perm[colptr[j]++] = k;
        }
    }
    for (idx_t i = num_columns; i > 0; i--) colptr[i] = colptr[i-1];
    colptr[0] = 0;

    /* use the sorting permutation to compute the reference distance
     * of each nonzero */
    for (int64_t k = 0; k < csrsize-1; k++) {
        idx_t j = colidx[perm[k]];
        idx_t jp1 = colidx[perm[k+1]];
        if (j == jp1) referencedist[perm[k]] = perm[k+1]-perm[k]-1;
        else referencedist[perm[k]] = -1;
    }
    if (csrsize > 0) referencedist[csrsize-1] = -1;
    free(perm); free(colptr);
    return 0;
}

/**
 * ‘main()’.
 */
int main(int argc, char *argv[])
{
    int err;
    struct timespec t0, t1;
    setlocale(LC_ALL, "");

    /* Set program invocation name. */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);

    /* 1. Parse program options. */
    struct program_options args;
    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        return EXIT_FAILURE;
    }

    /* 2. read the matrix from a Matrix Market file */
    if (args.verbose > 0) {
        fprintf(stderr, "mtxfile_read: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    enum streamtype streamtype;
    union stream stream;
#ifdef HAVE_LIBZ
    if (!args.gzip) {
#endif
        streamtype = stream_stdio;
        if ((stream.f = fopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
#ifdef HAVE_LIBZ
    } else {
        streamtype = stream_zlib;
        if ((stream.gzf = gzopen(args.Apath, "r")) == NULL) {
            fprintf(stderr, "%s: %s: %s\n",
                    program_invocation_short_name, args.Apath, strerror(errno));
            program_options_free(&args);
            return EXIT_FAILURE;
        }
    }
#endif

    enum mtxobject object;
    enum mtxformat format;
    enum mtxfield field;
    enum mtxsymmetry symmetry;
    idx_t num_rows;
    idx_t num_columns;
    int64_t num_nonzeros = 0;
    int64_t lines_read = 0;
    int64_t bytes_read = 0;
    err = mtxfile_fread_header(
        &object, &format, &field, &symmetry,
        &num_rows, &num_columns, &num_nonzeros,
        streamtype, stream, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    idx_t * rowidx = malloc(num_nonzeros * sizeof(idx_t));
    if (!rowidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    idx_t * colidx = malloc(num_nonzeros * sizeof(idx_t));
    if (!colidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * a = malloc(num_nonzeros * sizeof(double));
    if (!a) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(colidx); free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    err = mtxfile_fread_matrix_coordinate(
        field, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        streamtype, stream, &lines_read, &bytes_read);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s:%"PRId64": %s\n",
                program_invocation_short_name,
                args.Apath, lines_read+1, strerror(err));
        free(a); free(colidx); free(rowidx);
        stream_close(streamtype, stream);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                timespec_duration(t0, t1),
                1.0e-6 * bytes_read / timespec_duration(t0, t1));
    }
    stream_close(streamtype, stream);

    /* 3. convert to CSR format */
    if (args.verbose > 0) {
        fprintf(stderr, "csr_from_coo: ");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * csrrowptr = malloc((num_rows+1) * sizeof(int64_t));
    if (!csrrowptr) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    int64_t csrsize;
    idx_t rowsizemin, rowsizemax;
    idx_t diagsize;
    err = csr_from_coo_size(
        symmetry, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, &csrsize, &rowsizemin, &rowsizemax, &diagsize,
        args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    idx_t * csrcolidx = malloc(csrsize * sizeof(idx_t));
    if (!csrcolidx) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    #pragma omp parallel for
    for (idx_t i = 0; i < num_rows; i++) {
        for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
            csrcolidx[k] = 0;
    }
#endif
    double * csra = malloc(csrsize * sizeof(double));
    if (!csra) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrcolidx);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    double * csrad = malloc(diagsize * sizeof(double));
    if (!csrad) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csra); free(csrcolidx);
        free(csrrowptr); free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
#ifdef _OPENMP
    if (diagsize > 0) {
        #pragma omp parallel for
        for (idx_t i = 0; i < num_rows; i++) csrad[i] = 0;
    }
    #pragma omp parallel for
    for (idx_t i = 0; i < num_rows; i++) {
        for (int64_t k = csrrowptr[i]; k < csrrowptr[i+1]; k++)
            csra[k] = 0;
    }
#endif
    err = csr_from_coo(
        symmetry, num_rows, num_columns, num_nonzeros, rowidx, colidx, a,
        csrrowptr, csrsize, rowsizemin, rowsizemax, csrcolidx, csra, csrad,
        args.separate_diagonal);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    free(a); free(colidx); free(rowidx);

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "%'.6f seconds, %'"PRIdx" rows, %'"PRIdx" columns, %'"PRId64" nonzeros"
                ", %'"PRIdx" to %'"PRIdx" nonzeros per row",
                timespec_duration(t0, t1), num_rows, num_columns, csrsize+diagsize, rowsizemin, rowsizemax);
        fputc('\n', stderr);
    }

    /* 5. compute reference distance */
    if (args.verbose > 0) {
        fprintf(stderr, "computing reference distance:\n");
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    int64_t * referencedist = malloc(csrsize * sizeof(int64_t));
    if (!referencedist) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        free(a); free(colidx); free(rowidx);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    err = csrgemvreferencedist(
        referencedist, num_rows, num_columns,
        csrsize, csrrowptr, csrcolidx);
    if (err) {
        if (args.verbose > 0) fprintf(stderr, "\n");
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        free(referencedist);
        free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    if (args.verbose > 0) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "done computing reference distance in %'.6f seconds\n", timespec_duration(t0, t1));
    }

    /* 6. write the reference time to standard output */
    if (!args.quiet) {
        if (args.verbose > 0) {
            fprintf(stderr, "writing reference distance:\n");
            clock_gettime(CLOCK_MONOTONIC, &t0);
        }

        /* fprintf(stdout, "%%%%MatrixMarket vector array integer general\n"); */
        /* fprintf(stdout, "%"PRId64"\n", csrsize); */
        for (int64_t k = 0; k < csrsize; k++) fprintf(stdout, "%"PRId64"\n", referencedist[k]);
        if (args.verbose > 0) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "done writing reference distance in %'.6f seconds\n", timespec_duration(t0, t1));
        }
    }

    free(referencedist);
    free(csrad); free(csra); free(csrcolidx); free(csrrowptr);
    program_options_free(&args);
    return EXIT_SUCCESS;
}
