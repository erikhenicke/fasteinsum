#include "hello_bmm.h"

double *add_arrays(double *a, double *b, double *c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}