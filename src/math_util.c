#include "math_util.h"

int maxidx(double* vec, int len)
{
    double maxval = vec[0];
    double maxidx = 0;

    for (int i = 1; i < len; i++) {
        if (vec[i] > maxval) {
            maxval = vec[i];
            maxidx = i;
        }
    }

    return maxidx;
}
