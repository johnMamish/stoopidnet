#include "stoopidnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <pam.h>

#define PAM_MEMBER_OFFSET(mbrname)                    \
  ((unsigned long int)(char*)&((struct pam *)0)->mbrname)
#define PAM_MEMBER_SIZE(mbrname) \
  sizeof(((struct pam *)0)->mbrname)
#define PAM_STRUCT_SIZE(mbrname) \
(PAM_MEMBER_OFFSET(mbrname) + PAM_MEMBER_SIZE(mbrname))



typedef struct image {
    int width;
    int height;
    int depth;

    // supports images with up to 4 components.
    double* components;
} image_t;


static int maxidx(double* vec, int len)
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

image_t* image_create_from_pam(const char* file, char* argv0)
{
    image_t* result = calloc(1, sizeof(image_t*));

    pm_init(argv0, 0);

    struct pam image;
    tuple * tuplerow;
    unsigned int row;

    FILE* fp = fopen(file, "r");
    pnm_readpaminit(fp, &image, PAM_STRUCT_SIZE(tuple_type));

    result->height = image.height;
    result->width  = image.width;
    result->depth  = image.depth;
    if (result->depth > 4) {
        goto _fail_cleanup1;
    }

    result->components = calloc(result->height * result->width, sizeof(double));
    tuplerow = pnm_allocpamrow(&image);
    for (row = 0; row < image.height; row++) {
        unsigned int column;
        pnm_readpamrow(&image, tuplerow);
        for (column = 0; column < image.width; ++column) {
            unsigned int plane;
            for (plane = 0; plane < image.depth; ++plane) {
                result->components[(row * result->width) + column] = 1.f -
                    ((double)tuplerow[column][plane] / ((double)255.0));
            }
        }
    }
    pnm_freepamrow(tuplerow);

    return result;

//_fail_cleanup0:
//    free(result->components);
//    pnm_freepamrow(tuplerow);

_fail_cleanup1:
    free(result);
    return NULL;
}


void image_destroy(image_t* image)
{
    free(image->components);
    free(image);
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <stoopidnet input file> <pgm image>\n", argv[0]);
        return -1;
    }

    image_t* im;
    if ((im = image_create_from_pam(argv[2], argv[0])) == NULL) {
        printf("error opening image %s", argv[2]);
        return -1;
    }

    stoopidnet_t* net = stoopidnet_load_from_file(argv[1]);
    if (net == NULL) {
        return -1;
    }

    double* output;
    stoopidnet_evaluate(net, im->components, &output);
    int result = maxidx(output, 10);
    printf("%i\n", result);
    for (int i = 0; i < 10; i++)
        printf("%f ", output[i]);
    printf("\n");
    return 0;
}
