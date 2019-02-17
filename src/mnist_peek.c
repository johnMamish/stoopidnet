#include <stdio.h>
#include <stdlib.h>

#include "mnist_loader.h"

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("pipes a pam image to stdout representing the i-th image in the given "
               "mnist data file\n");
        printf("Usage: %s <mnist images file> <mnist labels file> <idx>\n", argv[0]);
        return -1;
    }

    uint32_t idx = strtol(argv[3], NULL, 10);

    // load pics
    uint32_t wh[2] = { 0 };
    uint8_t** pics;
    int numpics = load_data_file(argv[1], wh, &pics);

    // load labels
    uint8_t* labels;
    int numlabels = load_label_file(argv[2], &labels);

    // check
    if (numpics != numlabels) {
        printf("files don't match\n");
        return -1;
    }

    if (idx >= numpics) {
        printf("index too big. you only have %i images\n", numpics);
        return -1;
    }

    // output
    printf("P2\n");
    printf("# This image is at index %i and we expect it to be a '%i'\n", idx, (int)labels[idx]);
    printf("%i %i\n", wh[0], wh[1]);
    printf("255\n");
    int count = 0;
    for (int y = 0; y < wh[1]; y++) {
        for (int x = 0; x < wh[0]; x++, count++) {
            printf("%i ", (int)((pics[idx])[count]));
        }
        printf("\n");
    }
}
