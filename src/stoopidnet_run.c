#include "mnist_loader.h"
#include "stoopidnet.h"

#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: %s <stoopidnet input file> <mnist data> <mnist labels>\n", argv[0]);
        return -1;
    }

    // load files
    stoopidnet_t* net = stoopidnet_load_from_file(argv[1]);
    if (net == NULL) {
        return -1;
    }
    double** pics;
    double** labels;
    int npics = load_data_file_doubles(argv[2], &pics);
    int nlabels = load_label_file_doubles(argv[3], &labels);

    if ((npics != nlabels) || (nlabels == 0)) {
        fprintf(stderr, "Something went wrong loading the mnist files\n");
        return -1;
    }

    // run
    printf("  index | label | result | confidence\n"
           "--------|-------|--------|--------------\n");

    int goodcount = 0;
    for (int i = 0; i < npics; i++) {
        double* output;
        stoopidnet_evaluate(net, pics[i], &output);
        int label  = maxidx(labels[i], 10);
        int result = maxidx(output, 10);

        printf("%7i |     %i |      %i |   %1.5lf\n", i, label, result, output[result]);

        if (label == result) {
            goodcount++;
        }

        free(output);
    }
    printf("accuracy: %i / %i\r\n", goodcount, npics);
}
