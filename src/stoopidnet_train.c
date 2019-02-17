#include "mnist_loader.h"
#include "stoopidnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("Usage: %s <stoopidnet input file OR \"null\"> <stoopidnet output file> "
               "<mnist data> <mnist labels> <randseed>\n", argv[0]);
        return -1;
    }

    // srand
    srand(strtol(argv[5], NULL, 10));

    // load files
    stoopidnet_t* net;
    if (!strcmp(argv[1], "null")) {
        net = stoopidnet_create(784);
        stoopidnet_add_fc_layer(net, 20);
        stoopidnet_add_fc_layer(net, 10);
    } else {
        net = stoopidnet_load_from_file(argv[1]);
        if (net == NULL) {
            return -1;
        }
    }

    double** pics;
    double** labels;
    int npics = load_data_file_doubles(argv[3], &pics);
    int nlabels = load_label_file_doubles(argv[4], &labels);

    if ((npics != nlabels) || (nlabels == 0)) {
        fprintf(stderr, "Something went wrong loading the mnist files\n");
        return -1;
    }

    // train.
    stoopidnet_training_parameters_t train_params = { 3.0, 50 };
    stoopidnet_train(net, &train_params, npics, pics, labels);

    // store the final network
    stoopidnet_store_to_file(net, argv[2]);

    return 0;
}
