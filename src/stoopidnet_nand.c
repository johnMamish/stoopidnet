#include "stoopidnet.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    srand(0);

    // create stoopidnet
    stoopidnet_t *net;
    net = stoopidnet_create(2);
    stoopidnet_add_fc_layer(net, 2);
    stoopidnet_add_fc_layer(net, 1);

    int batch_size = 50;
    stoopidnet_training_parameters_t train_params = { 0.01, batch_size };

    // generate data and run
    double** data = malloc(sizeof(double*) * batch_size);
    double** labels = malloc(sizeof(double*) * batch_size);
    for (int i = 0; i < 100000; i++) {
        for (int j = 0; j < batch_size; j++) {
            data[j] = malloc(2 * sizeof(double));
            data[j][0] = floor(((double)rand() / (double)RAND_MAX) + 0.5);
            data[j][1] = floor(((double)rand() / (double)RAND_MAX) + 0.5);
            labels[j] = malloc(1 * sizeof(double));
            labels[j][0] = (double)(!!((data[j][0] > 0.5) && (data[j][1] > 0.5)));
        }
        stoopidnet_train(net, &train_params, batch_size, data, labels);
        for (int j = 0; j < batch_size; j++) {
            free(data[j]);
            free(labels[j]);
        }
    }

    return 0;
}
