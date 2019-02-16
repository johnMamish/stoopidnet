#include "stoopidnet.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct stoopidnet
{
    uint32_t num_layers;
    uint32_t* layer_sizes;

    /**
     * weights[i] has size layer_sizes[i] * layer_sizes[i + 1] and represents weights between layers
     * i and i + 1.
     *
     * weights[i] has weights w_0,0, w_0,1, w_0,2... w_0,{layer_sizes[i]} adjencent to each other.
     */
    double** weights;

    /**
     * biases[i] corresponds to layer i + 1.
     */
    double** biases;
};

////////////////////////////////////////////////////////////////
// static helper function decls
////////////////////////////////////////////////////////////////
/**
 * Calculates the sigmoid function for a given value.
 */
static double sigmoid(double z);


stoopidnet_t* stoopidnet_create(uint32_t num_input_nodes)
{
    stoopidnet_t *net = calloc(1, sizeof(stoopidnet_t));
    net->num_layers = 1;
    // clown shoes allocation so re-alloc will generally be ok.
    net->layer_sizes = malloc(8 * sizeof(uint32_t));
    net->weights = malloc(8 * sizeof(double*));
    net->biases  = malloc(8 * sizeof(double*));

    // setup values
    net->layer_sizes[0] = num_input_nodes;

    return net;
}


void stoopidnet_destroy(stoopidnet_t* net)
{
    for (int i = 0; i < net->num_layers; i++) {
        free(net->biases[i]);
    }

    for (int i = 0; i < (net->num_layers - 1); i++) {
        free(net->weights[i]);
    }
    free(net->layer_sizes);
    free(net);
}


void stoopidnet_add_fc_layer(stoopidnet_t* net, uint32_t num_nodes)
{
    net->num_layers++;

    // ======= allocate. =======
    net->layer_sizes = realloc(net->layer_sizes, net->num_layers * sizeof(uint32_t));
    net->weights     = realloc(net->weights, (net->num_layers - 1) * sizeof(double*));
    net->biases      = realloc(net->biases, (net->num_layers - 1) * sizeof(double*));

    // ======= fill. =======
    net->layer_sizes[net->num_layers - 1] = num_nodes;

    // Set up the biases and weights to be randomly distributed on [-1, 1]
    for (int i = 0; i < net->layer_sizes[net->num_layers - 1]; i++) {
        (net->biases[net->num_layers - 1])[i] = 2.0 * ((((double)rand()) / ((double)RAND_MAX)) - 0.5);
        for (int j = 0; j < net->layer_sizes[net->num_layers - 2]; j++) {
            int idx = (i * net->layer_sizes[net->num_layers - 2]) + j;
            double randn = 2.0 * ((((double)rand()) / ((double)RAND_MAX)) - 0.5);
            (net->weights[net->num_layers - 1])[idx] = randn;
        }
    }
}


void stoopidnet_add_fc_layer_with_starting_weights(stoopidnet_t* net,
                                                   uint32_t num_nodes,
                                                   double* weights)
{
    printf("XXX TODO");
    stoopidnet_add_fc_layer(net, num_nodes);
}


uint32_t stoopidnet_get_num_nodes_in_layer(stoopidnet_t* net, uint32_t layer_idx)
{
    assert(layer_idx < net->num_layers);
    return net->layer_sizes[layer_idx];
}


void stoopidnet_set_layer_weights(stoopidnet_t* net, uint32_t layer_idx, double* weights)
{

}


void stoopidnet_evaluate(stoopidnet_t* net, double* input, double** output)
{
    double *z = malloc(net->layer_sizes[0] * sizeof(double));
    memcpy(z, input, sizeof(double) * net->layer_sizes[0]);
    double *znext = NULL;

    for (int l = 1; l < net->num_layers; l++) {
        znext = malloc(net->layer_sizes[l]);

        // calculate next layer into znext
        for (int j = 0; j < net->layer_sizes[l]; j++) {
            double accum = 0.;
            for (int k = 0; k < net->layer_sizes[l - 1]; k++) {
                int idx = (j * net->layer_sizes[l - 1]) + k;
                accum += (net->weights[l - 1])[idx] * z[k];
            }
            znext[j] = sigmoid(accum - (net->biases[l - 1])[j]);
        }

        // copy new layer into z.
        free(z);
        z = znext;
    }

    //z now contains result. It's the caller's responsibility to free.
    *output = z;
}


void stoopidnet_train(stoopidnet_t* net,
                      const stoopidnet_training_parameters_t* params,
                      uint32_t n_inputs,
                      double** inputs,
                      double** outputs)
{

}


////////////////////////////////////////////////////////////////
// static function impl
////////////////////////////////////////////////////////////////
static double sigmoid(double z)
{
    return 1. / (1. + exp(-z));
}
