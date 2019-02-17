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


/**
 * serialization format:
 *
 * uint32_t num layers
 * uint32_t[num_layers] nodes_per_layer
 * double[num_layers - 1][] biases
 * double[num_layers - 1][nodes_per_layer[l]][nodes_per_layer[l - 1]] weights
 *
 * Returns size in bytes.
 */
uint32_t stoopidnet_serialize(stoopidnet_t* net, uint8_t** _target)
{
    uint32_t capacity = 1024;
    uint32_t size = 0;
    uint8_t* target = malloc(capacity);

    // First encode the number of layers
    *((uint32_t*)(target + size)) = net->num_layers;
    size += 4;

    // Encode the nodes per layer array.
    for (int i = 0; i < net->num_layers; i++) {
        if ((size + sizeof(uint32_t)) >= capacity) {
            capacity *= 2;
            target = realloc(target, capacity);
        }

        *((uint32_t*)(target + size)) = net->layer_sizes[i];
        size += sizeof(uint32_t);
    }

    // Encode the biases arrays
    for (int l = 0; l < (net->num_layers - 1); l++) {
        for (int j = 0; j < net->layer_sizes[l + 1]; j++) {
            if ((size + sizeof(double)) >= capacity) {
                capacity *= 2;
                target = realloc(target, capacity);
            }

            *((double*)(target + size)) = (net->biases[l])[j];
            size += sizeof(double);
        }
    }

    // Encode the weights
    for (int layer = 0; layer < (net->num_layers - 1); layer++) {
        int numweights = (net->layer_sizes[layer] * net->layer_sizes[layer + 1]);
        for (int weight_idx = 0; weight_idx < numweights; weight_idx++) {
            if ((size + sizeof(double)) >= capacity) {
                capacity *= 2;
                target = realloc(target, capacity);
            }

            *((double*)(target + size)) = (net->weights[layer])[weight_idx];
            size += sizeof(double);
        }
    }

    *_target = target;
    return size;
}


stoopidnet_t* stoopidnet_deserialize(uint8_t *data, uint32_t datalen)
{
    uint32_t idx = 0;
    stoopidnet_t* net = NULL;
    if (idx >= datalen) {
        goto failed_1;
    }

    uint32_t num_layers = *((uint32_t*)(data + idx));
    idx += sizeof(uint32_t);

    net = calloc(1, sizeof(stoopidnet_t));
    net->num_layers  = num_layers;
    net->layer_sizes = calloc(num_layers, sizeof(uint32_t));
    net->weights     = calloc(num_layers, sizeof(double*));
    net->biases      = calloc(num_layers, sizeof(double*));

    // unpack all layer sizes.
    for (int i = 0; i < net->num_layers; i++) {
        if (idx >= datalen) {
            goto failed_2;
        }
        net->layer_sizes[i] = *((uint32_t*)(data + idx));
        idx += sizeof(uint32_t);
    }

    // get all biases.
    for (int layer = 0; layer < (net->num_layers - 1); layer++) {
        net->biases[layer] = malloc(net->layer_sizes[layer + 1] * sizeof(double));
        for (int j = 0; j < net->layer_sizes[layer + 1]; j++) {
            if (idx >= datalen) {
                goto failed_3;
            }

            (net->biases[layer])[j] = *((double*)(data + idx));
            idx += sizeof(double);
        }
    }

    // get all weights.
    for (int layer = 0; layer < (net->num_layers - 1); layer++) {
        int numweights = (net->layer_sizes[layer] * net->layer_sizes[layer + 1]);
        net->weights[layer] = malloc(numweights * sizeof(double));
        for (int weight_idx = 0; weight_idx < numweights; weight_idx++) {
            if (idx > datalen) {
                goto failed_4;
            }

            (net->weights[layer])[weight_idx] = *((double*)(data + idx));
            idx += sizeof(double);
        }
    }

    if (idx != datalen) {
        goto failed_4;
    }

    return net;

failed_4:
    for (int i = 0; i < (net->num_layers - 1); i++) {
        free(net->weights[i]);
    }

failed_3:
    for (int i = 0; i < (net->num_layers - 1); i++) {
        free(net->biases[i]);
    }

failed_2:
    free(net->layer_sizes);
    free(net->biases);
    free(net->weights);
    free(net);

failed_1:
    fprintf(stderr, "Given buffer is wrong length for deseralization.\n");

    return NULL;
}


stoopidnet_t* stoopidnet_load_from_file(const char* file)
{
    FILE* fp = fopen(file, "rb");

    // get file length and load it.
    fseek(fp, 0L, SEEK_END);
    uint32_t sz = ftell(fp);
    rewind(fp);
    uint8_t* data = malloc(sz);
    fread(data, sizeof(uint8_t), sz, fp);

    stoopidnet_t* net = stoopidnet_deserialize(data, sz);
    return net;
}


int stoopidnet_store_to_file(stoopidnet_t* net, const char* file)
{
    int retval = 0;

    FILE* fp = fopen(file, "wb");
    uint8_t* data;
    uint32_t len = stoopidnet_serialize(net, &data);

    if (len == 0) {
        fprintf(stderr, "Tried to store invalid net to file\n");
        retval = -1;
        goto cleanup;
    }

    size_t writelen = fwrite(data, sizeof(uint8_t), len, fp);
    fclose(fp);
    if (writelen != len) {
        fprintf(stderr, "Error writing network to file\n");
        retval = -1;
        goto cleanup;
    }

cleanup:
    free(data);
    return retval;
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
    net->biases[net->num_layers - 2] = malloc(net->layer_sizes[net->num_layers - 1] * sizeof(double));
    for (int i = 0; i < net->layer_sizes[net->num_layers - 1]; i++) {
        (net->biases[net->num_layers - 2])[i] = 2.0 * ((((double)rand()) / ((double)RAND_MAX)) - 0.5);
    }

    int nweights = net->layer_sizes[net->num_layers - 1] * net->layer_sizes[net->num_layers - 2];
    net->weights[net->num_layers - 2] = malloc(nweights * sizeof(double));

    for (int i = 0; i < nweights; i++) {
        double randn = 2.0 * ((((double)rand()) / ((double)RAND_MAX)) - 0.5);
        (net->weights[net->num_layers - 2])[i] = randn;
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


uint32_t stoopidnet_get_num_layers(stoopidnet_t* net)
{
    return net->num_layers;
}


void stoopidnet_set_layer_weights(stoopidnet_t* net, uint32_t layer_idx, double* weights)
{

}


void stoopidnet_evaluate(stoopidnet_t* net, double* input, double** output)
{
    double *activation = malloc(net->layer_sizes[0] * sizeof(double));
    memcpy(activation, input, sizeof(double) * net->layer_sizes[0]);
    double *activiation_next = NULL;

    for (int l = 1; l < net->num_layers; l++) {
        activiation_next = calloc(net->layer_sizes[l], sizeof(double));

        // calculate next layer into activiation_next
        for (int j = 0; j < net->layer_sizes[l]; j++) {
            double accum = 0.;
            for (int k = 0; k < net->layer_sizes[l - 1]; k++) {
                int idx = (j * net->layer_sizes[l - 1]) + k;
                accum += (net->weights[l - 1])[idx] * activation[k];
            }
            activiation_next[j] = sigmoid(accum - (net->biases[l - 1])[j]);
        }

        // copy new layer into activiation.
        free(activation);
        activation = activiation_next;
    }

    //activation now contains result. It's the caller's responsibility to free.
    *output = activation;
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
