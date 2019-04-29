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

typedef struct stoopidnet_forward_prop_results
{
    /**
     * a[0] contains input activations
     */
    double** a;

    /**
     * z[0] contains nothing, z[1] contains z for layer 1 (first hidden layer).
     */
    double** z;
} stoopidnet_forward_prop_results_t;

////////////////////////////////////////////////////////////////
// static helper function decls
////////////////////////////////////////////////////////////////

static double box_mueller_norm()
{
    double u[2] = { ((double)rand()) / ((double)RAND_MAX), ((double)rand()) / ((double)RAND_MAX) };
    return sqrt(-2 * log(u[0])) * cos(2 * 3.141592653 * u[1]);
}

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

/**
 * Calculates the sigmoid function for a given value.
 */
static double sigmoid(double z);

/**
 * Calculates d sigmoid(z) / dz
 */
static double sigmoid_prime(double z);

/**
 * Returns an allocated list of numbers [0, n) that have been shuffled.
 */
static int* gen_shuffled_ints(int n);

/**
 *
 */
static void doubles_memset(double* d, int numel, double val);

//static void stoopidnet_forward_prop_results_destroy(stoopidnet_t* net,
//                                                    stoopidnet_forward_prop_results_t* res);

/**
 * Fundamentally does the same thing as eval, but stores some intermediate results that will be
 * useful for calculating backprop.
 */
static stoopidnet_forward_prop_results_t* stoopidnet_forward_prop(stoopidnet_t* net,
                                                                  double *input);


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
    int foo = fread(data, sizeof(uint8_t), sz, fp);

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
        (net->biases[net->num_layers - 2])[i] = box_mueller_norm();
    }

    int nweights = net->layer_sizes[net->num_layers - 1] * net->layer_sizes[net->num_layers - 2];
    net->weights[net->num_layers - 2] = malloc(nweights * sizeof(double));

    for (int i = 0; i < nweights; i++) {
        (net->weights[net->num_layers - 2])[i] = box_mueller_norm();
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
            activiation_next[j] = sigmoid(accum + (net->biases[l - 1])[j]);
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
    // shuffle training examples
    int* shuffle = gen_shuffled_ints(n_inputs);

    // setup empty arrays for accumulating average of gradients over training mini-batch
    double** weight_grads = calloc(net->num_layers - 1, sizeof(double*));
    double** bias_grads   = calloc(net->num_layers - 1, sizeof(double*));
    for (int i = 0; i < net->num_layers - 1; i++) {
        weight_grads[i] = malloc(net->layer_sizes[i] * net->layer_sizes[i + 1] * sizeof(double));
        bias_grads[i] = malloc(net->layer_sizes[i + 1] * sizeof(double));
    }

    // do mini batches
    int prev_print = 0;
    double* grad_cost = malloc(net->layer_sizes[net->num_layers - 1] * sizeof(double));
    for (int i = 0; i < n_inputs;) {
        // reset gradient vectors
        for (int j = 0; j < net->num_layers - 1; j++) {
            doubles_memset(weight_grads[j], net->layer_sizes[j] * net->layer_sizes[j + 1], 0.0);
            doubles_memset(bias_grads[j], net->layer_sizes[j + 1], 0.0);
        }

        // actual backprop happens here
        // allocate array for layer-by-layer error
        double** layer_error = calloc(net->num_layers, sizeof(double*));
        for (int j = 1; j < net->num_layers; j++) {
            layer_error[j] = malloc(net->layer_sizes[j] * sizeof(double));
        }

        for (int j = 0; j < params->batch_size; j++, i++) {
            // first run network forward and cache z-values and a-values.
            stoopidnet_forward_prop_results_t* fp = stoopidnet_forward_prop(net,
                                                                            inputs[shuffle[i]]);

            // backpropagate
            // final layer is special case:
            // BP1: d_L = grada(C) hadamard sig'(z_L)
            int backprop_layer = net->num_layers - 1;
            for (int k = 0; k < net->layer_sizes[net->num_layers - 1]; k++) {
                layer_error[backprop_layer][k] = ((fp->a[backprop_layer][k] -
                                                       (outputs[shuffle[i]])[k]) *
                                                      sigmoid_prime(fp->a[backprop_layer][k]));
                if (isnan(layer_error[backprop_layer][k])) {
                    printf("nan!!!\r\n");
                }
            }
            backprop_layer--;
            // calc BP2: d_l = ((w_{l+1})_T * d_{l+1}) hadamard sig'(z_l)
            for (;backprop_layer > 0; backprop_layer--) {
                for (int k = 0; k < net->layer_sizes[backprop_layer]; k++) {
                    layer_error[backprop_layer][k] = 0.;
                    for (int l = 0; l < net->layer_sizes[backprop_layer + 1]; l++) {
                        int idx = (l * net->layer_sizes[backprop_layer]) + k;
                        layer_error[backprop_layer][k] +=
                            (net->weights[backprop_layer][idx] *
                             layer_error[backprop_layer + 1][l]);
                    }
                    layer_error[backprop_layer][k] *= sigmoid_prime(fp->z[backprop_layer][k]);
                    if (isnan(layer_error[backprop_layer][k])) {
                        *((volatile char*)0) = 0;
                    }
                }
            }

            // add to gradient vectors.
            for (int k = 1; k < net->num_layers; k++) {
                for (int l = 0; l < net->layer_sizes[k]; l++) {
                    bias_grads[k - 1][l] += layer_error[k][l];
                    for (int m = 0; m < net->layer_sizes[k - 1]; m++) {
                        int idx = (l * net->layer_sizes[k - 1]) + m;
                        weight_grads[k - 1][idx] += fp->a[k - 1][m] * layer_error[k][l];
                    }
                }
            }
        }

        // update network state with gradient.
        double lrate = (params->learn_rate / ((double)params->batch_size));
        for (int layer = 0; layer < net->num_layers - 1; layer++) {
            for (int l_plus_1_idx = 0; l_plus_1_idx < net->layer_sizes[layer + 1]; l_plus_1_idx++) {
                net->biases[layer][l_plus_1_idx] -= lrate * (bias_grads[layer][l_plus_1_idx]);
                for (int l_idx = 0; l_idx < net->layer_sizes[layer]; l_idx++) {
                    int widx = (l_plus_1_idx * net->layer_sizes[layer]) + l_idx;
                    net->weights[layer][widx] -= lrate * (weight_grads[layer][widx]);
                }
            }
        }

        // print stats after every 10000 samples processed
        if (((prev_print + 10000) <= i) || (i >= n_inputs)) {
            prev_print = i;
            int num_good = 0;
            for (int j = 0; j < n_inputs; j++) {
                double* output;
                stoopidnet_evaluate(net, inputs[j], &output);

                if(maxidx(output, 10) == (maxidx(outputs[j], 10))) {
                    num_good++;
                }
            }
            printf("%i examples trained. %i / %i accuracy\n", i, num_good, n_inputs);
        }
    }
}


////////////////////////////////////////////////////////////////
// static function impl
////////////////////////////////////////////////////////////////
static double sigmoid(const double z)
{
    return 1. / (1. + exp(-z));
}

static double sigmoid_prime(const double z)
{
    return (sigmoid(z) * (1 - sigmoid(z)));
}

static int* gen_shuffled_ints(int n)
{
    int* array = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        array[i] = i;
    }

    if (n > 1) {
        size_t i;
	for (i = 0; i < n - 1; i++) {
	  size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
	  int t = array[j];
	  array[j] = array[i];
	  array[i] = t;
	}
    }

    return array;
}

static void doubles_memset(double* d, int numel, double val)
{
    for (int i = 0; i < numel; i++) {
        d[i] = val;
    }
}

static stoopidnet_forward_prop_results_t* stoopidnet_forward_prop(stoopidnet_t* net,
                                                                  double *input)
{
    stoopidnet_forward_prop_results_t* results =
        calloc(1, sizeof(stoopidnet_forward_prop_results_t));

    results->a = calloc(net->num_layers, sizeof(double*));
    results->z = calloc(net->num_layers, sizeof(double*));

    results->a[0] = malloc(net->layer_sizes[0] * sizeof(double));

    memcpy(results->a[0], input, sizeof(double) * net->layer_sizes[0]);
    double *activiation_next = NULL;

    for (int l = 1; l < net->num_layers; l++) {
        results->z[l] = calloc(net->layer_sizes[l], sizeof(double));
        results->a[l] = calloc(net->layer_sizes[l], sizeof(double));

        // calculate next layer into activiation_next
        for (int j = 0; j < net->layer_sizes[l]; j++) {
            double accum = 0.;
            for (int k = 0; k < net->layer_sizes[l - 1]; k++) {
                int idx = (j * net->layer_sizes[l - 1]) + k;
                accum += (net->weights[l - 1])[idx] * results->a[l - 1][k];
            }
            results->z[l][j] = accum + (net->biases[l - 1][j]);
            results->a[l][j] = sigmoid(results->z[l][j]);
        }
    }

    return results;
}
