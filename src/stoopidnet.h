#ifndef STOOPIDNET_H
#define STOOPIDNET_H

#include <stdint.h>

typedef struct stoopidnet stoopidnet_t;

typedef struct stoopidnet_training_parameters
{
    double learn_rate;
    uint32_t batch_size;
} stoopidnet_training_parameters_t;

/**
 * Creates a brand new stoopidnet_t with no layers except the input layer
 */
stoopidnet_t* stoopidnet_create(uint32_t num_input_nodes);

/**
 * Destroys the given stoopidnet_t.
 */
void stoopidnet_destroy(stoopidnet_t* net);


/**
 * Loads a stoopdinet into a flat uint8_t array so that it can be written to a file.
 *
 * Returns the number of bytes in the returned array.
 */
uint32_t stoopidnet_serialize(stoopidnet_t* net, uint8_t** target);

/**
 * Loads a stoopidnet that was serialized.
 */
stoopidnet_t* stoopidnet_deserialize(uint8_t *data, uint32_t datalen);


void stoopidnet_add_fc_layer(stoopidnet_t* net, uint32_t num_nodes);
void stoopidnet_add_fc_layer_with_starting_weights(stoopidnet_t* net,
                                                   uint32_t num_nodes,
                                                   double* weights);

uint32_t stoopidnet_get_num_layers(stoopidnet_t* net);
uint32_t stoopidnet_get_num_nodes_in_layer(stoopidnet_t* net, uint32_t layer_idx);
void stoopidnet_set_layer_weights(stoopidnet_t* net, uint32_t layer_idx, double* weights);

void stoopidnet_evaluate(stoopidnet_t *net, double *input, double **output);
void stoopidnet_train(stoopidnet_t* net,
                      const stoopidnet_training_parameters_t* params,
                      uint32_t n_inputs,
                      double** inputs,
                      double** expected_outputs);

#endif
