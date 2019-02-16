#ifndef STOOPIDNET_H
#define STOOPIDNET_H

#include <stdint.h>

typedef struct stoopidnet stoopidnet_t;

typedef struct stoopidnet_training_params
{
    double learn_rate;
} stoopidnet_training_params_t;

/**
 *
 */
stoopidnet_t* stoopidnet_create(uint32_t num_input_nodes);
void stoopidnet_destroy(stoopidnet_t* net);

void stoopidnet_add_fc_layer(stoopidnet_t* net, uint32_t num_nodes);
void stoopidnet_add_fc_layer_with_starting_weights(stoopidnet_t* net,
                                                   uint32_t num_nodes,
                                                   double* weights);

uint32_t stoopidnet_get_num_nodes_in_layer(stoopidnet_t* net, uint32_t layer_idx);
void stoopidnet_set_layer_weights(stoopidnet_t* net, uint32_t layer_idx, double* weights);

void stoopidnet_train(stoopidnet_t* net,
                      const stoopidnet_training_parameters_t* params,
                      double** inputs,
                      double** outputs);

#endif
