#include "stoopidnet.h"

#include <stdint.h>
#include <stdio.h>

int main(int argc, char** argv) {
    stoopidnet_t* net = stoopidnet_create(784);
    stoopidnet_add_fc_layer(net, 20);
    stoopidnet_add_fc_layer(net, 10);

    uint8_t* data;
    int len = stoopidnet_serialize(net, &data);
    stoopidnet_t* netnet = stoopidnet_deserialize(data, len);

    printf("len = %i; layers = %i %i\n", len, stoopidnet_get_num_layers(net), stoopidnet_get_num_layers(netnet));
    return 0;
}
