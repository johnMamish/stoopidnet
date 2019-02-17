#include "stoopidnet.h"

#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <stoopidnet input file OR \"null\"> <stoopidnet output file> "
               "<mnist data> <mnist labels>\n", argv[0]);
        return -1;
    }

    // load files
    stoopidnet_t* net;
    if (strcmp(argv[1], "null")) {
        net = stoopidnet_create(784);
        stoopidnet_add_fc_layer(net, 20);
        stoopidnet_add_fc_layer(net, 10);
    } else {
        net = stoopidnet_load_from_file(argv[1]);
    }

    load_label_file()
}
