#include <stdio.h>
#include <pam.h>

#include "stoopidnet.h"
#include "mnist_loader.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <mnist images file> <mnist labels file>\n", argv[0]);
        return -1;
    }

    uint8_t *labels = NULL;
    uint8_t **pics  = NULL;

    int nlabels = load_label_file(argv[2], &labels);
    for (int i = 0; i < nlabels; i++) {
        printf("%i ", labels[i]);
    }

    return 0;
}
