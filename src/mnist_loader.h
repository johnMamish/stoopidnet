#ifndef MNIST_LOADER
#define MNIST_LOADER

/**
 * the functions in this file load the mnist handwriting sets that can be downloaded at
 * http://yann.lecun.com/exdb/mnist/
 */

#include <stdint.h>

/**
 * Loads labels (bytes 0 - 9) into *target.
 */
int load_label_file(const char* filepath, uint8_t** target);

/**
 *
 */
int load_data_file(const char* filepath, uint32_t wh[2], uint8_t*** target);

/**
 *
 */
int load_label_file_doubles(const char* filepath, double*** target);

int load_data_file_doubles(const char* filepath, double*** target);


#endif
