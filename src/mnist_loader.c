#include "mnist_loader.h"

#include <byteswap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int load_label_file(const char* filepath, uint8_t** target)
{
    // open file
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) {
        printf("failed to open file %s\n", filepath);
        return 0;
    }

    // read first 8 bytes of file
    const int headlen = 2;
    uint32_t head[headlen];
    size_t r = fread(head, sizeof(uint32_t), headlen, fp);
    for(int i = 0; i < headlen; i++) { head[i] = __bswap_32(head[i]); }
    if ((r != headlen) || (head[0] != 0x00000801)) {
        printf("file %s is not a valid MNIST label file because its magic number is %08x\n",
               filepath, head[0]);
        fclose(fp);
        return 0;
    }

    // allocate memory and read the rest.
    //printf("file %s has %i samples\n", filepath, head[1]);
    *target = malloc(head[1] * sizeof(uint8_t));
    r = fread(*target, sizeof(uint8_t), head[1], fp);
    if (r != head[1]) {
        printf("Didn't read expected number of bytes from file %s. Expected %i, only got %i.\n",
               filepath, head[1], (int)r);
        fclose(fp);
        free(*target);
        *target = NULL;
        return 0;
    }

    return head[1];
}

/**
 *
 */
int load_data_file(const char* filepath, uint32_t wh[2], uint8_t*** target)
{
    // open file
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL) {
        printf("failed to open file %s\n", filepath);
        return 0;
    }

    // read first 8 bytes of file
    const int headlen = 4;
    uint32_t head[headlen];
    size_t r = fread(head, sizeof(uint32_t), headlen, fp);
    for(int i = 0; i < headlen; i++) { head[i] = __bswap_32(head[i]); }
    if ((r != headlen) || (head[0] != 0x00000803)) {
        printf("file %s is not a valid MNIST data file because its magic number is %08x\n",
               filepath, head[0]);
        fclose(fp);
        return 0;
    }

    wh[0] = head[2];
    wh[1] = head[3];

    // allocate memory and read the rest.
    //printf("file %s has %i samples with WxH = %i,%i\n", filepath, head[1], head[2], head[3]);
    *target = calloc(head[1], sizeof(uint8_t*));
    for (int i = 0; i < head[1]; i++) {
        (*target)[i] = malloc(head[2] * head[3] * sizeof(uint8_t));
        r = fread((*target)[i], sizeof(uint8_t), head[2] * head[3], fp);
        if (r != (head[2] * head[3])) {
            goto cleanup;
        }
    }

    return head[1];

cleanup:
    fclose(fp);
    for (int i = 0; i < head[1]; i++) {
        free((*target)[i]);
    }
    free(*target);
    *target = NULL;
    return 0;
}

/**
 *
 */
int load_label_file_doubles(const char* filepath, double*** target)
{
    uint8_t* labels_u8;
    int numel = load_label_file(filepath, &labels_u8);

    if (numel == 0) {
        *target = NULL;
    } else {
        *target = malloc(numel * sizeof(double*));
        for (int i = 0; i < numel; i++) {
            (*target)[i] = malloc(10 * sizeof(double));
            for (int j = 0; j < 10; j++) {
                ((*target)[i])[j] = (j == labels_u8[i]) ? 1.0 : 0.0;
            }
        }
    }
    free(labels_u8);

    return numel;
}

int load_data_file_doubles(const char* filepath, double*** target)
{
    uint8_t** data_u8;
    uint32_t wh[2];
    int numel = load_data_file(filepath, wh, &data_u8);

    if (numel == 0) {
        *target = NULL;
    } else {
        *target = calloc(numel, sizeof(double*));
        for (int i = 0; i < numel; i++) {
            ((*target)[i]) = malloc(sizeof(double) * wh[0] * wh[1]);
            for (int j = 0; j < (wh[0] * wh[1]); j++) {
                ((*target)[i])[j] = (double)(data_u8[i])[j];
                ((*target)[i])[j] /= 255.0;
            }
            free(data_u8[i]);
        }
    }

    free(data_u8);
    return numel;
}
