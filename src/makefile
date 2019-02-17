src = $(wildcard *.c)
obj = $(src:.c=.o)

CC = gcc
CFLAGS = -g -std=c99 -lnetpbm -Werror -Wall -Wpedantic -lm -Wno-unused-variable

libs = mnist_loader.c stoopidnet.c

all: mnist-shenanigans mnist-peek stoopidnet-test-serdes

mnist-shenanigans: main.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

mnist-peek: mnist_peek.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-test-serdes: stoopidnet_test_serdes.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) mnist-shenanigans
