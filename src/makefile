src = $(wildcard *.c)
obj = $(src:.c=.o)

CC = gcc
CFLAGS = -g -std=c99 -lnetpbm  -Wall -Wpedantic -lm -Wno-unused-variable -Ofast
#CFLAGS += -fsanitize=address

libs = mnist_loader.c stoopidnet.c math_util.c

execs = mnist-shenanigans mnist-peek stoopidnet-test-serdes stoopidnet-train stoopidnet-run stoopidnet-run-pgm

all: mnist-shenanigans mnist-peek stoopidnet-test-serdes stoopidnet-train stoopidnet-run stoopidnet-run-pgm

mnist-shenanigans: main.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

mnist-peek: mnist_peek.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-test-serdes: stoopidnet_test_serdes.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-train: stoopidnet_train.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-run: stoopidnet_run.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-nand: stoopidnet_nand.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

stoopidnet-run-pgm: stoopidnet_run_pgm.c $(libs)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) -lnetpbm

.PHONY: clean
clean:
	rm -f $(obj) $(execs)
