src = $(wildcard *.c)
obj = $(src:.c=.o)

CC = gcc
CFLAGS = -g -std=c99

mnist-shenanigans: $(obj)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) mnist-shenanigans