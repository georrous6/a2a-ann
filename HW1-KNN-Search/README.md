## To compile the target knnsearch_exact.c

gcc -g knnsearch_exact.c src/*.c -o build/knnsearch_exact -I/opt/OpenBLAS/include -I./include -L/opt/OpenBLAS/lib -I/usr/local/include -L/usr/local/lib -lmatio -lopenblas -lm -lpthread

## To run the target knnsearch_exact with valgrind

valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all --log-file=memory_usage.log ./knnsearch_exact ../test/validity_tests/test51.mat C Q 10

