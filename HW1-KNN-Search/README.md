## To compile the target knnsearch_exact.c

gcc -g main.c src/*.c -o build/knnsearch -I/opt/OpenBLAS/include -I./include -L/opt/OpenBLAS/lib -I/usr/local/include -L/usr/local/lib -lmatio -lopenblas -lm -lpthread

Open your shell's configuration file:
vim ~/.bashrc

and add the following lines:
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/MATLAB/R2024b/bin/glnxa64:$LD_LIBRARY_PATH


## To run the target knnsearch_exact with valgrind

cd build

valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all --log-file=memory_usage.log ./knnsearch ../test/validity_tests/test51.mat C Q 10

## To run the tests

cd test

chmod +x knn_tests.sh

./knn_tests path/to/executable
