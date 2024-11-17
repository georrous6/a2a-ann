## To compile the target knnsearch_exact.c

gcc -fdiagnostics-color=always -g main.c src/*.c -o build/knnsearch -I/opt/OpenBLAS/include -Iinclude -I/usr/local/MATLAB/R2024b/extern/include -L/usr/local/MATLAB/R2024b/bin/glnxa64 -L/opt/OpenBLAS/lib -L/usr/local/MATLAB/R2024b/sys/os/glnxa64 -lstdc++ -lopenblas -lm -lpthread -lmat -lmx

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

### or

gcc -fdiagnostics-color=always -g validity_tests.c src/*.c -o build/validity_tests -I/opt/OpenBLAS/include -Iinclude -I/usr/local/MATLAB/R2024b/extern/include -L/usr/local/MATLAB/R2024b/bin/glnxa64 -L/opt/OpenBLAS/lib -L/usr/local/MATLAB/R2024b/sys/os/glnxa64 -lstdc++ -lopenblas -lm -lpthread -lmat -lmx
