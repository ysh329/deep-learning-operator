#include "matrix.h"

int main(int argc, char *args[]) {
    const int rows = 5;
    const int cols = 3;
    const int a_len = rows * cols;
    const int range_max_val = 2;
    const int range_min_val = 1;
    DTYPE *a = calloc_matrix(a_len);
    rand_matrix(a, a_len, range_min_val, range_max_val);
    print_matrix(a, rows, cols);
    free_matrix(a);
    return 0;
}
