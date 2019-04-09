#include <stdio.h>
int main() {
  char         *p_char;
  int          *p_int;
  unsigned int *p_uint; 
  short        *p_short;
  long         *p_long;
  long long    *p_long_long;
  float        *p_float;
  double       *p_double;
  void         *p_void;

  printf("sizeof(p_char):%lu\n", sizeof(p_char));
  printf("sizeof(p_int):%lu\n", sizeof(p_int));
  printf("sizeof(p_uint):%lu\n", sizeof(p_uint));
  printf("sizeof(p_long):%lu\n", sizeof(p_long));
  printf("sizeof(p_long_long):%lu\n", sizeof(p_long_long));
  printf("sizeof(p_float):%lu\n", sizeof(p_float));
  printf("sizeof(p_double):%lu\n", sizeof(p_double));
  printf("sizeof(p_void):%lu\n", sizeof(p_void));

  printf("------------------\n");

  printf("sizeof(float):%lu\n", sizeof(float));
  printf("sizeof(double):%lu\n", sizeof(double));
  printf("sizeof(int):%lu\n", sizeof(int));
  printf("sizeof(long):%lu\n", sizeof(long));
  printf("sizeof(char):%lu\n", sizeof(char));
  printf("sizeof(_Bool):%lu\n", sizeof(_Bool));

  return 0;
}
