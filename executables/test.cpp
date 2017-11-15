#include "util.h"
#include <gflags/gflags.h>


DEFINE_int32 (t, 0, "test_function");


int main (int argc, char** argv) {
  float vv[] = {6, 5, 4, 1, 2, 3, 11};

  std::vector<float> v (vv, vv + sizeof(vv) / sizeof(vv[0]) );

  std::vector<size_t> idx = util::argsort(v);
  
  for (size_t i = 0; i < idx.size(); i++) {
    printf("%f\n", v[idx[i]]);
  }

}