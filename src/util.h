#ifndef _UTIL_H_
#define _UTIL_H_

#include <ctime>
#include <cstdlib>


// helpers
inline int myrandom (int i) {
  static bool initialized = false;
  if (!initialized) {
    std::srand(unsigned(std::time(0)));
    initialized = true;
  }  
  return std::rand()%i;
}

inline void randomSample(int n, int k, std::vector<int>& sel) {
  if (k > n) {
    k = n;
  }

  sel.resize(n);
  for (int i = 0; i < n; i++) {
    sel[i] = i;
  }

  std::random_shuffle (sel.begin(), sel.end(), myrandom);
  sel.resize(k);
}


#endif
