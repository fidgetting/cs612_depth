/*
 * filter.h
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <vector>
#include <cmath>
#include "image.h"
#include "misc.h"
#include "convolve.h"
#include "imconv.h"


#define MAKE_FILTER_SIG(name)                                   \
  std::vector<float> make_ ## name (float sigma)

#define MAKE_FILTER(name, fun)                                  \
  std::vector<float> make_ ## name (float sigma) {              \
    sigma = std::max(sigma, 0.01F);                             \
    int len = (int)ceil(sigma * WIDTH) + 1;                     \
    std::vector<float> mask(len);                               \
    for (int i = 0; i < len; i++) {                             \
      mask[i] = fun;                                            \
    }                                                           \
    return mask;                                                \
  }

MAKE_FILTER_SIG(fgauss);

void normalize(std::vector<float> &mask);
image<float> *smooth(image<float> *src, float sigma);
image<float> *laplacian(image<float> *src);

#endif /* FILTER_H_ */
