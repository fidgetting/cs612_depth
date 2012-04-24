
#ifndef CONVOLVE_H
#define CONVOLVE_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "image.h"

void convolve_even(image<float> *src, image<float> *dst, std::vector<float> &mask);
void convolve_odd (image<float> *src, image<float> *dst, std::vector<float> &mask);

#endif
