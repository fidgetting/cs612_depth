/*
 * imconv.h
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

#ifndef IMCONV_H_
#define IMCONV_H_

#include <climits>
#include "image.h"
#include "imutil.h"
#include "misc.h"

image<uchar> *imageRGBtoGRAY(image<rgb> *input);
image<rgb> *imageGRAYtoRGB(image<uchar> *input);
image<float> *imageUCHARtoFLOAT(image<uchar> *input);
image<float> *imageINTtoFLOAT(image<int> *input);
image<uchar> *imageFLOATtoUCHAR(image<float> *input, float min, float max);
static image<uchar> *imageFLOATtoUCHAR(image<float> *input);
image<long> *imageUCHARtoLONG(image<uchar> *input);
image<uchar> *imageLONGtoUCHAR(image<long> *input, long min, long max);
image<uchar> *imageLONGtoUCHAR(image<long> *input);
image<uchar> *imageSHORTtoUCHAR(image<short> *input, short min, short max);
image<uchar> *imageSHORTtoUCHAR(image<short> *input);

#endif /* IMCONV_H_ */
