/*
 * segment-header.h
 *
 *  Created on: Feb 14, 2012
 *      Author: norton
 */

#ifndef SEGMENT_HEADER_H_
#define SEGMENT_HEADER_H_

#include <algorithm>
#include <cmath>
#include "disjoint-set.h"

typedef struct {
  float w;
  int a, b;
} edge;

inline float diff(image<float> *r, image<float> *g, image<float> *b,
       int x1, int y1, int x2, int y2);

image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size, int *num_ccs);
universe *segment_graph(int num_vertices, int num_edges, edge *edges, float c);

#endif /* SEGMENT_HEADER_H_ */
