#ifndef _OCTREE_MATH_FUNCTIONS_
#define _OCTREE_MATH_FUNCTIONS_

#include <vector>

using std::vector;

const float ESP = 1.0e-30f;

void bounding_sphere(float& radius, float* center, const float* pt, const int npt);
void bouding_box(float* bbmin, float* bbmax, const float* pt, const int npt);

void rotation_matrix(float* rot, const float angle, const float* axis);
// the rotation matrix that rotates axis0 to axis1 with minimal angle
void rotation_matrix(float* rot, const float* axis0, const float* axis1);

void cross_prod(float* c, const float* a, const float* b);
float dot_prod(const float* a, const float* b);
void matrix_prod(float* C, const float* A, const float* B, const int M,
    const int N, const int K);

// give the z axis, output the x and y axis
void axes(float* x, float* y, const float* z);

void inverse_transpose_3x3(float* const out, const float* const mat);
bool almost_equal_3x3(const float* const mat1, const float* const mat2);
void normalize_nx3(float* const pts, int npt);

float norm2(const vector<float>& vec);
float norm2(const float* vec, int n);

template<typename Dtype>
Dtype clamp(Dtype val, Dtype val_min, Dtype val_max);


#endif // _OCTREE_MATH_FUNCTIONS_
