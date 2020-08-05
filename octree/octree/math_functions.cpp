#include "math_functions.h"
#include <cfloat>
#include <cmath>

#ifdef USE_MINIBALL
#include <Miniball.hpp>

void bounding_sphere(float& radius, float* center, const float* pt, const int npt) {
  const int dim = 3;
  radius = 1.0e-10f; // !!! avoid zero radius
  center[0] = center[1] = center[2] = 0;
  if (npt < 1) return;

  // mini-ball
  const float** ap = new const float*[npt];
  for (int i = 0; i < npt; ++i) { ap[i] = pt + dim * i; }
  typedef const float** PointIterator;
  typedef const float* CoordIterator;
  Miniball::Miniball<Miniball::CoordAccessor<PointIterator, CoordIterator> >
  miniball(dim, ap, ap + npt);

  // get result
  if (miniball.is_valid()) {
    const float* cnt = miniball.center();
    for (int i = 0; i < dim; ++i) {
      center[i] = cnt[i];
    }
    radius += sqrtf(miniball.squared_radius());
  } else {
    // the miniball might fail sometimes
    // if so, just calculate the bounding box
    float bbmin[3] = { 0.0f, 0.0f, 0.0f };
    float bbmax[3] = { 0.0f, 0.0f, 0.0f };
    bounding_box(bbmin, bbmax, pt, npt);

    for (int j = 0; j < dim; ++j) {
      center[j] = (bbmax[j] + bbmin[j]) / 2.0f;
      float width = (bbmax[j] - bbmin[j]) / 2.0f;
      radius += width * width;
    }

    radius = sqrtf(radius);
  }

  // release
  delete[] ap;
}

#else

void bounding_sphere(float& radius, float* center, const float* pt, const int npt) {
  float bb[3][2] = {
    { FLT_MAX, -FLT_MAX }, { FLT_MAX, -FLT_MAX }, { FLT_MAX, -FLT_MAX }
  };
  int id[6];
  for (int i = 0; i < 3 * npt; i += 3) {
    if (pt[i] < bb[0][0]) {
      id[0] = i; bb[0][0] = pt[i];
    }
    if (pt[i] > bb[0][1]) {
      id[1] = i; bb[0][1] = pt[i];
    }
    if (pt[i + 1] < bb[1][0]) {
      id[2] = i; bb[1][0] = pt[i + 1];
    }
    if (pt[i + 1] > bb[1][1]) {
      id[3] = i; bb[1][1] = pt[i + 1];
    }
    if (pt[i + 2] < bb[2][0]) {
      id[4] = i; bb[2][0] = pt[i + 2];
    }
    if (pt[i + 2] > bb[2][1]) {
      id[5] = i; bb[2][1] = pt[i + 2];
    }
  }

  radius = 0;
  int choose_id = -1;
  for (int i = 0; i < 3; i++) {
    float dx = pt[id[2 * i]] - pt[id[2 * i + 1]];
    float dy = pt[id[2 * i] + 1] - pt[id[2 * i + 1] + 1];
    float dz = pt[id[2 * i] + 2] - pt[id[2 * i + 1] + 2];
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > radius) {
      radius = r2; choose_id = 2 * i;
    }
  }
  center[0] = 0.5f * (pt[id[choose_id]] + pt[id[choose_id + 1]]);
  center[1] = 0.5f * (pt[id[choose_id] + 1] + pt[id[choose_id + 1] + 1]);
  center[2] = 0.5f * (pt[id[choose_id] + 2] + pt[id[choose_id + 1] + 2]);

  float radius2 = radius * 0.25f;
  radius = sqrtf(radius2);

  for (int i = 0; i < 3 * npt; i += 3) {
    float dx = pt[i] - center[0], dy = pt[i + 1] - center[1], dz = pt[i + 2] - center[2];
    float dis2 = dx * dx + dy * dy + dz * dz;
    if (dis2 > radius2) {
      float old_to_p = sqrt(dis2);
      radius = (radius + old_to_p) * 0.5f;
      radius2 = radius * radius;
      float old_to_new = old_to_p - radius;
      center[0] = (radius * center[0] + old_to_new * pt[i]) / old_to_p;
      center[1] = (radius * center[1] + old_to_new * pt[i + 1]) / old_to_p;
      center[2] = (radius * center[2] + old_to_new * pt[i + 2]) / old_to_p;
    }
  }
}

#endif


void bounding_box(float* bbmin, float* bbmax, const float* pt, const int npt) {
  const int dim = 3;
  if (npt < 1) return;
  for (int i = 0; i < 3; ++i) {
    bbmin[i] = bbmax[i] = pt[i];
  }

  for (int i = 1; i < npt; ++i) {
    int i3 = i * 3;
    for (int j = 0; j < dim; ++j) {
      float tmp = pt[i3 + j];
      if (tmp < bbmin[j]) bbmin[j] = tmp;
      if (tmp > bbmax[j]) bbmax[j] = tmp;
    }
  }
}

void rotation_matrix(float* rot, const float angle, const float* axis) {
  float cosa = cos(angle); // angle in radian
  float cosa1 = 1 - cosa;
  float sina = sin(angle);

  rot[0] = cosa + axis[0] * axis[0] * cosa1;
  rot[1] = axis[0] * axis[1] * cosa1 + axis[2] * sina;
  rot[2] = axis[0] * axis[2] * cosa1 - axis[1] * sina;

  rot[3] = axis[0] * axis[1] * cosa1 - axis[2] * sina;
  rot[4] = cosa + axis[1] * axis[1] * cosa1;
  rot[5] = axis[1] * axis[2] * cosa1 + axis[0] * sina;

  rot[6] = axis[0] * axis[2] * cosa1 + axis[1] * sina;
  rot[7] = axis[1] * axis[2] * cosa1 - axis[0] * sina;
  rot[8] = cosa + axis[2] * axis[2] * cosa1;
}

void rotation_matrix(float* rot, const float* angle) {
  float cosx = cos(angle[0]), sinx = sin(angle[0]);
  float cosy = cos(angle[1]), siny = sin(angle[1]);
  float cosz = cos(angle[2]), sinz = sin(angle[2]);
  const float rotx[9] = { 1.0f, 0, 0, 0, cosx, sinx, 0, -sinx, cosx };
  const float roty[9] = { cosy, 0, -siny, 0, 1.0f, 0, siny, 0, cosy };
  const float rotz[9] = { cosz, sinz, 0, -sinz, cosz, 0, 0, 0, 1.0f };
  float tmp[9];
  matrix_prod(tmp, rotx, roty, 3, 3, 3);
  matrix_prod(rot, tmp,  rotz, 3, 3, 3);
}


void rotation_matrix(float* rot, const float* axis0, const float* axis1) {
  float angle = dot_prod(axis0, axis1);
  if (angle < -1) angle = -1;
  if (angle > 1) angle = 1;
  angle = acos(angle);

  float axis[3];
  cross_prod(axis, axis0, axis1);
  float ilen = 1.0 / (norm2(axis, 3) + 1.0e-10);
  for (int i = 0; i < 3; ++i) { axis[i] *= ilen; }

  rotation_matrix(rot, angle, axis);
}

void cross_prod(float * c, const float * a, const float * b) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

float dot_prod(const float * a, const float * b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void matrix_prod(float* C, const float* A, const float* B,
    const int M, const int N, const int K) {
  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      C[n * M + m] = 0;
      for (int k = 0; k < K; ++k) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
      }
    }
  }
}

void axes(float * x, float * y, const float * z) {
  // Find the rotation matrix that rotate the vector (0, 0, 1) to z,
  // then the first and second colomns of the rotation matrix are the output
  float z0[3] = { 0, 0, 1 }, rot[9];
  rotation_matrix(rot, z0, z);

  for (int i = 0; i < 3; ++i) {
    x[i] = rot[i];
    y[i] = rot[3 + i];
  }
}

void inverse_transpose_3x3(float* const out, const float* const mat) {
  double a = mat[4] * mat[8] - mat[7] * mat[5];
  double b = mat[3] * mat[8] - mat[5] * mat[6];
  double c = mat[3] * mat[7] - mat[4] * mat[6];

  double det = mat[0] * a - mat[1] * b + mat[2] * c;
  double invdet = 1 / det;

  out[0] =  a * invdet;
  out[3] = -(mat[1] * mat[8] - mat[2] * mat[7]) * invdet;
  out[6] = (mat[1] * mat[5] - mat[2] * mat[4]) * invdet;
  out[1] = -b * invdet;
  out[4] = (mat[0] * mat[8] - mat[2] * mat[6]) * invdet;
  out[7] = -(mat[0] * mat[5] - mat[3] * mat[2]) * invdet;
  out[2] =  c * invdet;
  out[5] = -(mat[0] * mat[7] - mat[6] * mat[1]) * invdet;
  out[8] = (mat[0] * mat[4] - mat[3] * mat[1]) * invdet;
}

bool almost_equal_3x3(const float* const mat1, const float* const mat2) {
  for (int i = 0; i < 9; ++i) {
    float diff = fabsf(mat1[i] - mat2[i]);
    float a = fabsf(mat1[i]);
    float b = fabsf(mat2[i]);
    float largest = (b > a) ? b : a;
    if (diff > largest * FLT_EPSILON) {
      return false;
    }
  }
  return true;
}

void normalize_nx3(float* const pts, int npt) {
  const int dim = 3;
  for (int i = 0; i < npt; ++i) {
    int ix3 = i * dim;
    float inv_mag = 1.0f / (norm2(pts + ix3, dim) + 1.0e-15f);
    for (int m = 0; m < dim; ++m) {
      pts[ix3 + m] = pts[ix3 + m] * inv_mag;
    }
  }
}

float norm2(const vector<float>& vec) {
  return norm2(vec.data(), vec.size());
}

float norm2(const float* vec, int n) {
  float len = 0.0f;
  for (int i = 0; i < n; ++i) {
    len += vec[i] * vec[i];
  }
  return sqrtf(len);
}

template<typename Dtype>
Dtype clamp(Dtype val, Dtype val_min, Dtype val_max) {
  if (val < val_min) val = val_min;
  if (val > val_max) val = val_max;
  return val;
}
template int clamp<int>(int val, int val_min, int val_max);
template float clamp<float>(float val, float val_min, float val_max);
