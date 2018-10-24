#include "util.h"

#include <fstream>
#include <octree_config.h>

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/types.h>
#include <sys/stat.h>
#endif

void mkdir(const string& dir) {
#if defined _MSC_VER
  _mkdir(dir.c_str());
#elif defined __GNUC__
  mkdir(dir.c_str(), 0744);
#endif
}

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
    bouding_box(bbmin, bbmax, pt, npt);

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

#ifdef USE_WINDOWS_IO
#include <io.h>

void get_all_filenames(vector<string>& all_filenames, const string& filename_in) {
  all_filenames.clear();
  string file_path = extract_path(filename_in) + "/";
  string filename = file_path + "*" + filename_in.substr(filename_in.rfind('.'));
  
  _finddata_t c_file;
  intptr_t hFile = _findfirst(filename.c_str(), &c_file);
  do {
    if (hFile == -1) break;
    all_filenames.push_back(file_path + string(c_file.name));
  } while (_findnext(hFile, &c_file) == 0);
  _findclose(hFile);
}

#else

void get_all_filenames(vector<string>& all_filenames, const string& data_list) {
  all_filenames.clear();

  std::ifstream infile(data_list);
  if (!infile) return;

  string line;
  while (std::getline(infile, line)) {
    all_filenames.push_back(line);
  }
  infile.close();
}

#endif

void bouding_box(float* bbmin, float* bbmax, const float* pt, const int npt) {
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

bool write_obj(const string& filename, const vector<float>& V, const vector<int>& F) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) return false;

  int nv = V.size() / 3;
  int nf = F.size() / 3;
  if (V.size() % 3 != 0 || F.size() % 3 != 0) return false;
  const int len = 64;
  char* buffer = new char[(nv + nf) * len];

  // convert to string
  char* pV = buffer;
  #pragma omp parallel for
  for (int i = 0; i < nv; i++) {
    sprintf(pV + i * len, "v %.6g %.6g %.6g\n", V[3 * i], V[3 * i + 1], V[3 * i + 2]);
  }

  char* pF = buffer + nv * len;
  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    sprintf(pF + i * len, "f %d %d %d\n", F[3 * i] + 1, F[3 * i + 1] + 1, F[3 * i + 2] + 1);
  }

  // shrink
  int k = 0;
  for (int i = 0; i < nv; i++) {
    for (int j = len * i; j < len * (i + 1); j++) {
      if (pV[j] == 0) break;
      buffer[k++] = pV[j];
    }
  }
  for (int i = 0; i < nf; i++) {
    for (int j = len * i; j < len * (i + 1); j++) {
      if (pF[j] == 0) break;
      buffer[k++] = pF[j];
    }
  }

  // write into file
  outfile.write(buffer, k);

  // close file
  outfile.close();
  delete[] buffer;
  return true;
}

//void write_ply(const string& filename, const vector<float>& V, const vector<int>& F) {
//  // open ply
//  p_ply ply = ply_create(filename.c_str(), PLY_LITTLE_ENDIAN, nullptr, 0, nullptr);
//  if (!ply) throw std::runtime_error("Unable to write PLY file!");
//
//  //  add vertex
//  int nv = V.size() / 3;
//  ply_add_element(ply, "vertex", nv);
//  ply_add_scalar_property(ply, "x", PLY_FLOAT);
//  ply_add_scalar_property(ply, "y", PLY_FLOAT);
//  ply_add_scalar_property(ply, "z", PLY_FLOAT);
//
//  // add face
//  int nf = F.size() / 3;
//  ply_add_element(ply, "face", nf);
//  ply_add_list_property(ply, "vertex_indices", PLY_UINT8, PLY_INT);
//
//  // write header
//  ply_write_header(ply);
//
//  // write vertex
//  for (int i = 0; i < nv; i++) {
//    for (int j = 0; j < 3; ++j) {
//      ply_write(ply, V[i * 3 + j]);
//    }
//  }
//
//  // write face
//  for (int i = 0; i < nf; i++) {
//    ply_write(ply, 3);
//    for (int j = 0; j < 3; ++j) {
//      ply_write(ply, F[i * 3 + j]);
//    }
//  }
//
//  // close ply
//  ply_close(ply);
//}


string extract_path(string str) {
  std::replace(str.begin(), str.end(), '\\', '/');
  size_t pos = str.rfind('/');
  if (string::npos == pos) {
    return string(".");
  } else {
    return str.substr(0, pos);
  }
}

string extract_filename(string str) {
  std::replace(str.begin(), str.end(), '\\', '/');
  size_t pos = str.rfind('/') + 1;
  size_t len = str.rfind('.');
  if (string::npos != len) len -= pos;
  return str.substr(pos, len);
}