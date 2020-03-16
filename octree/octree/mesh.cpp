#include "mesh.h"

#include <cfloat>
#include <cmath>
#include <fstream>
#include <cstring>

#include "math_functions.h"
#include "filenames.h"

bool read_mesh(const string& filename, vector<float>& V, vector<int>& F) {
  bool succ = false;
  string suffix = extract_suffix(filename);
  if (suffix == "obj") {
    succ = read_obj(filename, V, F);
  } else if (suffix == "off") {
    succ = read_off(filename, V, F);
  } else if (suffix == "ply") {
    succ = read_ply(filename, V, F);
  } else {
    //cout << "Error : Unsupported file formate!" << endl;
  }
  return succ;
}


bool write_mesh(const string& filename, const vector<float>& V, const vector<int>& F) {
  bool succ = false;
  string suffix = extract_suffix(filename);
  if (suffix == "obj") {
    succ = write_obj(filename, V, F);
  } else if (suffix == "off") {
    //succ = write_off(filename, V, F); //todo
  } else if (suffix == "ply") {
    succ = write_ply(filename, V, F);
  } else {
    //cout << "Error : Unsupported file formate!" << endl;
  }
  return succ;
}


bool read_obj(const string& filename, vector<float>& V, vector<int>& F) {
  std::ifstream infile(filename, std::ifstream::binary);
  if (!infile) {
    //std::cout << "Open OBJ file error!" << std::endl;
    return false;
  }

  // get length of file
  infile.seekg(0, infile.end);
  int len = infile.tellg();
  infile.seekg(0, infile.beg);

  // load the file into memory
  char* buffer = new char[len + 1];
  infile.read(buffer, len);
  buffer[len] = 0;
  infile.close();

  // parse buffer data
  vector<char*> pVline, pFline;
  char* pch = strtok(buffer, "\n");
  while (pch != nullptr) {
    if (pch[0] == 'v' && pch[1] == ' ') {
      pVline.push_back(pch + 2);
    } else if (pch[0] == 'f' && pch[1] == ' ') {
      pFline.push_back(pch + 2);
    }

    pch = strtok(nullptr, "\n");
  }

  // load V
  V.resize(3 * pVline.size());
  //#pragma omp parallel for
  for (int i = 0; i < pVline.size(); i++) {
    //!!! strtok() is not thread safe in some platforms
    char* p = strtok(pVline[i], " ");
    for (int j = 0; j < 3; j++) {
      V[3 * i + j] = atof(p);
      p = strtok(nullptr, " ");
    }
  }

  // load F
  F.resize(3 * pFline.size());
  //#pragma omp parallel for
  for (int i = 0; i < pFline.size(); i++) {
    char* p = strtok(pFline[i], " ");
    for (int j = 0; j < 3; j++) {
      F[3 * i + j] = atoi(p) - 1;
      p = strtok(nullptr, " ");
    }
  }

  // release
  delete[] buffer;
  return true;
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
    int ix3 = i * 3;
    sprintf(pV + i * len, "v %.6g %.6g %.6g\n", V[ix3], V[ix3 + 1], V[ix3 + 2]);
  }

  char* pF = buffer + nv * len;
  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    sprintf(pF + i * len, "f %d %d %d\n", F[ix3] + 1, F[ix3 + 1] + 1, F[ix3 + 2] + 1);
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


bool read_off(const string& filename, vector<float>& V, vector<int>& F) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    //std::cout << "Open " + filename + " error!" << std::endl;
    return false;
  }

  // face/vertex number
  int nv, nf, ne;
  char head[256];
  infile >> head; // eat head
  if (head[0] == 'O' && head[1] == 'F' && head[2] == 'F') {
    if (head[3] == 0) {
      infile >> nv >> nf >> ne;
    } else if (head[3] == ' ') {
      vector<char*> tokens;
      char* pch = strtok(head + 3, " ");
      while (pch != nullptr) {
        tokens.push_back(pch);
        pch = strtok(nullptr, " ");
      }
      if (tokens.size() != 3) {
        //std::cout << filename + " is not an OFF file!" << std::endl;
        return false;
      }
      nv = atoi(tokens[0]);
      nf = atoi(tokens[1]);
      ne = atoi(tokens[2]);
    } else {
      //std::cout << filename + " is not an OFF file!" << std::endl;
      return false;
    }
  } else {
    //std::cout << filename + " is not an OFF file!" << std::endl;
    return false;
  }

  // get length of file
  int p1 = infile.tellg();
  infile.seekg(0, infile.end);
  int p2 = infile.tellg();
  infile.seekg(p1, infile.beg);
  int len = p2 - p1;

  // load the file into memory
  char* buffer = new char[len + 1];
  infile.read(buffer, len);
  buffer[len] = 0;

  // close file
  infile.close();

  // parse buffer data
  std::vector<char*> pV;
  pV.reserve(3 * nv);
  char* pch = strtok(buffer, " \r\n");
  pV.push_back(pch);
  for (int i = 1; i < 3 * nv; i++) {
    pch = strtok(nullptr, " \r\n");
    pV.push_back(pch);
  }
  std::vector<char*> pF;
  pF.reserve(3 * nf);
  for (int i = 0; i < nf; i++) {
    // eat the first data
    pch = strtok(nullptr, " \r\n");
    for (int j = 0; j < 3; j++) {
      pch = strtok(nullptr, " \r\n");
      pF.push_back(pch);
    }
  }

  // load vertex
  V.resize(3 * nv);
  float* p = V.data();
  #pragma omp parallel for
  for (int i = 0; i < 3 * nv; i++) {
    *(p + i) = atof(pV[i]);
  }

  // load face
  F.resize(3 * nf);
  int* q = F.data();
  #pragma omp parallel for
  for (int i = 0; i < 3 * nf; i++) {
    *(q + i) = atoi(pF[i]);
  }

  //release
  delete[] buffer;
  return true;
}


#ifdef USE_RPLY
#include <rply.h>

bool read_ply(const string& filename, vector<float>& V, vector<int>& F) {
  // open ply file
  p_ply ply = ply_open(filename.c_str(), nullptr, 0, nullptr);
  if (!ply) {
    // std::cout << "Open PLY file error!" << std::endl;
    return false;
  }

  // read file header
  if (!ply_read_header(ply)) {
    ply_close(ply);
    // std::cout << "Open PLY header error!" << std::endl;
    return false;
  }

  // get vertex number and face number
  p_ply_element element = nullptr;
  uint32_t nv = 0, nf = 0;
  while ((element = ply_get_next_element(ply, element)) != nullptr) {
    const char *name;
    long nInstances;
    ply_get_element_info(element, &name, &nInstances);
    if (!strcmp(name, "vertex"))
      nv = (uint32_t)nInstances;
    else if (!strcmp(name, "face"))
      nf = (uint32_t)nInstances;
  }

  // init F&V
  F.resize(3 * nf);
  V.resize(3 * nv);

  // callback
  auto rply_vertex_cb = [](p_ply_argument argument) -> int {
    vector<float> *pV; long index, coord;
    ply_get_argument_user_data(argument, (void **)&pV, &coord);
    ply_get_argument_element(argument, nullptr, &index);
    (*pV)[3 * index + coord] = (float)ply_get_argument_value(argument);
    return 1;
  };

  auto rply_index_cb = [](p_ply_argument argument) -> int {
    vector<int> *pF;
    ply_get_argument_user_data(argument, (void **)&pF, nullptr);
    long length, value_index, index;
    ply_get_argument_property(argument, nullptr, &length, &value_index);
    //if (length != 3) throw std::runtime_error("Only triangle faces are supported!");
    ply_get_argument_element(argument, nullptr, &index);
    if (value_index >= 0)
      (*pF)[3 * index + value_index] = (int)ply_get_argument_value(argument);
    return 1;
  };

  // set vertex callback
  ply_set_read_cb(ply, "vertex", "x", rply_vertex_cb, &V, 0);
  ply_set_read_cb(ply, "vertex", "y", rply_vertex_cb, &V, 1);
  ply_set_read_cb(ply, "vertex", "z", rply_vertex_cb, &V, 2);

  // set face callback
  long nfr = ply_set_read_cb(ply, "face", "vertex_indices", rply_index_cb, &F, 0);
  if (nfr < 1) ply_set_read_cb(ply, "face", "vertex_index", rply_index_cb, &F, 0);

  //ply_read
  ply_read(ply);
  ply_close(ply);

  return true;
}

bool write_ply(const string& filename, const vector<float>& V, const vector<int>& F) {
  // open ply
  p_ply ply = ply_create(filename.c_str(), PLY_LITTLE_ENDIAN, nullptr, 0, nullptr);
  if (!ply) {
    // throw std::runtime_error("Unable to write PLY file!");
    return false;
  }

  //  add vertex
  int nv = V.size() / 3;
  ply_add_element(ply, "vertex", nv);
  ply_add_scalar_property(ply, "x", PLY_FLOAT);
  ply_add_scalar_property(ply, "y", PLY_FLOAT);
  ply_add_scalar_property(ply, "z", PLY_FLOAT);

  // add face
  int nf = F.size() / 3;
  ply_add_element(ply, "face", nf);
  ply_add_list_property(ply, "vertex_indices", PLY_UINT8, PLY_INT);

  // write header
  ply_write_header(ply);

  // write vertex
  for (int i = 0; i < nv; i++) {
    for (int j = 0; j < 3; ++j) {
      ply_write(ply, V[i * 3 + j]);
    }
  }

  // write face
  for (int i = 0; i < nf; i++) {
    ply_write(ply, 3);
    for (int j = 0; j < 3; ++j) {
      ply_write(ply, F[i * 3 + j]);
    }
  }

  // close ply
  ply_close(ply);
  return true;
}

#else
#include <iostream>
#include <happly.h>

bool read_ply(const string& filename, vector<float>& V, vector<int>& F) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "Error, cannot read ply files!" << std::endl;
    return false;
  }

  happly::PLYData plyIn(infile);
  V = plyIn.getVertices();
  F = plyIn.getTriFaces();

  return true;
}

bool write_ply(const string& filename, const vector<float>& V, const vector<int>& F) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    std::cerr << "Error, cannot read ply files!" << std::endl;
    return false;
  }

  happly::PLYData plyOut;
  plyOut.addVertices(V);
  plyOut.addTriFaces(F);

  plyOut.write(outfile);
  return false;
}

#endif


void compute_face_center(vector<float>& Fc, const vector<float>& V,
    const vector<int>& F) {
  int nf = F.size() / 3;
  Fc.assign(3 * nf, 0);
  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    for (int j = 0; j < 3; j++) {
      int fx3 = F[ix3 + j] * 3;
      for (int k = 0; k < 3; ++k) {
        Fc[ix3 + k] += V[fx3 + k];
      }
    }
    for (int k = 0; k < 3; ++k) {
      Fc[ix3 + k] /= 3.0f;
    }
  }
}

void compute_face_normal(vector<float>& face_normal, vector<float>& face_area,
    const vector<float>& V, const vector<int>& F) {
  int nf = F.size() / 3;
  face_normal.resize(3 * nf);
  face_area.resize(nf);
  const float* pt = V.data();
  const float EPS = 1.0e-10;
  float* normal = face_normal.data();

  #pragma omp parallel for
  for (int i = 0; i < nf; i++) {
    int ix3 = i * 3;
    const float* v0 = pt + F[ix3] * 3;
    const float* v1 = pt + F[ix3 + 1] * 3;
    const float* v2 = pt + F[ix3 + 2] * 3;

    float p01[3], p02[3];
    for (int j = 0; j < 3; ++j) {
      p01[j] = v1[j] - v0[j];
      p02[j] = v2[j] - v0[j];
    }

    float* normal_i = normal + ix3;
    cross_prod(normal_i, p01, p02);
    float len = norm2(normal_i, 3);
    if (len < EPS) len = EPS;
    for (int j = 0; j < 3; ++j) {
      normal_i[j] /= len;
    }

    face_area[i] = len * 0.5;
  }
}