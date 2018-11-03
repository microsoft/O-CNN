#ifndef _OCTREE_UTIL_
#define _OCTREE_UTIL_

#include <vector>
#include <string>

using std::vector;
using std::string;

void mkdir(const string& dir);

void bounding_sphere(float& radius, float* center, const float* pt, const int npt);
void bouding_box(float* bbmin, float* bbmax, const float* pt, const int npt);

void rotation_matrix(float* rot, const float angle, const float* axis);
void matrix_prod(float* C, const float* A, const float* B,
    const int M, const int N, const int K);

void inverse_transpose_3x3(float* const out, const float* const mat);
bool almost_equal_3x3(const float* const mat1, const float* const mat2);
void normalize_nx3(float* const pts, int npt);

void get_all_filenames(vector<string>& all_filenames, const string& filename);

bool write_obj(const string& filename, const vector<float>& V, const vector<int>& F);
void write_ply(const string& filename, const vector<float>& V, const vector<int>& F);

string extract_path(string str);
string extract_filename(string str);

#endif // _OCTREE_UTIL_
