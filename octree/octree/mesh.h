#ifndef _OCTREE_MESH_
#define _OCTREE_MESH_

#include <vector>
#include <string>

using std::vector;
using std::string;

// I/O
bool read_mesh(const string& filename, vector<float>& V, vector<int>& F);
bool write_mesh(const string& filename, const vector<float>& V, const vector<int>& F);

bool read_obj(const string& filename, vector<float>& V, vector<int>& F);
bool write_obj(const string& filename, const vector<float>& V, const vector<int>& F);
bool read_off(const string& filename, vector<float>& V, vector<int>& F);
bool read_ply(const string& filename, vector<float>& V, vector<int>& F);
bool write_ply(const string& filename, const vector<float>& V, const vector<int>& F);

// mesh related calculation
void compute_face_center(vector<float>& Fc, const vector<float>& V, const vector<int>& F);
void compute_face_normal(vector<float>& face_normal, vector<float>& face_area,
    const vector<float>& V, const vector<int>& F);

#endif // _OCTREE_MESH_
