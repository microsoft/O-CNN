#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <rply.h>

#include "util.h"
#include "points.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(verbose, kOptional, true, "Output logs");

bool read_ply(vector<float>& pts, vector<float>& normals, const string filename) {
  // open ply file
  p_ply ply = ply_open(filename.c_str(), nullptr, 0, nullptr);
  if (!ply) {
    cout << "Open PLY file error!" << endl;
    return false;
  }

  // read file header
  if (!ply_read_header(ply)) {
    ply_close(ply);
    cout << "Open PLY header error!" << endl;
    return false;
  }

  // get vertex number and face number
  p_ply_element element = nullptr;
  int nv = 0, nf = 0;
  while ((element = ply_get_next_element(ply, element)) != nullptr) {
    const char *name;
    long nInstances;

    ply_get_element_info(element, &name, &nInstances);
    if (!strcmp(name, "vertex")) nv = nInstances;
    if (!strcmp(name, "face")) nf = nInstances;
  }

  // init
  pts.resize(3 * nv);
  normals.resize(3 * nv);

  // callback
  auto rply_vertex_cb = [](p_ply_argument argument) -> int {
    float *ptr; long index, coord;
    ply_get_argument_user_data(argument, (void **)(&ptr), &coord);
    ply_get_argument_element(argument, nullptr, &index);
    ptr[3 * index + coord] = (float)ply_get_argument_value(argument);
    return 1;
  };

  // set vertex callback
  ply_set_read_cb(ply, "vertex", "x", rply_vertex_cb, pts.data(), 0);
  ply_set_read_cb(ply, "vertex", "y", rply_vertex_cb, pts.data(), 1);
  ply_set_read_cb(ply, "vertex", "z", rply_vertex_cb, pts.data(), 2);
  ply_set_read_cb(ply, "vertex", "nx", rply_vertex_cb, normals.data(), 0);
  ply_set_read_cb(ply, "vertex", "ny", rply_vertex_cb, normals.data(), 1);
  ply_set_read_cb(ply, "vertex", "nz", rply_vertex_cb, normals.data(), 2);

  // read
  ply_read(ply);
  ply_close(ply);
  return true;
}


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: ply2points.exe");
    return 0;
  }

  // file path
  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);

  for (int i = 0; i < all_files.size(); i++) {
    vector<float> vtx, normal;
    read_ply(vtx, normal, all_files[i]);

    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    filename = output_path + filename + ".points";

    Points pts;
    bool succ = pts.set_points(vtx, normal);
    if (!succ) {
      if (FLAGS_verbose) cout << "Failed: " << filename << std::endl;
      continue;
    }
    pts.write_points(filename);
  }

  return 0;
}
