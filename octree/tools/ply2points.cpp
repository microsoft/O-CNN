#include <iostream>
#include <string>
#include <vector>
#include <happly.h>

#include "filenames.h"
#include "points.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(const_normal, kOptional, "1", "Set constant normal if there is normal");
DEFINE_bool(verbose, kOptional, true, "Output logs");


bool read_ply(vector<float>& pts, vector<float>& normals, const string filename) {
  ifstream infile(filename, ios::binary);
  if (infile.fail()) {
    cout << "Can not open " << filename << endl;
    return false;
  }

  happly::PLYData plyIn(infile);

  pts = plyIn.getVertices();
  bool has_normal = plyIn.getElement("vertex").hasProperty("nx");
  if (has_normal) {
    normals = plyIn.getNormals();
  } else {
    normals.clear();
  }

  return true;
}

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: ply2points");
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
    bool succ = false;
    if (normal.empty()) {
      if (FLAGS_const_normal) {
        normal.assign(vtx.size(), sqrtf(3.0f) / 3.0f);
        succ = pts.set_points(vtx, normal);
      } else {
        vector<float> feature(vtx.size() / 3, 1.0f);
        succ = pts.set_points(vtx, normal, feature);
      }
    } else {
      succ = pts.set_points(vtx, normal);
    }

    if (!succ) {
      if (FLAGS_verbose) cout << "Failed: " << filename << std::endl;
      continue;
    }
    pts.write_points(filename);
  }

  return 0;
}
