#include <iostream>
#include <fstream>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"
#include "math_functions.h"
#include "transform_points.h"

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_float(scale, kOptional, 1.0, "The scale factor");
DEFINE_float(trans, kOptional, 0.0, "The translation factor");
DEFINE_float(offset, kOptional, 0.0, "Offset the points along its normal");
DEFINE_float(ratio, kOptional, 0.5, "The dropout ratio");
DEFINE_float(dim, kOptional, 0, "The resolution for dropout");
DEFINE_float(std_nm, kOptional, 0.0, "The std of normal noise ");
DEFINE_float(std_pt, kOptional, 0.0, "The std of posistion noise");
DEFINE_string(mat, kOptional, "", "A txt file which contains a matrix");
DEFINE_bool(verbose, kOptional, true, "Output logs");

using std::cout;
using std::vector;

vector<float> load_matrix(const string& filename) {
  vector<float> mat;
  std::ifstream infile(filename);
  if (!infile) { return mat; }
  while (infile) {
    float m = 0;
    infile >> m;
    mat.push_back(m);
  }
  return mat;
}


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: transform_points");
    return 0;
  }

  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  for (int i = 0; i < all_files.size(); i++) {
    Points pts;
    pts.read_points(all_files[i]);

    if (FLAGS_trans != 0.0) {
      float trans[3] = { FLAGS_trans, FLAGS_trans, FLAGS_trans };
      pts.translate(trans);
    }

    if (FLAGS_scale != 1.0) {
      pts.uniform_scale(FLAGS_scale);
    }

    if (FLAGS_offset != 0) {
      pts.displace(FLAGS_offset);
    }

    if (FLAGS_mat.empty() == false) {
      vector<float> mat = load_matrix(FLAGS_mat);
      if (mat.size() > 9) {
        pts.transform(mat.data());
      }
    }

    if (FLAGS_dim > 0) {
      float radius, center[3];
      bounding_sphere(radius, center, pts.points(), pts.info().pt_num());
      float bbmin[3] = { center[0] - radius, center[1] - radius, center[2] - radius };
      float bbmax[3] = { center[0] + radius, center[1] + radius, center[2] + radius };
      DropPoints drop_points(FLAGS_dim, FLAGS_ratio, bbmin, bbmax);
      drop_points.dropout(pts);
    }

    if (FLAGS_std_pt > 1.0e-5f || FLAGS_std_nm > 1.0e-5f) {
      pts.add_noise(FLAGS_std_pt, FLAGS_std_nm);
    }

    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    pts.write_points(output_path + filename + ".trans.points");
  }

  return 0;
}