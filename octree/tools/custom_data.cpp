#include <iostream>
#include <fstream>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"


DEFINE_string(output_path, kOptional, ".", "The output path");

using std::cout;
using std::vector;


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: custom_points");
    return 0;
  }

  Points point_cloud;
  vector<float> points, normals, features, labels;
  // Set your data in points, normals, features, and labels.
  // The points must not be empty, the labels may be empty,
  // the normals & features must not be empty at the same time.
  //   points: 3 channels, x_1, y_1, z_1, ..., x_n, y_n, z_n
  //   normals: 3 channels, nx_1, ny_1, nz_1, ..., nx_n, ny_n, nz_n
  //   features (such as RGB color): k channels, r_1, g_1, b_1, ..., r_n, g_n, b_n
  //   labels: 1 channels, per-points labels

  // For example:
  // The following array contains two points: (1.0, 2.0, 3.0) and (4.0, 5.0, 6.0)
  float pts[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
  points.assign(pts, pts + 6);
  // Their normals are (1.0, 0.0, 0.0) and (0.57735, 0.57735, 0.57735)
  float ns[] = { 1.0, 0.0, 0.0, 0.57735, 0.57735, 0.57735 };
  normals.assign(ns, ns + 6);
  // They may also have 4 channel colors (0.5, 0.5, 0.5, 1.0) and (1.0, 0, 0, 0, 0.8)
  float rgba[] = { 0.5, 0.5, 0.5, 1.0, 1.0, 0, 0, 0, 0.8 };
  features.assign(rgba, rgba + 8);
  // Their labels are 0 and 1 respectively
  float lb[] = { 0.0, 1.0 };
  labels.assign(lb, lb + 2);

  point_cloud.set_points(points, normals, features, labels);
  point_cloud.write_points(FLAGS_output_path +  "\\my_points.points");

  return 0;
}