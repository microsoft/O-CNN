#include <iostream>
#include <string>
#include <vector>

#include "math_functions.h"
#include "filenames.h"
#include "points.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(type, kOptional, "box", "Choose from box and sphere");

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: bbox.exe");
    return 0;
  }

  string file_path = FLAGS_filenames;
  vector<string> all_files;
  get_all_filenames(all_files, file_path);

  for (int i = 0; i < all_files.size(); i++) {
    string filename = extract_filename(all_files[i]);
    cout << filename << ", ";

    Points pts;
    pts.read_points(all_files[i]);
    int npt = pts.info().pt_num();
    const float* pt = pts.ptr(PointsInfo::kPoint);

    if (FLAGS_type == "box") {
      float bbmin[3], bbmax[3];
      bounding_box(bbmin, bbmax, pt, npt);
      cout << bbmin[0] << ", " << bbmin[1] << ", " << bbmin[2] << ", "
           << bbmax[0] << ", " << bbmax[1] << ", " << bbmax[2] << endl;
    } else {
      float center[3], radius;
      bounding_sphere(radius, center, pt, npt);
      cout << center[0] << ", " << center[1] << ", " << center[2] << ", "
           << radius << endl;
    }
  }

  return 0;
}
