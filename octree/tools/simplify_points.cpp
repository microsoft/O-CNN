#include <fstream>
#include <iostream>

#include "cmd_flags.h"
#include "filenames.h"
#include "simplify_points.h"

using std::string;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_int(dim, kOptional, 256, "The maximum resolution");
DEFINE_float(offset, kOptional, 0.55f, "The offset value for handing thin shapes");
DEFINE_bool(avg_points, kOptional, true, "Average points as output");
DEFINE_bool(verbose, kOptional, true, "Output logs");


// The points must have normals! Can not deal with labela and feature! (TODO)

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: simplify_points");
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

  //#pragma omp parallel for
  SimplifyPoints simplify_pts(FLAGS_dim, FLAGS_avg_points, FLAGS_offset);
  for (int i = 0; i < all_files.size(); i++) {
    string error_msg = simplify_pts.set_point_cloud(all_files[i]);
    if (!error_msg.empty()) {
      std::cout << error_msg << std::endl;
      continue;
    }

    simplify_pts.simplify();
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) {
      std::cout << "Processing: " + filename + "\n";
    }
    simplify_pts.write_point_cloud(output_path + filename + ".smp.points");
  }

  std::cout << "Done: " << FLAGS_filenames << std::endl;
  return 0;
}