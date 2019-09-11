#include <iostream>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_float(scale, kOptional, 1.0, "The scale factor");
DEFINE_float(trans, kOptional, 0.0, "The translation factor");
DEFINE_float(offset, kOptional, 0.0, "Offset the points along its normal");
DEFINE_bool(verbose, kOptional, true, "Output logs");

using std::cout;

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: transform_points.exe");
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

    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    pts.write_points(output_path + filename + ".points");
  }

  return 0;
}