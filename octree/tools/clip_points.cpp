#include <iostream>

#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_float(bbmin, kOptional, -1.0, "The bottom left corner of the bounding box");
DEFINE_float(bbmax, kOptional, 1.0, "The top right corner of the bounding box");
DEFINE_bool(verbose, kOptional, true, "Output logs");

using std::cout;

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: clip_points.exe");
    return 0;
  }

  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  float bbmin[3] = {FLAGS_bbmin, FLAGS_bbmin, FLAGS_bbmin };
  float bbmax[3] = {FLAGS_bbmax, FLAGS_bbmax, FLAGS_bbmax };

  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  for (int i = 0; i < all_files.size(); i++) {
    Points pts;
    pts.read_points(all_files[i]);
    pts.clip(bbmin, bbmax);

    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    pts.write_points(output_path + filename + ".clip.points");
  }

  return 0;
}