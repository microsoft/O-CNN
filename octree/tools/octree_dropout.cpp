#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "filenames.h"
#include "points.h"
#include "octree.h"
#include "contour.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_int(depth_dropout, kOptional, 2, "The dropout depth");
DEFINE_float(dropout_ratio, kOptional, 0.5, "The dropout ratio");
DEFINE_bool(verbose, kOptional, true, "Output logs");


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree_dropout");
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

  char suffix[128];
  sprintf(suffix, ".drop_%d_%.2f.octree", FLAGS_depth_dropout, FLAGS_dropout_ratio);
  for (int i = 0; i < all_files.size(); i++) {
    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    //// load octree
    //Octree octree;
    //bool succ = octree.read_octree(all_files[i]);
    //if (!succ) {
    //  cout << "Can not load " << filename << std::endl;
    //  continue;
    //}
    //string msg;
    //succ = octree.info().check_format(msg);
    //if (!succ) {
    //  cout << filename << ": format error!\n" << msg << std::endl;
    //  continue;
    //}

    //// dropout
    //Octree octree_output;
    //octree.dropout(octree_output, FLAGS_depth_dropout, FLAGS_dropout_ratio);

    //// save points
    //string filename_output = output_path + filename + suffix;
    //octree_output.write_octree(filename_output);
  }

  return 0;
}
