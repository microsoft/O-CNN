#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "mesh.h"
#include "octree.h"
#include "contour.h"
#include "cmd_flags.h"
#include "filenames.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_int(depth_start, kOptional, 0, "The starting depth");
DEFINE_int(depth_end, kOptional, 10, "The ending depth");
DEFINE_bool(rescale, kOptional, true, "Scale the mesh according to the bbox");
DEFINE_bool(pu, kOptional, false, "Partition of Unity");
DEFINE_bool(verbose, kOptional, true, "Output logs");


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree2mesh.exe");
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
    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    // load octree
    Octree octree;
    bool succ = octree.read_octree(all_files[i]);
    if (!succ) {
      cout << "Can not load " << filename << std::endl;
      continue;
    }
    string msg;
    succ = octree.info().check_format(msg);
    if (!succ) {
      cout << filename << ": format error!\n" << msg << std::endl;
      continue;
    }

    // convert
    vector<float> V; vector<int> F;
    if (!FLAGS_pu) {
      octree.octree2mesh(V, F, FLAGS_depth_start, FLAGS_depth_end, FLAGS_rescale);
    } else {
      clock_t t = clock();
      Contour contour(&octree, FLAGS_rescale);
      contour.marching_cube(V, F);
      t = clock() - t;
      cout << "time : " << t << endl;
    }

    // save points
    filename = output_path + filename + ".obj";
    write_obj(filename, V, F);
  }

  return 0;
}
