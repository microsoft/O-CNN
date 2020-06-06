#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <fstream>
#include <ctime>

#include "octree.h"
#include "filenames.h"
#include "cmd_flags.h"
#include "math_functions.h"
#include "transform_octree.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_string(axis, kOptional, "y", "The upright axis of the input model");
DEFINE_bool(verbose, kOptional, true, "Output logs");

class RandAxis {
 public:
  RandAxis(float upright[3])
    : generator_(static_cast<unsigned int>(time(nullptr))),
      distribution_(-1.0, 1.0) {
    for (int i = 0; i < 3; ++i) { upright_dir_[i] = upright[i]; }
  }

  void operator()(float* axis) {
    float dot = 1.0f;
    while (dot > 0) {
      for (int i = 0; i < 3; ++i) { axis[i] = distribution_(generator_); }
      float len = norm2(axis, 3);
      if (len < 1.0e-6) continue; // ignore zero vector
      for (int i = 0; i < 3; ++i) { axis[i] /= len; }
      dot = dot_prod(axis, upright_dir_);
    }
  }

 private:
  float upright_dir_[3];
  std::default_random_engine generator_;
  std::uniform_real_distribution<float> distribution_;
};


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree2mesh.exe");
    return 0;
  }

  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  float upright[] = { 0.0f, 0.0f, 0.0f };
  if (FLAGS_axis == "x") upright[0] = 1.0f;
  else if (FLAGS_axis == "y") upright[1] = 1.0f;
  else upright[2] = 1.0f;
  RandAxis rand_axis(upright);

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

    // dropout
    ScanOctree zbuffer;
    vector<char> octree_out;
    vector<float> axis = { 0, 0, 1, 0, 0, 1 };
    rand_axis(axis.data());
    rand_axis(axis.data() + 3);
    zbuffer.scan(octree_out, octree, axis);

    // save octree
    string filename_output = output_path + filename + ".zbuffer.octree";
    std::ofstream outfile(filename_output, std::ios::binary);
    outfile.write(octree_out.data(), octree_out.size());
    outfile.close();
  }

  return 0;
}
