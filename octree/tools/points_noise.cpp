#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

#include "cmd_flags.h"
#include "points.h"
#include "filenames.h"


using std::vector;
using std::string;
using std::cout;
using std::endl;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(verbose, kOptional, true, "Output logs");

class GaussianRand {
 public:
  GaussianRand(float dev)
    : generator_(static_cast<unsigned int>(time(nullptr))),
      distribution_(0, dev) {}

  void operator()(float& rnd) {
    rnd = distribution_(generator_);
  }

  void operator()(float* rnd, int n) {
    for (int i = 0; i < n; ++i) {
      rnd[i] = distribution_(generator_);
    }
  }

 private:
  std::default_random_engine generator_;
  std::normal_distribution<float> distribution_;
};

class BernoulliRand {
 public:
  BernoulliRand(float p)
    : generator_(static_cast<unsigned int>(time(nullptr))),
      distribution_(p) {}

  void operator()(int rnd) {
    rnd = static_cast<int>(distribution_(generator_));
  }

  void operator()(int* rnd, int n) {
    for (int i = 0; i < n; ++i) {
      rnd[i] = static_cast<int>(distribution_(generator_));
    }
  }

 private:
  std::default_random_engine generator_;
  std::bernoulli_distribution distribution_;
};



void add_noise(Points& pts) {
  BernoulliRand bernoulli(0.1f);
  GaussianRand pt_noise(3.0f), normal_noise(0.1f);

  // add pt noise
  int npts = pts.info().pt_num();
  vector<float> noise(npts);
  pt_noise(noise.data(), npts);
  vector<int> mask(npts);
  bernoulli(mask.data(), npts);

  float* ptr_pts = pts.mutable_ptr(PointsInfo::kPoint);
  float* ptr_normal = pts.mutable_ptr(PointsInfo::kNormal);
  for (int i = 0; i < npts; ++i) {
    if (mask[i] == 0) continue;
    for (int c = 0; c < 3; ++c) {
      ptr_pts[i * 3 + c] += noise[i] * ptr_normal[i * 3 + c];
    }
  }

  // add normal noise
  noise.resize(3 * npts);
  normal_noise(noise.data(), npts * 3);
  for (int i = 0; i < npts; ++i) {
    int ix3 = i * 3;
    float len = 0;
    for (int c = 0; c < 3; ++c) {
      ptr_normal[ix3 + c] += noise[ix3 + c];
      len += ptr_normal[ix3 + c] * ptr_normal[ix3 + c];
    }

    len = sqrtf(len + 1.0e-10f);
    for (int c = 0; c < 3; ++c) {
      ptr_normal[ix3 + c] /= len;
    }
  }
}

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

  for (int i = 0; i < all_files.size(); i++) {
    Points pts;
    pts.read_points(all_files[i]);

    add_noise(pts);

    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    filename = output_path + filename + "_noise.points";
    pts.write_points(filename);
  }

  cout << "Done: " << FLAGS_filenames << endl;
  return 0;
}