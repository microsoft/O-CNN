#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cmd_flags.h"
#include "filenames.h"
#include "math_functions.h"
#include "points.h"

using cflags::Require;
using std::cout;
using std::endl;
using std::string;
using std::vector;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_float(ratio, kOptional, 0.1f, "The ratio of perturbed points");
DEFINE_float(dp, kOptional, 1.0f, "The deviation of point noise");
DEFINE_float(dn, kOptional, 0.1f, "The deviation of normal noise");
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


void add_noise(Points& pts, GaussianRand& pt_noise, GaussianRand& normal_noise,
    BernoulliRand& bernoulli) {
  // bounding sphere of the points
  float radius = 0, center[3] = { 0 };
  int npts = pts.info().pt_num();
  bounding_sphere(radius, center, pts.points(), npts);
  radius = radius / 100.0f;

  // add pt noise
  vector<float> noise(npts);
  pt_noise(noise.data(), npts);
  for (int i = 0; i < npts; ++i) {
    noise[i] *= radius;    // rescale the noise according to the radius
  }
  //vector<int> mask(npts);
  //bernoulli(mask.data(), npts);

  float* ptr_pts = pts.mutable_points();
  float* ptr_normal = pts.mutable_normal();
  for (int i = 0; i < npts; ++i) {
    //if (mask[i] == 0) continue;
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
    cflags::PrintHelpInfo("\nUsage: points_noise");
    return 0;
  }

  // file path
  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") { mkdir(output_path); }
  else { output_path = extract_path(file_path); }
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);

  // declare the random number generator as global variables
  BernoulliRand bernoulli(FLAGS_ratio);
  GaussianRand pt_noise(FLAGS_dp), normal_noise(FLAGS_dn);

  for (int i = 0; i < all_files.size(); i++) {
    Points pts;
    pts.read_points(all_files[i]);

    add_noise(pts, pt_noise, normal_noise, bernoulli);

    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    filename = output_path + filename + ".points";
    pts.write_points(filename);
  }

  cout << "Done: " << FLAGS_filenames << endl;
  return 0;
}
