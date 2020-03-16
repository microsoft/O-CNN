#include <fstream>
#include <string>
#include <iostream>
#include <vector>

#include "cmd_flags.h"
#include "octree.h"
#include "filenames.h"
#include "math_functions.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;
using cflags::Require;
const float kPI = 3.14159265f;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_string(axis, kOptional, "y", "The upright axis of the input model");
DEFINE_int(depth, kOptional, 6, "The maximum depth of the octree");
DEFINE_int(full_depth, kOptional, 2, "The full layer of the octree");
DEFINE_int(rot_num, kOptional, 12, "Number of poses rotated along the upright axis");
DEFINE_float(offset, kOptional, 0.55f, "The offset value for handing thin shapes");
DEFINE_bool(node_dis, kOptional, false, "Output per-node displacement");
DEFINE_bool(node_feature, kOptional, false, "Compute per node feature");
DEFINE_bool(split_label, kOptional, false, "Compute per node splitting label");
DEFINE_bool(adaptive, kOptional, false, "Build adaptive octree");
DEFINE_int(adp_depth, kOptional, 4, "The starting depth of adaptive octree");
DEFINE_float(th_distance, kOptional, 2.0f, "The threshold for simplifying octree");
DEFINE_float(th_normal, kOptional, 0.1f, "The threshold for simplifying octree");
DEFINE_bool(key2xyz, kOptional, false, "Convert the key to xyz when serialization");
DEFINE_bool(extrapolate, kOptional, false, "Exptrpolate the node feature");
DEFINE_bool(save_pts, kOptional, false, "Save the average points as signal");
DEFINE_bool(verbose, kOptional, true, "Output logs");


// OctreeBuilder shows a basic example for building an octree with a point cloud
class OctreeBuilder {
 public:
  bool set_point_cloud(string filename) {
    // load point cloud
    bool succ = point_cloud_.read_points(filename);
    if (!succ) {
      cout << "Can not load " << filename << endl;
      return false;
    }
    string msg;
    succ = point_cloud_.info().check_format(msg);
    if (!succ) {
      cout << filename << endl << msg << endl;
      return false;
    }

    // deal with empty points
    int npt = point_cloud_.info().pt_num();
    if (npt == 0) {
      cout << "This is an empty points!" << endl;
      return false;
    }

    // bounding sphere
    bounding_sphere(radius_, center_, point_cloud_.points(), npt);
    //radius_ = 128.0;
    //center_[0] = center_[1] = center_[2] = 128.0;

    // centralize & displacement
    float trans[3] = { -center_[0], -center_[1], -center_[2] };
    point_cloud_.translate(trans);
    if (FLAGS_offset > 1.0e-10f) {
      float offset = FLAGS_offset * 2.0f * radius_ / float(1 << FLAGS_depth);
      point_cloud_.displace(offset);
      radius_ += offset;
    }

    return true;
  }

  void set_octree_info() {
    octree_info_.initialize(FLAGS_depth, FLAGS_full_depth, FLAGS_node_dis,
        FLAGS_node_feature, FLAGS_split_label, FLAGS_adaptive, FLAGS_adp_depth,
        FLAGS_th_distance, FLAGS_th_normal, FLAGS_key2xyz, FLAGS_extrapolate,
        FLAGS_save_pts, point_cloud_);

    // the point cloud has been centralized,
    // so initializing the bbmin & bbmax in the following way
    float bbmin[] = { -radius_, -radius_, -radius_ };
    float bbmax[] = { radius_, radius_, radius_ };
    octree_info_.set_bbox(bbmin, bbmax);
  }

  void build_octree() {
    octree_.build(octree_info_, point_cloud_);
  }

  void save_octree(const string& output_filename) {
    // Modify the bounding box before saving, because the center of
    // the point cloud is translated to (0, 0, 0) when building the octree
    octree_.mutable_info().set_bbox(radius_, center_);
    octree_.write_octree(output_filename);
  }

 public:
  Points point_cloud_;
  float radius_, center_[3];
  OctreeInfo octree_info_;
  Octree octree_;
};


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: Octree.exe");
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

  #pragma omp parallel for
  for (int i = 0; i < all_files.size(); i++) {
    OctreeBuilder builder;
    bool succ = builder.set_point_cloud(all_files[i]);

    string filename = extract_filename(all_files[i]);
    if (!succ) {
      if (FLAGS_verbose) cout << "Warning: " + filename + " is invalid!\n";
      continue;
    }
    builder.set_octree_info();

    // data augmentation
    float angle = 2.0f * kPI / float(FLAGS_rot_num);
    float axis[] = { 0.0f, 0.0f, 0.0f };
    if (FLAGS_axis == "x") axis[0] = 1.0f;
    else if (FLAGS_axis == "y") axis[1] = 1.0f;
    else axis[2] = 1.0f;

    if (FLAGS_verbose) cout << "Processing: " + filename + "\n";
    for (int v = 0; v < FLAGS_rot_num; ++v) {
      // output filename
      char file_suffix[64];
      sprintf(file_suffix, "_%d_%d_%03d.octree", FLAGS_depth, FLAGS_full_depth, v);

      // build
      builder.build_octree();

      // save octree
      builder.save_octree(output_path + filename + file_suffix);

      // rotate point for the next iteration
      builder.point_cloud_.rotate(angle, axis);

      // message
      //cout << "Processing: " << filename.substr(filename.rfind('\\') + 1) << endl;
    }
  }

  if (FLAGS_verbose) cout << "Done: " << FLAGS_filenames << endl;
  return 0;
}

