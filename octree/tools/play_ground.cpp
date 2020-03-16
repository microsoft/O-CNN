#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "filenames.h"
#include "points.h"
#include "octree.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;
//
//DEFINE_string(filenames, kRequired, "", "The input filenames");
//DEFINE_string(output_path, kOptional, ".", "The output path");
//DEFINE_bool(verbose, kOptional, true, "Output logs");
//
//void octree2pts(Points& point_cloud, const vector<int>& label3, Octree& oct) {
//  const int depth = oct.info().depth();
//
//  vector<int> label0(label3), label1;
//  for (int d = 3; d < 5; ++d) {
//    int nnum = oct.info().node_num(d);
//    const int* child = oct.children_cpu(d);
//
//    label1.resize(oct.info().node_num(d + 1));
//    for (int i = 0; i < nnum; ++i) {
//      int t = child[i];
//      if (t >= 0) { // non-empty
//        for (int j = 0; j < 8; ++j) {
//          label1[t * 8 + j] = label0[i];
//        }
//      }
//    }
//
//    label0.swap(label1);
//  }
//
//  int num = oct.info().node_num(depth);
//  const int* child_d = oct.children_cpu(depth);
//  vector<float> pts, normals, labels;
//  for (int i = 0; i < num; ++i) {
//    float n[3], pt[3];
//    oct.node_normal(n, i, depth);
//    float len = abs(n[0]) + abs(n[1]) + abs(n[2]);
//    if (len == 0) continue;
//    oct.node_pos(pt, i, depth);
//
//    for (int c = 0; c < 3; ++c) {
//      normals.push_back(n[c]);
//      pts.push_back(pt[c]); // !!! note the scale and bbmin
//    }
//    labels.push_back(label0[i]);
//  }
//
//  point_cloud.set_points(pts, normals, vector<float>(), labels);
//}
//
//void load_labels(vector<int>&label3, const string& filename) {
//  label3.clear();
//  ifstream infile(filename);
//  if (!infile) return;
//
//  while (infile) {
//    int i;
//    infile >> i;
//    label3.push_back(i);
//  }
//  infile.close();
//}
//
//int main(int argc, char* argv[]) {
//  bool succ = cflags::ParseCmd(argc, argv);
//  if (!succ) {
//    cflags::PrintHelpInfo("\nUsage: points2ply.exe");
//    return 0;
//  }
//
//  // file path
//  string file_path = FLAGS_filenames;
//  string output_path = FLAGS_output_path;
//  if (output_path != ".") mkdir(output_path);
//  else output_path = extract_path(file_path);
//  output_path += "/";
//
//  vector<string> all_files;
//  get_all_filenames(all_files, file_path);
//
//  for (int i = 0; i < all_files.size(); i++) {
//    Octree oct;
//    oct.read_octree(all_files[i]);
//    
//    string filename = all_files[i];
//    filename.replace(filename.find(".octree"), string::npos, ".txt");
//    vector<int> label3;
//    load_labels(label3, filename);
//
//    Points pts;
//    octree2pts(pts, label3, oct);
//    
//    filename = extract_filename(filename);
//    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
//    filename = output_path + filename + ".points";
//    pts.write_points(filename);
//  }
//
//  return 0;
//}


//////////////////////////////////////////

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(verbose, kOptional, true, "Output logs");

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: points2ply.exe");
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
    Octree oct;
    oct.read_octree(all_files[i]);
    
    int depth = oct.info().depth();
    int num = oct.info().node_num(depth);
    const float* feat = oct.feature_cpu(depth);
    vector<float> pts(3 * num, 0), buf(num, 0);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < 3; ++j) {
        pts[i * 3 + j] = feat[(3 + j)*num + i];
      }
    }

    Points point_cloud;
    point_cloud.set_points(pts, vector<float>(), buf);

    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;
    filename = output_path + filename + ".points";
    point_cloud.write_points(filename);
  }

  return 0;
}
