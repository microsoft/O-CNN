#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

#include "filenames.h"
#include "points.h"
#include "octree.h"
#include "cmd_flags.h"

using namespace std;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(verbose, kOptional, true, "Output logs");

void octree2pts(vector<int>& pts, vector<int>& labels, const Octree& octree) {
  const int depth = octree.info().depth();
  const int num = octree.info().node_num(depth);

  const int* child = octree.children_cpu(depth);
  const float* label = octree.label_cpu(depth);
  const float* normal = octree.feature_cpu(depth);
  const uintk* key = octree.key_cpu(depth);

  pts.clear(); labels.clear();
  for (int i = 0; i < num; ++i) {
    float n[3] = { normal[i], normal[i + num], normal[i + 2 * num] };
    float len = fabs(n[0]) + fabs(n[1]) + fabs(n[2]);
    if (len == 0) continue;

    int pt[3];
    octree.key2xyz(pt, key[i], depth);
    for (int c = 0; c < 3; ++c) {
      pts.push_back(pt[c]);
    }
    labels.push_back(static_cast<int>(label[i]));
  }
}


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree2pts_suncg.exe");
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
    string filename = extract_filename(all_files[i]);;
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    // load octree
    Octree octree;
    bool succ = octree.read_octree(all_files[i]);
    if (!succ) {
      if (FLAGS_verbose) cout << "Can not load " << filename << std::endl;
      continue;
    }
    string msg;
    succ = octree.info().check_format(msg);
    if (!succ) {
      if (FLAGS_verbose) cout << filename << std::endl << msg << std::endl;
      continue;
    }

    // convert
    vector<int> pts, labels;
    octree2pts(pts, labels, octree);

    // write
    filename = output_path + filename + ".suncg";
    ofstream outfile(filename, ios::binary);
    if (!outfile) {
      cout << "Can not open :" << filename << endl;
      continue;
    }

    int num = labels.size();
    outfile.write((char*)(&num), sizeof(int));
    outfile.write((char*)pts.data(), sizeof(int) * 3 * num);
    outfile.write((char*)labels.data(), sizeof(int) * num);

    outfile.close();
  }

  return 0;
}
