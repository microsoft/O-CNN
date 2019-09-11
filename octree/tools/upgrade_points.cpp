#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "filenames.h"
#include "points.h"
#include "cmd_flags.h"

using namespace std;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_bool(has_label, kOptional, false, "The file contains label");
DEFINE_bool(verbose, kOptional, true, "Output logs");


void load_pointcloud(vector<float>& pt, vector<float>& normal,
    vector<int>& seg, const string& filename) {
  const int channel = 3;
  std::ifstream infile(filename, std::ios::binary);

  infile.seekg(0, infile.end);
  size_t len = infile.tellg();
  infile.seekg(0, infile.beg);

  int n;
  infile.read((char*)(&n), sizeof(int));
  pt.resize(3 * n);
  infile.read((char*)pt.data(), sizeof(float) * 3 * n);
  normal.resize(channel * n);
  infile.read((char*)normal.data(), sizeof(float) * channel * n);

  if (FLAGS_has_label &&
      6 * n * sizeof(float) + (n + 1) * sizeof(int) == len) {
    seg.resize(n);
    infile.read((char*)seg.data(), sizeof(int)*n);
  }

  infile.close();
}

void gen_test_pointcloud(vector<float>& pt, vector<float>& normal,
    vector<float>& label) {
  const float pt_input[] = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  const float nm_input[] = { 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f };
  const float lb_input[] = { 0.0f, 1.0f };

  pt.assign(pt_input, pt_input + 6);
  normal.assign(nm_input, nm_input + 6);
  label.assign(lb_input, lb_input + 2);
}

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: upgrade_points.exe");
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
  //#pragma omp parallel for
  for (int i = 0; i < all_files.size(); ++i) {
    string filename = extract_filename(all_files[i]);

    /// from the old formate to this new format
    vector<float> pts, normals, labels;
    vector<int> seg;
    load_pointcloud(pts, normals, seg, all_files[i]);
    if (FLAGS_has_label && seg.size() == 0) {
      cout << "Error in " << filename << endl;
      continue;
    }
    labels.assign(seg.begin(), seg.end());

    Points points;
    points.set_points(pts, normals, vector<float>(), labels);

    points.write_points(output_path + filename + ".upgrade.points");

    if (FLAGS_verbose) cout << "Processing: " + filename + "\n";
  }
  cout << "Done: " << FLAGS_filenames << endl;
  //// generate testing points
  //vector<float> pts, normals, labels;
  //gen_test_pointcloud(pts, normals, labels);
  //Points points;
  //points.set_points(pts, normals, vector<float>(), labels);
  //points.write_points("test.points");

  ///// backward conversion: from this new formate to the old format
  //points.load_points(filename + ".points");
  //ofstream outfile(filename + "_1.points", ofstream::binary);
  //int n = points.info().pt_num();
  //outfile.write((char*)(&n), sizeof(int));
  //outfile.write((const char*)points.prop_ptr(PtsInfo::kPoint),
  //  n * sizeof(float) * points.info().channel(PtsInfo::kPoint));
  //outfile.write((const char*)points.prop_ptr(PtsInfo::kNormal),
  //  n * sizeof(float) * points.info().channel(PtsInfo::kNormal));
  //outfile.close();

  return 0;
}
