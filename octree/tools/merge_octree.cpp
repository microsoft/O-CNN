#include <fstream>
#include <iostream>

#include "octree.h"
#include "merge_octrees.h"
#include "cmd_flags.h"
#include "filenames.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_int(num, kOptional, 2, "The number of octrees to be merged");


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: merge_octrees");
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
  for (int i = 0; i < all_files.size() / FLAGS_num; i ++) {
    // load FLAGS_num octrees
    vector<const char*> octrees_in(FLAGS_num);
    vector<Octree> octrees(FLAGS_num);
    for (int j = 0; j < FLAGS_num; ++j) {
      string filename = all_files[i * FLAGS_num + j];
      bool succ = octrees[j].read_octree(filename);
      if (!succ) {
        cout << "Can not load: " << filename << endl;
        continue;
      }
      octrees_in[j] = octrees[j].ptr_raw_cpu();
    }

    // merge
    vector<char> octree_out;
    merge_octrees(octree_out, octrees_in);

    // save
    char buffer[64];
    sprintf(buffer, "batch_%03d.octree", i);
    cout << "Processing: " << buffer << endl;
    string filename = output_path + buffer;
    ofstream outfile(filename, ios::binary);
    outfile.write(octree_out.data(), octree_out.size());
    outfile.close();
  }

  cout << "Done: " << FLAGS_filenames << endl;
  return 0;
}