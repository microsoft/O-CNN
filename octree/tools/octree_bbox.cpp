#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "filenames.h"
#include "octree.h"
#include "cmd_flags.h"

using namespace std;
using cflags::Require;

DEFINE_string(filenames, kRequired, "", "The input filenames");

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree_bbox");
    return 0;
  }

  string file_path = FLAGS_filenames;
  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  std::sort(all_files.begin(), all_files.end());

  for (int i = 0; i < all_files.size(); i += 2) {
    string filename = extract_filename(all_files[i + 1]);
    cout << filename << "\n";

    // load octree
    Octree octree1, octree2;
    octree1.read_octree(all_files[i]);
    octree2.read_octree(all_files[i + 1]);

    // update bbox
    const float* bbmin = octree1.info().bbmin();
    const float* bbmax = octree1.info().bbmax();
    octree2.mutable_info().set_bbox(bbmin, bbmax);

    // save octree
    octree2.write_octree(all_files[i + 1]);
  }

  return 0;
}
