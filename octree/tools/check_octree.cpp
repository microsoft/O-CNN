#include <iostream>
#include <string>
#include <vector>

#include "filenames.h"
#include "cmd_flags.h"
#include "octree.h"

using namespace std;

DEFINE_string(filenames, kRequired, "", "The input filenames");

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: check_octree.exe");
    return 0;
  }

  vector<string> all_files;
  get_all_filenames(all_files, FLAGS_filenames);

  for (int i = 0; i < all_files.size(); i++) {
    cout << "\n===============" << endl;

    // get filename
    size_t pos = all_files[i].rfind('\\') + 1;
    string filename = all_files[i].substr(pos);
    cout << filename << " infomation:" << std::endl;

    // load octree
    Octree octree;
    bool succ = octree.read_octree(all_files[i]);
    if (!succ) {
      cout << "Can not load " << filename << std::endl;
      continue;
    }

    // check format
    string msg;
    succ = octree.info().check_format(msg);
    if (!succ) {
      cout << filename << std::endl << msg << std::endl;
      continue;
    } else {
      cout << "This is a valid octree!" << endl;
    }

    // output info
    int depth = octree.info().depth();
    cout << "magic_str:" << OctreeInfo::kMagicStr << endl;
    cout << "batch_size: " << octree.info().batch_size() << endl;
    cout << "depth: " << octree.info().depth() << endl;
    cout << "full_layer: " << octree.info().full_layer() << endl;
    cout << "adaptive_layer: " << octree.info().adaptive_layer() << endl;
    cout << "threshold_distance: " << octree.info().threshold_distance() << endl;
    cout << "threshold_normal: " << octree.info().threshold_normal() << endl;
    cout << "is_adaptive: " << octree.info().is_adaptive() << endl;
    cout << "has_displace: " << octree.info().has_displace() << endl;
    cout << "nnum: ";
    for (int d = 0; d < 16; ++d) {
      cout << octree.info().node_num(d) << " ";
    }
    cout << endl << "nnum_cum: ";
    for (int d = 0; d < 16; ++d) {
      cout << octree.info().node_num_cum(d) << " ";
    }
    cout << endl << "nnum_nempty: ";
    for (int d = 0; d < 16; ++d) {
      cout << octree.info().node_num_nempty(d) << " ";
    }
    cout << endl << "total_nnum: " << octree.info().total_nnum() << endl;
    cout << "total_nnum_capacity: " << octree.info().total_nnum_capacity() << endl;
    cout << "channel: ";
    for (int j = 0; j < OctreeInfo::kPTypeNum; ++j) {
      cout << octree.info().channel(static_cast<OctreeInfo::PropType>(1 << j)) << " ";
    }
    cout << endl << "locations: ";
    for (int j = 0; j < OctreeInfo::kPTypeNum; ++j) {
      cout << octree.info().locations(static_cast<OctreeInfo::PropType>(1 << j)) << " ";
    }
    cout << endl << "bbox_max: ";
    for (int j = 0; j < 3; ++j) {
      cout << octree.info().bbmax()[j] << " ";
    }
    cout << endl << "bbox_min: ";
    for (int j = 0; j < 3; ++j) {
      cout << octree.info().bbmin()[j] << " ";
    }
    cout << "\nkey2xyz: " << octree.info().is_key2xyz() << endl;
    cout << "sizeof_octree: " << octree.info().sizeof_octree() << endl;
    cout << "===============\n" << endl;
  }

  return 0;
}
