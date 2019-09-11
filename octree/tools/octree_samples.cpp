#include <iostream>
#include <fstream>

#include "octree_samples.h"
#include "filenames.h"
#include "cmd_flags.h"


DEFINE_string(output_path, kRequired, ".", "The output path");

using std::cout;


int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: octree_samples.exe");
    return 0;
  }

  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  output_path += "/";

  for (int i = 1; i < 7; i++) {
    string filename = "octree_" + std::to_string(i);
    size_t size = 0;
    const char* ptr = (const char*)octree::get_one_octree(filename.c_str(), &size);

    std::ofstream outfile(output_path + filename + ".octree", std::ios::binary);
    outfile.write(ptr, size);
    outfile.close();

    cout << "Save: " << filename << std::endl;
  }

  return 0;
}