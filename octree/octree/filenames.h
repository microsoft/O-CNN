#ifndef _OCTREE_FILENAMES_
#define _OCTREE_FILENAMES_

#include <vector>
#include <string>

using std::vector;
using std::string;

void mkdir(const string& dir);

string extract_path(string str);
string extract_filename(string str);
string extract_suffix(string str);

void get_all_filenames(vector<string>& all_filenames, const string& filename);

#endif // _OCTREE_FILENAMES_
