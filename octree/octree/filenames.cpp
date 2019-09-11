#include "filenames.h"
#include <algorithm>


string extract_path(string str) {
  std::replace(str.begin(), str.end(), '\\', '/');
  size_t pos = str.rfind('/');
  if (string::npos == pos) {
    return string(".");
  } else {
    return str.substr(0, pos);
  }
}

string extract_filename(string str) {
  std::replace(str.begin(), str.end(), '\\', '/');
  size_t pos = str.rfind('/') + 1;
  size_t len = str.rfind('.');
  if (string::npos != len) len -= pos;
  return str.substr(pos, len);
}

string extract_suffix(string str) {
  string suffix;
  size_t pos = str.rfind('.');
  if (pos != string::npos) {
    suffix = str.substr(pos + 1);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
  }
  return suffix;
}


#if defined _MSC_VER
#include <direct.h>

void mkdir(const string& dir) {
  _mkdir(dir.c_str());
}

#elif defined __GNUC__

#include <sys/types.h>
#include <sys/stat.h>

void mkdir(const string& dir) {
  mkdir(dir.c_str(), 0744);
}
#endif


#ifdef USE_WINDOWS_IO
#include <io.h>

void get_all_filenames(vector<string>& all_filenames, const string& filename_in) {
  all_filenames.clear();
  string file_path = extract_path(filename_in) + "/";
  string filename = file_path + "*" + filename_in.substr(filename_in.rfind('.'));

  _finddata_t c_file;
  intptr_t hFile = _findfirst(filename.c_str(), &c_file);
  do {
    if (hFile == -1) break;
    all_filenames.push_back(file_path + string(c_file.name));
  } while (_findnext(hFile, &c_file) == 0);
  _findclose(hFile);
}

#else
#include <fstream>

void get_all_filenames(vector<string>& all_filenames, const string& data_list) {
  all_filenames.clear();

  std::ifstream infile(data_list);
  if (!infile) return;

  string line;
  while (std::getline(infile, line)) {
    all_filenames.push_back(line);
  }
  infile.close();
}

#endif
