#include "mesh.h"

#include <iostream>
#include <random>
#include <ctime>
#include "points.h"
#include "filenames.h"
#include "cmd_flags.h"

DEFINE_string(filenames, kRequired, "", "The input filenames");
DEFINE_string(output_path, kOptional, ".", "The output path");
DEFINE_float(area_unit, kOptional, 1.0, "The area unit used to sample points");
DEFINE_bool(verbose, kOptional, true, "Output logs");

using std::cout;

std::default_random_engine generator(static_cast<unsigned int>(time(nullptr)));
std::uniform_real_distribution<float> distribution(0.01, 0.99);

void sample_points(vector<float>& pts, vector<float>& normals,
    const vector<float>& V, const vector<int>& F, float area_unit) {
  vector<float> face_normal, face_center, face_area;
  compute_face_normal(face_normal, face_area, V, F);
  compute_face_center(face_center, V, F);

  float avg_area = 0;
  for (auto& a : face_area) { avg_area += a;}
  avg_area /= face_area.size();
  area_unit *= avg_area;
  if (area_unit <= 0) area_unit = 1.0e-5f;

  int nf = F.size() / 3;
  vector<float> point_num(nf);
  int total_pt_num = 0;
  for (int i = 0; i < nf; ++i) {
    int n = static_cast<int>(face_area[i] / area_unit + 0.5f);
    if (n < 1) n = 1;
    if (n > 100) n = 100;
    point_num[i] = n;
    total_pt_num += n;
  }

  pts.resize(3 * total_pt_num);
  normals.resize(3 * total_pt_num);
  for (int i = 0, id = 0; i < nf; ++i) {
    int ix3 = i * 3, idx3 = id * 3;
    for (int k = 0; k < 3; ++k) {
      pts[idx3 + k] = face_center[ix3 + k];
      normals[idx3 + k] = face_normal[ix3 + k];
    }

    for (int j = 1; j < point_num[i]; ++j) {
      float x = 0, y = 0, z = 0;
      while (z < 0.01 || z > 0.99) {
        x = distribution(generator);
        y = distribution(generator);
        z = 1.0 - x - y;
      }
      idx3 = (id + j) * 3;
      int f0x3 = F[ix3] * 3, f1x3 = F[ix3 + 1] * 3, f2x3 = F[ix3 + 2] * 3;
      for (int k = 0; k < 3; ++k) {
        pts[idx3 + k] = x * V[f0x3 + k] + y * V[f1x3 + k] + z * V[f2x3 + k];
        normals[idx3 + k] = face_normal[ix3 + k];

      }
    }
    id += point_num[i];
  }
}

int main(int argc, char* argv[]) {
  bool succ = cflags::ParseCmd(argc, argv);
  if (!succ) {
    cflags::PrintHelpInfo("\nUsage: mesh2points.exe");
    return 0;
  }

  string file_path = FLAGS_filenames;
  string output_path = FLAGS_output_path;
  if (output_path != ".") mkdir(output_path);
  else output_path = extract_path(file_path);
  output_path += "/";

  vector<string> all_files;
  get_all_filenames(all_files, file_path);
  for (int i = 0; i < all_files.size(); i++) {
    // get filename
    string filename = extract_filename(all_files[i]);
    if (FLAGS_verbose) cout << "Processing: " << filename << std::endl;

    // load mesh
    vector<float> V;
    vector<int> F;
    read_mesh(all_files[i], V, F);

    // sample points
    vector<float> pts, normals;
    sample_points(pts, normals, V, F, FLAGS_area_unit);

    // save points
    Points point_cloud;
    point_cloud.set_points(pts, normals);
    point_cloud.write_points(output_path + filename + ".points");
  }

  return 0;
}