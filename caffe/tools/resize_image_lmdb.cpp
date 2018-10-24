#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;
using std::pair;
using std::vector;
using boost::scoped_ptr;
using std::ofstream;

#ifdef USE_OPENCV

DEFINE_string(type, "resize", "Choose from \"resize\" and \"revert\"");
DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 224, "Width images are resized to");
DEFINE_int32(resize_height, 224, "Height images are resized to");
DEFINE_bool(data_list_only, false, "Only output the data list file");

cv::Mat DatumToCVMat(const Datum& datum) {
  int channel = datum.channels();
  int height = datum.height();
  int width = datum.width();
  const string& buffer = datum.data();
  CHECK_EQ(channel, 3) << "Only color image is supported!";
  cv::Mat cv_img(height, width, CV_8UC3);
  for (int h = 0; h < height; ++h) {
    uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channel; ++c) {
        int datum_index = (c * height + h) * width + w;
        ptr[img_index++] = static_cast<char>(buffer[datum_index]);
      }
    }
  }
  return cv_img;
}

void revert_image_database(const string& db_name_input, string& path_output) {
  scoped_ptr<db::DB> db_input(db::GetDB(FLAGS_backend));
  db_input->Open(db_name_input, db::READ);
  scoped_ptr<db::Cursor> cursor_input(db_input->NewCursor());

  LOG(INFO) << "Starting ...";
  int count = 0;
  Datum datum_input;
  string datalist;
  while (cursor_input->valid()) {
    datum_input.ParseFromString(cursor_input->value());
    string key_str = cursor_input->key();
    cv::Mat img = DatumToCVMat(datum_input);

    // update datalist
    char buffer[64];
    sprintf(buffer, " %d\n", datum_input.label());
    datalist += key_str + buffer;

    // save the image to disk
    if (!FLAGS_data_list_only) {
      string filename = key_str;
      for (int i = 0; i < filename.size(); ++i) {
        if (filename[i] == '\\' || filename[i] == '/') filename[i] = '_';
      }
      cv::imwrite(path_output + filename, img);
    }

    // next datum
    cursor_input->Next();
    if (++count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }

  ofstream outfile(path_output + "datalist.txt");
  CHECK(outfile) << path_output + "datalist.txt";
  outfile.write(datalist.data(), datalist.size());
  outfile.close();
}

void resize_image_database(const string& db_name_input, const string& db_name_output,
    const int width, const int height) {
  scoped_ptr<db::DB> db_input(db::GetDB(FLAGS_backend));
  db_input->Open(db_name_input, db::READ);
  scoped_ptr<db::Cursor> cursor_input(db_input->NewCursor());

  LOG(INFO) << "Writing data to DB";
  scoped_ptr<db::DB> db_output(db::GetDB(FLAGS_backend));
  db_output->Open(db_name_output, db::NEW);
  scoped_ptr<db::Transaction> txn_output(db_output->NewTransaction());

  LOG(INFO) << "Starting ...";
  int count = 0;
  string octree_output, out_str;
  Datum datum_input, datum_output;
  while (cursor_input->valid()) {
    string key_str = cursor_input->key();
    datum_input.ParseFromString(cursor_input->value());
    cv::Mat img = DatumToCVMat(datum_input);

    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(width, height));
    CVMatToDatum(img_resize, &datum_output);
    datum_output.set_label(datum_input.label());

    CHECK(datum_output.SerializeToString(&out_str));
    txn_output->Put(key_str, out_str);

    cursor_input->Next();
    if (++count % 1000 == 0) {
      txn_output->Commit();
      txn_output.reset(db_output->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  if (count % 10000 != 0) {
    txn_output->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;  // Print output to stderr (while still logging)
  gflags::SetUsageMessage("\nUsage:\n"
      "\trevert_octree_database [FLAGS] INPUT_DB OUTPUT_DB/\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  if (FLAGS_type == "revert") {
    string db_input(argv[1]), path_output(argv[2]);
    revert_image_database(db_input, path_output);
  } else {
    string db_input(argv[1]), db_output(argv[2]);
    resize_image_database(db_input, db_output, FLAGS_resize_width, FLAGS_resize_height);
  }

  return 0;
}
#else
int main() {
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
  return 0;
}
#endif  // USE_OPENCV