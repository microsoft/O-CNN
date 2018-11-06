#ifndef LMDB_BUILDER_H_
#define LMDB_BUILDER_H_

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>

#include <memory>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

namespace db = caffe::db;
using std::shared_ptr;
using std::string;
using caffe::Datum;

class LmdbBuilder
{
public:
    LmdbBuilder();
    virtual ~LmdbBuilder();

    void AddData(const string& buffer, int label);
    void Close();
    void Open(const string& dbPath);

private:
    void CheckOpen();
    void CheckClosed();
    shared_ptr<db::DB> m_db;
    shared_ptr<db::Transaction> m_txn;
    int m_count;
    bool m_open;
};

#endif
