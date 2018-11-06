#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/format.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "glog/logging.h"
#include "ocnn/caffe/lmdb_builder.h"

using caffe::Datum;
using namespace std;
namespace db = caffe::db;


LmdbBuilder::LmdbBuilder()
    : m_count(0),
      m_open(false)
{
}

void LmdbBuilder::CheckOpen()
{
    if (!m_open)
    {
        throw logic_error("Database is closed");
    }
}

void LmdbBuilder::CheckClosed()
{
    if (m_open)
    {
        throw logic_error("Database is still open");
    }
}

void LmdbBuilder::Open(const string& dbPath)
{
    CheckClosed();
    m_db = unique_ptr<db::DB>(db::GetDB("lmdb"));
    m_db->Open(dbPath.c_str(), db::NEW);
    m_txn = unique_ptr<db::Transaction>((m_db->NewTransaction()));

    m_count = 0;
    m_open = true;
}

void LmdbBuilder::Close()
{
    if (m_open)
    {
        if (m_count % 1000 != 0)
        {
            m_txn->Commit();
        }
    }
    m_open = false;
    m_count = 0;
}

LmdbBuilder::~LmdbBuilder()
{
}

void LmdbBuilder::AddData(const string& buffer, int label)
{
    Datum datum;
    datum.set_label(label);
    datum.set_data(buffer);
    datum.set_channels(buffer.size());
    datum.set_height(1);
    datum.set_width(1);

    string key_str = caffe::format_int(m_count, 8);
    string out;
    CHECK(datum.SerializeToString(&out));
    m_txn->Put(key_str, out);

    m_count += 1;
    if (m_count % 1000 == 0) {
        m_txn->Commit();
        m_txn.reset(m_db->NewTransaction());
    }
}

