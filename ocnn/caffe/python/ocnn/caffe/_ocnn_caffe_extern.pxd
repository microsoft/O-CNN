# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "ocnn/caffe/lmdb_builder.h" nogil:
    cdef cppclass LmdbBuilder:
        LmdbBuilder()
        void Open(const string&)
        void AddData(const string&, int)
        void Close()
