from ocnn.caffe cimport _ocnn_caffe_extern
from ocnn.dataset._writable_data cimport WritableData

import os

from libcpp.string cimport string

cdef class LmdbBuilder:
    cdef _ocnn_caffe_extern.LmdbBuilder c_builder

    def __cinit__(self):
        pass

    def open(self, db_path):
        if os.path.exists(db_path):
            raise RuntimeError("DB Path {0} already exists".format(db_path))

        cdef string stl_string = db_path.encode('UTF-8')
        self.c_builder.Open(stl_string)

    def close(self):
        self.c_builder.Close()

    def add_data(self, WritableData writable_data, int label):
        with nogil:
            self.c_builder.AddData(writable_data.cpp_string, label)


