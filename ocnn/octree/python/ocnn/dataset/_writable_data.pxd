from libcpp.string cimport string

cdef class WritableData:
    cdef string cpp_string
