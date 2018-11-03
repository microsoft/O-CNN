from ocnn.dataset cimport _writable_data

cdef class WritableData:
    def __cinit__(self):
        pass

    def get_buffer(self):
        return self.cpp_string



