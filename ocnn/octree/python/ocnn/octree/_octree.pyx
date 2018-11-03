from ocnn.octree cimport _octree_extern
from ocnn.dataset._writable_data cimport WritableData

import numpy as np
cimport numpy as np

from libcpp cimport bool
from libcpp.string cimport string

def _check_array(array, shape):
    if shape != array.shape:
        raise ValueError('Illegal array dimensionality {0}, expected {1}'.format(array.shape, shape))

def _ensure_contiguous(np.ndarray array):
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    return array

cdef class Points:
    cdef _octree_extern.Points c_points

    def __cinit__(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        cdef bool points_read
        with nogil:
            points_read = self.c_points.read_points(stl_string)
        if not points_read:
            raise RuntimeError('Could not read points file: {0}'.format(filename))

    def write_file(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_points.write_points(stl_string)

    def center_about(self, np.ndarray center):
        center = _ensure_contiguous(center)
        _check_array(center, (3,))

        cdef float[::1] center_view = center.ravel()

        with nogil:
            self.c_points.center_about(&center_view[0])

    def displace(self, float displacement):
        with nogil:
            self.c_points.displace(displacement)

    def rotate(self, float angle, np.ndarray axis):
        axis = _ensure_contiguous(axis)
        _check_array(axis, (3,))

        cdef float[::1] axis_view = axis.ravel()
        with nogil:
            self.c_points.rotate(angle, &axis_view[0])

    def transform(self, np.ndarray transformation_matrix):
        transformation_matrix = _ensure_contiguous(transformation_matrix)
        _check_array(transformation_matrix, (3,3))

        cdef float[::1] mat_view = transformation_matrix.ravel()

        with nogil:
            self.c_points.transform(&mat_view[0])

    def get_points_bounds(self):
        cdef _octree_extern.PointsBounds points_bounds = self.c_points.get_points_bounds()
        cdef float[:] center_view = points_bounds.center

        center = np.empty_like (center_view)
        center[:] = center_view

        return points_bounds.radius, center

    def get_points_data(self):
        cdef _octree_extern.PointsData points_data = self.c_points.get_points_data()
        cdef Py_ssize_t nrows = points_data.npt, ncols=3
        cdef const float[:,::1] points_view = <float[:nrows, :ncols]> points_data.pts
        cdef const float[:,::1] normals_view = <float[:nrows, :ncols]>points_data.normals

        points = np.empty_like (points_view)
        points[:] = points_view
        normals = np.empty_like (normals_view)
        normals[:] = normals_view

        return points, normals

cdef class PropType:
    kKey = _octree_extern.PropType.kKey
    kChild = _octree_extern.PropType.kChild
    kNeigh = _octree_extern.PropType.kNeigh
    kFeature = _octree_extern.PropType.kFeature
    kLabel = _octree_extern.PropType.kLabel
    kSplit = _octree_extern.PropType.kSplit

cdef class OctreeInfo:
    cdef _octree_extern.OctreeInfo c_octree_info
    def set_batch_size(self, int batch_size):
        self.c_octree_info.set_batch_size(batch_size)
    def set_depth(self, int depth):
        self.c_octree_info.set_depth(depth)
    def set_full_layer(self, int full_layer):
        self.c_octree_info.set_full_layer(full_layer)
    def set_channel(self, _octree_extern.PropType prop_type, int channel):
        self.c_octree_info.set_channel(prop_type, channel)
    def set_bbox(self, np.ndarray bbox_min, np.ndarray bbox_max):

        bbox_min = _ensure_contiguous(bbox_min)
        _check_array(bbox_min, (3,))
        cdef float[::1] bbox_min_view = bbox_min.ravel()

        bbox_max = _ensure_contiguous(bbox_max)
        _check_array(bbox_max, (3,))
        cdef float[::1] bbox_max_view = bbox_max.ravel()

        self.c_octree_info.set_bbox(&bbox_min_view[0], &bbox_max_view[0])
    def set_key2xyz(self, bool key2xyz):
        self.c_octree_info.set_key2xyz(key2xyz)
    def set_node_dis(self, bool node_dis):
        self.c_octree_info.set_node_dis(node_dis)
    def set_adaptive(self, bool adaptive):
        self.c_octree_info.set_adaptive(adaptive)
    def set_adaptive_layer(self, int adaptive_layer):
        self.c_octree_info.set_adaptive_layer(adaptive_layer)
    def set_threshold_dist(self, float threshold_dist):
        self.c_octree_info.set_threshold_dist(threshold_dist)
    def set_threshold_normal(self, float threshold_normal):
        self.c_octree_info.set_threshold_normal(threshold_normal)

cdef class Octree(WritableData):
    cdef _octree_extern.Octree c_octree

    def __cinit__(self, OctreeInfo info, Points points):
        c_points = points.c_points
        c_info = info.c_octree_info
        with nogil:
            self.c_octree.build(c_info, c_points)
            self.cpp_string = self.c_octree.get_binary_string()

    def write_file(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_octree.save(stl_string)
