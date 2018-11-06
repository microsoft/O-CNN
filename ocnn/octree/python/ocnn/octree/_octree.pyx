from ocnn.octree cimport _octree_extern
from ocnn.dataset._writable_data cimport WritableData

import numpy as np
cimport numpy as np

from cython.operator cimport dereference
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

    def center(self):
        _, center = self.get_points_bounds()
        self.center_about(center)

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
        cdef _octree_extern.PointsBounds points_bounds
        with nogil:
            points_bounds = self.c_points.get_points_bounds()
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

cdef class OctreeInfo:
    cdef _octree_extern.OctreeInfo c_octree_info
    def initialize(
            self,
            int depth,
            int full_depth,
            bool node_displacement,
            bool node_feature,
            bool split_label,
            bool adaptive,
            int adaptive_depth,
            float threshold_distance,
            float threshold_normal,
            bool key2xyz,
            Points points):
        c_points_ptr = &points.c_points
        self.c_octree_info.initialize(
                depth,
                full_depth,
                node_displacement,
                node_feature,
                split_label,
                adaptive,
                adaptive_depth,
                threshold_distance,
                threshold_normal,
                key2xyz,
                dereference(c_points_ptr))

    def set_bbox(self, float radius, np.ndarray center):
        center = _ensure_contiguous(center)
        _check_array(center, (3,))

        cdef float[::1] center_view = center.ravel()
        self.c_octree_info.set_bbox(radius, &center_view[0])

    def set_bbox(self, np.ndarray bbox_min, np.ndarray bbox_max):

        bbox_min = _ensure_contiguous(bbox_min)
        _check_array(bbox_min, (3,))
        cdef float[::1] bbox_min_view = bbox_min.ravel()

        bbox_max = _ensure_contiguous(bbox_max)
        _check_array(bbox_max, (3,))
        cdef float[::1] bbox_max_view = bbox_max.ravel()

        self.c_octree_info.set_bbox(&bbox_min_view[0], &bbox_max_view[0])

cdef class Octree(WritableData):
    cdef _octree_extern.Octree c_octree

    def __cinit__(self, OctreeInfo info, Points points):
        c_points_ptr = &points.c_points
        c_info_ptr = &info.c_octree_info
        with nogil:
            self.c_octree.build(dereference(c_info_ptr), dereference(c_points_ptr))
            self.cpp_string = self.c_octree.get_binary_string()

    def write_file(self, filename):
        cdef string stl_string = filename.encode('UTF-8')
        with nogil:
            self.c_octree.save(stl_string)
