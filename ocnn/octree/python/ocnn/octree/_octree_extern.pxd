# distutils: language = c++

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "points.h":
    cdef cppclass PointsData:
        int npt
        const float* pts
        const float* normals
        const float* seg
        const float* features
        const float* labels

cdef extern from "points.h":
    cdef cppclass PointsBounds:
        float radius
        float center[3]

cdef extern from "points.h" nogil:
    cdef cppclass Points:
        Points()
        bool read_points(const string&)
        bool write_points(const string&)
        PointsData get_points_data()
        PointsBounds get_points_bounds()
        void center_about(const float*)
        void displace(const float)
        void rotate(const float, const float*)
        void transform(const float*)

cdef extern from "octree_info.h" namespace "OctreeInfo":
    cdef enum PropType:
        kKey=1
        kChild=2
        kNeigh=4
        kFeature=8
        kLabel=16
        kSplit=32

cdef extern from "octree_info.h" nogil:
    cdef cppclass OctreeInfo:
        void set_batch_size(int)
        void set_depth(int)
        void set_full_layer(int)
        void set_channel(OctreeInfo.PropType, int)
        void set_bbox(const float*, const float*)
        void set_key2xyz(bool)
        void set_node_dis(bool)
        void set_adaptive(bool)
        void set_adaptive_layer(int)
        void set_threshold_dist(float)
        void set_threshold_normal(float)

cdef extern from "octree.h" nogil:
    cdef cppclass Octree:
        Octree()
        void build(const OctreeInfo&, const Points&);
        bool save(const string&)
        string get_binary_string()
