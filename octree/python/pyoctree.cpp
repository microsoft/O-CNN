#include <string>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <points.h>
#include <octree_samples.h>
#include <marching_cube.h>

namespace py = pybind11;

int add(int i, int j) {
  return i + j;
}

py::array_t<double> make_array(const py::ssize_t size) {
  // No pointer is passed, so NumPy will allocate the buffer
  return py::array_t<double>(size);
}

PYBIND11_MODULE(pyoctree, m) {
  // examples
  m.def("add", &add);
  m.def("subtract", [](int i, int j) { return i - j; });

  //m.def("make_array", &make_array,
  //  py::return_value_policy::move);
  // Return policy can be left default, i.e. return_value_policy::automatic

  // pyoctree interface
  m.def("get_one_octree", [](const char *name) {
    size_t size = 0;
    const char* str = (const char*)octree::get_one_octree(name, &size);
    return py::bytes(std::string(str, size));
  });


  using vectorf = vector<float>;
  using vectorfc = const vector<float>;
  auto Points_set_points = (bool(Points::*)(vectorfc&, vectorfc&, vectorfc&,
              vectorfc&))&Points::set_points;
  auto Mcube_compute = (void(MarchingCube::*)(vectorfc&, float, vectorfc&,
              int))&MarchingCube::compute;

  // points interface
  py::class_<Points>(m, "Points")
  .def(py::init<>())
  .def("read_points", &Points::read_points)
  .def("write_points", &Points::write_points)
  .def("write_ply", &Points::write_ply)
  .def("normalize", &Points::normalize)
  .def("orient_normal", &Points::orient_normal)
  .def("set_points", Points_set_points)
  .def("pts_num", [](const Points & pts) {
    return pts.info().pt_num();
  })
  // todo: fix the functions points(), normals(), labels(),
  // It is inefficient, since there is a memory-copy here
  .def("points", [](const Points & pts) {
    const float* ptr = pts.points();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num * 3);
  })
  .def("normals", [](const Points & pts) {
    const float* ptr = pts.normal();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num * 3);
  })
  .def("features", [](const Points & pts) {
    const float* ptr = pts.feature();
    int num = pts.info().pt_num();
    const int ch = pts.info().channel(PointsInfo::kFeature);
    return vectorf(ptr, ptr + num * ch);
  })
  .def("labels", [](const Points & pts) {
    const float* ptr = pts.label();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num);
  })
  .def("buffer", [](const Points & pts) {
    const char* ptr = pts.data();
    return py::bytes(std::string(ptr, pts.info().sizeof_points()));
  });

  // marching cube
  py::class_<MarchingCube>(m, "MCube")
  .def(py::init<>())
  .def("compute", Mcube_compute)
  .def("get_vtx", [](const MarchingCube & mcube) {
    return mcube.vtx_;
  })
  .def("get_face", [](const MarchingCube & mcube) {
    return mcube.face_;
  });
}
