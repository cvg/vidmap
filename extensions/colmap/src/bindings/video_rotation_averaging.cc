#include "vidmap_native/video_rotation_averaging.h"

#include "bindings.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindVideoRotationAveraging(py::module_& m) {
  py::class_<VideoRotationAveragingOptions>(m, "RotationAveragingOptions")
      .def(py::init<>())
      .def_readwrite("random_seed", &VideoRotationAveragingOptions::random_seed)
      .def_readwrite("image_order_passes",
                     &VideoRotationAveragingOptions::image_order_passes)
      .def_readwrite("filter_unregistered_images",
                     &VideoRotationAveragingOptions::filter_unregistered)
      .def_readwrite("skip_risky_loop_closure_pairs",
                     &VideoRotationAveragingOptions::skip_risky_lc_pairs)
      .def_readwrite("max_rotation_error_deg",
                     &VideoRotationAveragingOptions::max_rotation_error_deg)
      .def_readwrite("tracking_huber_scale",
                     &VideoRotationAveragingOptions::video_tracking_huber_scale)
      .def_readwrite("loop_closure_cauchy_scale",
                     &VideoRotationAveragingOptions::video_lc_cauchy_scale)
      .def_readwrite("num_threads", &VideoRotationAveragingOptions::num_threads)
      .def_readwrite("max_num_iterations",
                     &VideoRotationAveragingOptions::max_num_iterations)
      .def("validate", &VideoRotationAveragingOptions::Validate);

  py::class_<RotationAveragingResult>(m, "RotationAveragingResult")
      .def_readonly("success", &RotationAveragingResult::success)
      .def_readonly("registered_image_ids",
                    &RotationAveragingResult::registered_image_ids);

  m.def("run_video_rotation_averaging",
        &RunVideoRotationAveraging,
        py::arg("options"),
        py::arg("image_map_order"),
        py::arg("pair_map_order"),
        py::arg("problem"));
}

}  // namespace vidmap
