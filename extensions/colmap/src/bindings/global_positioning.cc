#include "vidmap_native/global_positioning.h"

#include "bindings.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindGlobalPositioning(py::module_& m) {
  py::enum_<LossFunctionType>(m, "LossFunctionType")
      .value("TRIVIAL", LossFunctionType::kTrivial)
      .value("SOFT_L1", LossFunctionType::kSoftL1)
      .value("CAUCHY", LossFunctionType::kCauchy)
      .value("HUBER", LossFunctionType::kHuber);

  py::class_<LossConfig>(m, "LossConfig")
      .def(py::init<>())
      .def_readwrite("type", &LossConfig::type)
      .def_readwrite("scale", &LossConfig::scale)
      .def_readwrite("weight", &LossConfig::weight)
      .def("validate", &LossConfig::Validate);

  py::enum_<LinearSolverType>(m, "LinearSolverType")
      .value("DENSE_SCHUR", LinearSolverType::kDenseSchur)
      .value("SPARSE_SCHUR", LinearSolverType::kSparseSchur)
      .value("ITERATIVE_SCHUR", LinearSolverType::kIterativeSchur);

  py::enum_<PreconditionerType>(m, "PreconditionerType")
      .value("JACOBI", PreconditionerType::kJacobi)
      .value("SCHUR_JACOBI", PreconditionerType::kSchurJacobi)
      .value("CLUSTER_JACOBI", PreconditionerType::kClusterJacobi)
      .value("CLUSTER_TRIDIAGONAL", PreconditionerType::kClusterTridiagonal);

  py::class_<SolverBackendOptions>(m, "SolverBackendOptions")
      .def(py::init<>())
      .def_readwrite("linear_solver", &SolverBackendOptions::linear_solver)
      .def_readwrite("preconditioner", &SolverBackendOptions::preconditioner)
      .def_readwrite("use_cuda", &SolverBackendOptions::use_cuda)
      .def("validate", &SolverBackendOptions::Validate);

  py::enum_<MetricDepthResidualType>(m, "MetricDepthResidualType")
      .value("LINEAR", MetricDepthResidualType::kLinear)
      .value("LOG", MetricDepthResidualType::kLog)
      .value("LOG_LINEAR", MetricDepthResidualType::kLogLinear);

  py::enum_<GlobalPositioningOrdering>(m, "GlobalPositioningOrdering")
      .value("GROUPED", GlobalPositioningOrdering::kGrouped)
      .value("SINGLETON", GlobalPositioningOrdering::kSingleton);

  py::enum_<GlobalPositioningCenterMode>(m, "GlobalPositioningCenterMode")
      .value("FRAME", GlobalPositioningCenterMode::kFrame)
      .value("IMAGE", GlobalPositioningCenterMode::kImage);

  py::class_<TemporalAccelerationPrior>(m, "TemporalAccelerationPrior")
      .def(py::init<>())
      .def_readwrite("prev_image_id", &TemporalAccelerationPrior::prev_image_id)
      .def_readwrite("image_id", &TemporalAccelerationPrior::image_id)
      .def_readwrite("next_image_id", &TemporalAccelerationPrior::next_image_id)
      .def_readwrite("dt_prev", &TemporalAccelerationPrior::dt_prev)
      .def_readwrite("dt_next", &TemporalAccelerationPrior::dt_next)
      .def_readwrite("sqrt_observation_count",
                     &TemporalAccelerationPrior::sqrt_observation_count);

  py::class_<GlobalPositionerOptions>(m, "GlobalPositioningOptions")
      .def(py::init<>())
      .def_readwrite("generate_random_positions",
                     &GlobalPositionerOptions::generate_random_positions)
      .def_readwrite("generate_random_points",
                     &GlobalPositionerOptions::generate_random_points)
      .def_readwrite("generate_scales",
                     &GlobalPositionerOptions::generate_scales)
      .def_readwrite("initialize_warm_start_scales",
                     &GlobalPositionerOptions::initialize_warm_start_scales)
      .def_readwrite("optimize_positions",
                     &GlobalPositionerOptions::optimize_positions)
      .def_readwrite("optimize_points",
                     &GlobalPositionerOptions::optimize_points)
      .def_readwrite("optimize_scales",
                     &GlobalPositionerOptions::optimize_scales)
      .def_readwrite("sequential_support_warmup_rounds",
                     &GlobalPositionerOptions::sequential_support_warmup_rounds)
      .def_readwrite(
          "sequential_support_observations_per_track",
          &GlobalPositionerOptions::sequential_support_observations_per_track)
      .def_readwrite("sequential_support_loss",
                     &GlobalPositionerOptions::sequential_support_loss)
      .def_readwrite(
          "sequential_support_image_timeline",
          &GlobalPositionerOptions::sequential_support_image_timeline)
      .def_readwrite("min_num_views_per_track",
                     &GlobalPositionerOptions::min_num_view_per_track)
      .def_readwrite("random_seed", &GlobalPositionerOptions::random_seed)
      .def_readwrite("random_init_scale",
                     &GlobalPositionerOptions::random_init_scale)
      .def_readwrite("loss", &GlobalPositionerOptions::loss)
      .def_readwrite(
          "apply_uncalibrated_loss_downweight",
          &GlobalPositionerOptions::apply_uncalibrated_loss_downweight)
      .def_readwrite("use_loop_closure_observations",
                     &GlobalPositionerOptions::use_lc_observations)
      .def_readwrite("use_initial_positions",
                     &GlobalPositionerOptions::use_init)
      .def_readwrite("use_parameter_block_ordering",
                     &GlobalPositionerOptions::use_parameter_block_ordering)
      .def_readwrite("parameter_ordering",
                     &GlobalPositionerOptions::parameter_ordering)
      .def_readwrite("center_mode", &GlobalPositionerOptions::center_mode)
      .def_readwrite("use_metric_depth_constraint",
                     &GlobalPositionerOptions::use_metric_depth_constraint)
      .def_readwrite(
          "use_log_depth_map_scales",
          &GlobalPositionerOptions::use_log_scale_for_depth_map_scales)
      .def_readwrite("metric_depth_residual_type",
                     &GlobalPositionerOptions::metric_depth_residual_type)
      .def_readwrite("zero_residual_behind_camera",
                     &GlobalPositionerOptions::zero_residual_behind)
      .def_readwrite("log_linear_threshold",
                     &GlobalPositionerOptions::log_linear_threshold)
      .def_readwrite("scale_prior_stddev",
                     &GlobalPositionerOptions::scale_prior_stddev)
      .def_readwrite("filter_depth_outliers",
                     &GlobalPositionerOptions::filter_depth_outliers)
      .def_readwrite("initial_depth_map_scales",
                     &GlobalPositionerOptions::initial_dmap_scales)
      .def_readwrite("initial_frame_centers",
                     &GlobalPositionerOptions::initial_frame_centers)
      .def_readwrite("use_temporal_acceleration_prior",
                     &GlobalPositionerOptions::use_temporal_acceleration_prior)
      .def_readwrite("temporal_acceleration_priors",
                     &GlobalPositionerOptions::temporal_acceleration_priors)
      .def_readwrite(
          "temporal_acceleration_prior_stddev",
          &GlobalPositionerOptions::temporal_acceleration_prior_stddev)
      .def_readwrite(
          "temporal_acceleration_prior_weight",
          &GlobalPositionerOptions::temporal_acceleration_prior_weight)
      .def_readwrite(
          "temporal_acceleration_prior_loss_dead_zone",
          &GlobalPositionerOptions::temporal_acceleration_prior_loss_dead_zone)
      .def_readwrite("temporal_acceleration_prior_loss_huber_width",
                     &GlobalPositionerOptions::
                         temporal_acceleration_prior_loss_huber_width)
      .def_readwrite("loss_normal_geometry",
                     &GlobalPositionerOptions::loss_normal_geometry)
      .def_readwrite("loss_normal_depth",
                     &GlobalPositionerOptions::loss_normal_depth)
      .def_readwrite("loss_loop_closure_geometry",
                     &GlobalPositionerOptions::loss_lc_geometry)
      .def_readwrite("loss_loop_closure_depth",
                     &GlobalPositionerOptions::loss_lc_depth)
      .def_readwrite("loss_normal_depth_outlier",
                     &GlobalPositionerOptions::loss_normal_depth_outlier)
      .def_readwrite("loss_scale_prior",
                     &GlobalPositionerOptions::loss_scale_prior)
      .def_readwrite("num_threads", &GlobalPositionerOptions::num_threads)
      .def_readwrite("max_num_iterations",
                     &GlobalPositionerOptions::max_num_iterations)
      .def_readwrite("function_tolerance",
                     &GlobalPositionerOptions::function_tolerance)
      .def_readwrite("gradient_tolerance",
                     &GlobalPositionerOptions::gradient_tolerance)
      .def_readwrite("parameter_tolerance",
                     &GlobalPositionerOptions::parameter_tolerance)
      .def_readwrite("solver_backend", &GlobalPositionerOptions::solver_backend)
      .def_readwrite("playback", &GlobalPositionerOptions::playback)
      .def("validate", &GlobalPositionerOptions::Validate);

  py::class_<GlobalPositioningDiagnostics>(m, "GlobalPositioningDiagnostics")
      .def_readonly("num_bata_residuals",
                    &GlobalPositioningDiagnostics::num_bata_residuals)
      .def_readonly("num_metric_depth_residuals",
                    &GlobalPositioningDiagnostics::num_metric_depth_residuals)
      .def_readonly("num_scale_prior_residuals",
                    &GlobalPositioningDiagnostics::num_scale_prior_residuals)
      .def_readonly(
          "num_temporal_acceleration_residuals",
          &GlobalPositioningDiagnostics::num_temporal_acceleration_residuals)
      .def_readonly(
          "num_regular_observations_used",
          &GlobalPositioningDiagnostics::num_regular_observations_used)
      .def_readonly(
          "num_loop_closure_observations_used",
          &GlobalPositioningDiagnostics::num_loop_closure_observations_used)
      .def_readonly("num_bata_scales",
                    &GlobalPositioningDiagnostics::num_bata_scales)
      .def_readonly("num_depth_map_scales",
                    &GlobalPositioningDiagnostics::num_depth_map_scales)
      .def_readonly("num_camera_centers",
                    &GlobalPositioningDiagnostics::num_camera_centers)
      .def_readonly("num_point3D_parameters",
                    &GlobalPositioningDiagnostics::num_point3D_parameters)
      .def_readonly("num_residual_blocks",
                    &GlobalPositioningDiagnostics::num_residual_blocks)
      .def_readonly("num_parameter_blocks",
                    &GlobalPositioningDiagnostics::num_parameter_blocks)
      .def_readonly("num_parameters",
                    &GlobalPositioningDiagnostics::num_parameters)
      .def_readonly("num_iterations",
                    &GlobalPositioningDiagnostics::num_iterations)
      .def_readonly("termination_type",
                    &GlobalPositioningDiagnostics::termination_type)
      .def_readonly("initial_cost", &GlobalPositioningDiagnostics::initial_cost)
      .def_readonly("final_cost", &GlobalPositioningDiagnostics::final_cost);

  py::class_<GlobalPositioningResult>(m, "GlobalPositioningResult")
      .def_readonly("success", &GlobalPositioningResult::success)
      .def_readonly("depth_map_scales",
                    &GlobalPositioningResult::depth_map_scales)
      .def_readonly("initial_frame_centers",
                    &GlobalPositioningResult::initial_frame_centers)
      .def_readonly("initial_point3D_xyz",
                    &GlobalPositioningResult::initial_point3D_xyz)
      .def_readonly("initial_bata_scales",
                    &GlobalPositioningResult::initial_bata_scales)
      .def_readonly("final_bata_scales",
                    &GlobalPositioningResult::final_bata_scales)
      .def_readonly("diagnostics", &GlobalPositioningResult::diagnostics);

  m.def(
      "run_global_positioning",
      [](const GlobalPositionerOptions& options, MappingProblem* problem) {
        py::gil_scoped_release release;
        return RunGlobalPositioning(options, problem);
      },
      py::arg("options"),
      py::arg("problem"));
}

}  // namespace vidmap
