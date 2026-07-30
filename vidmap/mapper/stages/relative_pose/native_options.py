"""Translate relative-pose configuration into native options."""

from vidmap.mapper.native.extension import native
from vidmap.mapper.options.view_graph import InlierThresholdOptions


def build_inlier_threshold_options(options: InlierThresholdOptions):
    native_options = native.InlierThresholdOptions()
    native_options.max_epipolar_error_essential = options.max_epipolar_error_E
    native_options.max_epipolar_error_fundamental = options.max_epipolar_error_F
    native_options.max_epipolar_error_homography = options.max_epipolar_error_H
    native_options.min_angle_from_epipole_deg = options.min_angle_from_epipole
    native_options.validate()
    return native_options
