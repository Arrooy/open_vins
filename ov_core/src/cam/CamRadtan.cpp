/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "CamRadtan.h"

using namespace ov_core;

CamRadtan::~CamRadtan() {}

//
// FORK CHANGE: these definitions were moved out of CamRadtan.h.
//
// Defining them in the class body makes them implicitly inline, so EVERY binary
// that instantiates CamRadtan emitted its own COMDAT copy of them and of the
// vtable -- vo_runner, libov_msckf_lib.so and libov_init_lib.so each had one,
// while libov_core_lib.so (which contains the tracker that calls them) had
// none. All copies have default visibility, so the dynamic loader bound every
// call to whichever came first in load order: the executable. Rebuilding the
// library therefore did not change behaviour, and an app built against a
// different version of the header than the prebuilt .so was an ODR violation
// resolved silently by link order.
//
// Out of line there is exactly one definition, here, and undistort_f becomes
// the class's key function so the vtable is emitted here too. The methods are
// virtual and always reached through a CamBase pointer, so nothing was being
// inlined at the call sites anyway.
//

Eigen::Vector2f CamRadtan::undistort_f(const Eigen::Vector2f &uv_dist) {

  // Determine what camera parameters we should use
  cv::Matx33d camK = camera_k_OPENCV;
  cv::Vec4d camD = camera_d_OPENCV;

  // Convert to opencv format
  cv::Mat mat(1, 2, CV_32F);
  mat.at<float>(0, 0) = uv_dist(0);
  mat.at<float>(0, 1) = uv_dist(1);
  mat = mat.reshape(2); // Nx1, 2-channel

  // Undistort it!
  cv::undistortPoints(mat, mat, camK, camD);

  // Construct our return vector
  Eigen::Vector2f pt_out;
  mat = mat.reshape(1); // Nx2, 1-channel
  pt_out(0) = mat.at<float>(0, 0);
  pt_out(1) = mat.at<float>(0, 1);
  return pt_out;
}

Eigen::Vector2f CamRadtan::distort_f(const Eigen::Vector2f &uv_norm) {

  // Get our camera parameters
  Eigen::MatrixXd cam_d = camera_values;

  // Calculate distorted coordinates for radial
  double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
  double r_2 = r * r;
  double r_4 = r_2 * r_2;
  double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
              cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
  double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
              2 * cam_d(7) * uv_norm(0) * uv_norm(1);

  // Return the distorted point
  Eigen::Vector2f uv_dist;
  uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
  uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
  return uv_dist;
}

void CamRadtan::compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) {

  // Get our camera parameters
  Eigen::MatrixXd cam_d = camera_values;

  // Calculate distorted coordinates for radial
  double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
  double r_2 = r * r;
  double r_4 = r_2 * r_2;

  // Jacobian of distorted pixel to normalized pixel
  H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
  double x = uv_norm(0);
  double y = uv_norm(1);
  double x_2 = uv_norm(0) * uv_norm(0);
  double y_2 = uv_norm(1) * uv_norm(1);
  double x_y = uv_norm(0) * uv_norm(1);
  H_dz_dzn(0, 0) = cam_d(0) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * x_2 + 4 * cam_d(5) * x_2 * r_2) +
                               2 * cam_d(6) * y + (2 * cam_d(7) * x + 4 * cam_d(7) * x));
  H_dz_dzn(0, 1) = cam_d(0) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
  H_dz_dzn(1, 0) = cam_d(1) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
  H_dz_dzn(1, 1) = cam_d(1) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * y_2 + 4 * cam_d(5) * y_2 * r_2) +
                               2 * cam_d(7) * x + (2 * cam_d(6) * y + 4 * cam_d(6) * y));

  // Calculate distorted coordinates for radtan
  double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
              cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
  double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
              2 * cam_d(7) * uv_norm(0) * uv_norm(1);

  // Compute the Jacobian in respect to the intrinsics
  H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
  H_dz_dzeta(0, 0) = x1;
  H_dz_dzeta(0, 2) = 1;
  H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * r_2;
  H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * r_4;
  H_dz_dzeta(0, 6) = 2 * cam_d(0) * uv_norm(0) * uv_norm(1);
  H_dz_dzeta(0, 7) = cam_d(0) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
  H_dz_dzeta(1, 1) = y1;
  H_dz_dzeta(1, 3) = 1;
  H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * r_2;
  H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * r_4;
  H_dz_dzeta(1, 6) = cam_d(1) * (r_2 + 2 * uv_norm(1) * uv_norm(1));
  H_dz_dzeta(1, 7) = 2 * cam_d(1) * uv_norm(0) * uv_norm(1);
}
