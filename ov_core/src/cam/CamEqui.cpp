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

#include "CamEqui.h"

using namespace ov_core;

CamEqui::~CamEqui() {}

//
// FORK CHANGE: these definitions were moved out of CamEqui.h.
//
// Defining them in the class body makes them implicitly inline, so EVERY binary
// that instantiates CamEqui emitted its own COMDAT copy of them and of the
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

Eigen::Vector2f CamEqui::undistort_f(const Eigen::Vector2f &uv_dist) {

  // FORK CHANGE: do NOT use cv::fisheye::undistortPoints here.
  //
  // OpenCV's implementation returns silently wrong values once theta_d grows
  // large, which for a strongly-curved lens happens well inside the image.
  // Measured on config/openvins/xiao_sense (k1=0.719, fx=253.9, 800x600,
  // corner radius 478 px) against an analytic root-find, as the ANGLE between
  // the returned bearing and the true bearing:
  //
  //     radius (px)   median bearing error   fraction > 1 deg
  //       0 - 300           0.0000 deg              0.0 %
  //     300 - 350           0.0000 deg              1.1 %
  //     350 - 400         139.85   deg             96.6 %
  //     400 - 450         145.32   deg            100   %
  //     450 - 700         150.68   deg            100   %
  //
  // i.e. over the outer 26.4% of the frame the bearing is not merely noisy,
  // it is inverted. Features there can never triangulate consistently, never
  // pass the chi2 gates, and so never get promoted to SLAM features -- the
  // observable symptom being SLAM features clustering near the image centre.
  // It is NOT a convergence-budget problem: 10, 100 and 500 iterations with
  // tightened epsilon all return bit-identical garbage. It is also NOT a bad
  // calibration -- theta_d(theta) for the above intrinsics stays monotonic
  // out to theta = 80.7 deg (r = 564 px), i.e. over the whole image, so a
  // correct inverse exists everywhere on the frame.
  //
  // We therefore solve theta_d = theta*(1 + k1*theta^2 + k2*theta^4 +
  // k3*theta^6 + k4*theta^8) directly, with bisection-safeguarded Newton on
  // the bracket where the polynomial is known to be increasing. See
  // ov_core/test/test_cam_equi_undistort.cpp.

  const double fx = camera_k_OPENCV(0, 0);
  const double fy = camera_k_OPENCV(1, 1);
  const double cx = camera_k_OPENCV(0, 2);
  const double cy = camera_k_OPENCV(1, 2);
  const double k1 = camera_d_OPENCV(0);
  const double k2 = camera_d_OPENCV(1);
  const double k3 = camera_d_OPENCV(2);
  const double k4 = camera_d_OPENCV(3);

  // Normalized *distorted* coordinates; their radius IS theta_d by definition
  // of the equidistant model.
  const double xd = ((double)uv_dist(0) - cx) / fx;
  const double yd = ((double)uv_dist(1) - cy) / fy;
  const double theta_d = std::sqrt(xd * xd + yd * yd);

  // On the optical axis the direction is degenerate but the answer is exact.
  if (theta_d < 1e-9) {
    return Eigen::Vector2f((float)xd, (float)yd);
  }

  auto poly = [&](double t) { return t * (1.0 + t * t * (k1 + t * t * (k2 + t * t * (k3 + t * t * k4)))); };
  auto dpoly = [&](double t) {
    const double t2 = t * t;
    return 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
  };

  // Upper end of the bracket: walk out to just under 90 deg and stop at the
  // first turning point, since the model is only invertible while increasing.
  // Capping below pi/2 also keeps the tan() below finite.
  const double theta_cap = 89.0 * M_PI / 180.0;
  double theta_hi = theta_cap;
  {
    const int kSteps = 256;
    for (int i = 1; i <= kSteps; i++) {
      const double t = theta_cap * (double)i / (double)kSteps;
      if (dpoly(t) <= 0.0) {
        theta_hi = theta_cap * (double)(i - 1) / (double)kSteps;
        break;
      }
    }
  }

  // Rays beyond what this polynomial can represent: clamp to the horizon of
  // the model rather than returning something arbitrary.
  double theta;
  const double theta_d_max = poly(theta_hi);
  if (theta_d >= theta_d_max) {
    theta = theta_hi;
  } else {
    // Bisection-safeguarded Newton (Numerical Recipes "rtsafe"). poly() is
    // increasing on [lo, hi], so the bracket always contains the root.
    //
    // The safeguard must test PROGRESS, not merely containment: from a poor
    // starting point Newton will happily oscillate between two points that
    // are both inside the bracket and never converge. So we bisect whenever
    // the Newton step leaves the bracket, or fails to at least halve the
    // interval that the previous step spanned.
    double lo = 0.0, hi = theta_hi;
    double dx_old = hi - lo;
    double dx = dx_old;
    theta = 0.5 * (lo + hi);
    double f = poly(theta) - theta_d;
    double d = dpoly(theta);
    for (int i = 0; i < 100; i++) {
      const double newton = theta - f / d;
      if (d <= 1e-12 || !(newton > lo && newton < hi) || std::abs(2.0 * f) > std::abs(dx_old * d)) {
        dx_old = dx;
        dx = 0.5 * (hi - lo);
        theta = lo + dx;
      } else {
        dx_old = dx;
        dx = theta - newton;
        theta = newton;
      }
      if (std::abs(dx) < 1e-14)
        break;
      f = poly(theta) - theta_d;
      d = dpoly(theta);
      if (f > 0.0)
        hi = theta;
      else
        lo = theta;
    }
  }

  // Normalized *undistorted* radius, then rescale the direction.
  const double scale = std::tan(theta) / theta_d;
  return Eigen::Vector2f((float)(xd * scale), (float)(yd * scale));
}

Eigen::Vector2f CamEqui::distort_f(const Eigen::Vector2f &uv_norm) {

  // Get our camera parameters
  Eigen::MatrixXd cam_d = camera_values;

  // Calculate distorted coordinates for fisheye
  double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
  double theta = std::atan(r);
  double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                   cam_d(7) * std::pow(theta, 9);

  // Handle when r is small (meaning our xy is near the camera center)
  double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
  double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

  // Calculate distorted coordinates for fisheye
  Eigen::Vector2f uv_dist;
  double x1 = uv_norm(0) * cdist;
  double y1 = uv_norm(1) * cdist;
  uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
  uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
  return uv_dist;
}

void CamEqui::compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) {

  // Get our camera parameters
  Eigen::MatrixXd cam_d = camera_values;

  // Calculate distorted coordinates for fisheye
  double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
  double theta = std::atan(r);
  double theta_d = theta + cam_d(4) * std::pow(theta, 3) + cam_d(5) * std::pow(theta, 5) + cam_d(6) * std::pow(theta, 7) +
                   cam_d(7) * std::pow(theta, 9);

  // Handle when r is small (meaning our xy is near the camera center)
  double inv_r = (r > 1e-8) ? 1.0 / r : 1.0;
  double cdist = (r > 1e-8) ? theta_d * inv_r : 1.0;

  // Jacobian of distorted pixel to "normalized" pixel
  Eigen::Matrix<double, 2, 2> duv_dxy = Eigen::Matrix<double, 2, 2>::Zero();
  duv_dxy << cam_d(0), 0, 0, cam_d(1);

  // Jacobian of "normalized" pixel to normalized pixel
  Eigen::Matrix<double, 2, 2> dxy_dxyn = Eigen::Matrix<double, 2, 2>::Zero();
  dxy_dxyn << theta_d * inv_r, 0, 0, theta_d * inv_r;

  // Jacobian of "normalized" pixel to r
  Eigen::Matrix<double, 2, 1> dxy_dr = Eigen::Matrix<double, 2, 1>::Zero();
  dxy_dr << -uv_norm(0) * theta_d * inv_r * inv_r, -uv_norm(1) * theta_d * inv_r * inv_r;

  // Jacobian of r pixel to normalized xy
  Eigen::Matrix<double, 1, 2> dr_dxyn = Eigen::Matrix<double, 1, 2>::Zero();
  dr_dxyn << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

  // Jacobian of "normalized" pixel to theta_d
  Eigen::Matrix<double, 2, 1> dxy_dthd = Eigen::Matrix<double, 2, 1>::Zero();
  dxy_dthd << uv_norm(0) * inv_r, uv_norm(1) * inv_r;

  // Jacobian of theta_d to theta
  double dthd_dth = 1 + 3 * cam_d(4) * std::pow(theta, 2) + 5 * cam_d(5) * std::pow(theta, 4) + 7 * cam_d(6) * std::pow(theta, 6) +
                    9 * cam_d(7) * std::pow(theta, 8);

  // Jacobian of theta to r
  double dth_dr = 1 / (r * r + 1);

  // Total Jacobian wrt normalized pixel coordinates
  H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
  H_dz_dzn = duv_dxy * (dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn);

  // Calculate distorted coordinates for fisheye
  double x1 = uv_norm(0) * cdist;
  double y1 = uv_norm(1) * cdist;

  // Compute the Jacobian in respect to the intrinsics
  H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
  H_dz_dzeta(0, 0) = x1;
  H_dz_dzeta(0, 2) = 1;
  H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 3);
  H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 5);
  H_dz_dzeta(0, 6) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 7);
  H_dz_dzeta(0, 7) = cam_d(0) * uv_norm(0) * inv_r * std::pow(theta, 9);
  H_dz_dzeta(1, 1) = y1;
  H_dz_dzeta(1, 3) = 1;
  H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 3);
  H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 5);
  H_dz_dzeta(1, 6) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 7);
  H_dz_dzeta(1, 7) = cam_d(1) * uv_norm(1) * inv_r * std::pow(theta, 9);
}
