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

#ifndef OV_CORE_CAM_EQUI_H
#define OV_CORE_CAM_EQUI_H

#include "CamBase.h"

#include <cmath>

namespace ov_core {

/**
 * @brief Fisheye / equadistant model pinhole camera model class
 *
 * As fisheye or wide-angle lenses are widely used in practice, we here provide mathematical derivations
 * of such distortion model as in [OpenCV fisheye](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html#details).
 *
 * \f{align*}{
 * \begin{bmatrix} u \\ v \end{bmatrix}:= \mathbf{z}_k &= \mathbf h_d(\mathbf{z}_{n,k}, ~\boldsymbol\zeta)
 * = \begin{bmatrix}  f_x * x + c_x \\
 * f_y * y + c_y \end{bmatrix}\\[1em]
 * \empty
 * {\rm where}~~
 * x &= \frac{x_n}{r} * \theta_d \\
 * y &= \frac{y_n}{r} * \theta_d \\
 * \theta_d &= \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8) \\
 * \quad r^2 &= x_n^2 + y_n^2 \\
 * \theta &= atan(r)
 * \f}
 *
 * where \f$ \mathbf{z}_{n,k} = [ x_n ~ y_n ]^\top\f$ are the normalized coordinates of the 3D feature
 * and u and v are the distorted image coordinates on the image plane.
 * Clearly, the following distortion intrinsic parameters are used in the above model:
 *
 * \f{align*}{
 * \boldsymbol\zeta = \begin{bmatrix} f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4 \end{bmatrix}^\top
 * \f}
 *
 * In analogy to the previous radial distortion (see @ref ov_core::CamRadtan) case, the following Jacobian for these
 * parameters is needed for intrinsic calibration:
 * \f{align*}{
 * \frac{\partial \mathbf h_d (\cdot)}{\partial \boldsymbol\zeta} =
 * \begin{bmatrix}
 * x_n & 0  & 1 & 0 & f_x*(\frac{x_n}{r}\theta^3) & f_x*(\frac{x_n}{r}\theta^5) & f_x*(\frac{x_n}{r}\theta^7) & f_x*(\frac{x_n}{r}\theta^9)
 * \\[5pt] 0  & y_n & 0 & 1 & f_y*(\frac{y_n}{r}\theta^3) & f_y*(\frac{y_n}{r}\theta^5) & f_y*(\frac{y_n}{r}\theta^7) &
 * f_y*(\frac{y_n}{r}\theta^9) \end{bmatrix} \f}
 *
 * Similarly, with the chain rule of differentiation,
 * we can compute the following Jacobian with respect to the normalized coordinates:
 *
 * \f{align*}{
 * \frac{\partial \mathbf h_d(\cdot)}{\partial \mathbf{z}_{n,k}}
 * &=
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial x_ny_n}+
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial r}\frac{\partial r}{\partial x_ny_n}+
 * \frac{\partial uv}{\partial xy}\frac{\partial xy}{\partial \theta_d}\frac{\partial \theta_d}{\partial \theta}\frac{\partial
 * \theta}{\partial r}\frac{\partial r}{\partial x_ny_n} \\[1em] \empty
 * {\rm where}~~~~
 * \frac{\partial uv}{\partial xy} &= \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial x_ny_n} &= \begin{bmatrix} \theta_d/r & 0 \\ 0 & \theta_d/r \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial r} &= \begin{bmatrix} -\frac{x_n}{r^2}\theta_d \\ -\frac{y_n}{r^2}\theta_d \end{bmatrix} \\
 * \empty
 * \frac{\partial r}{\partial x_ny_n} &= \begin{bmatrix} \frac{x_n}{r} & \frac{y_n}{r} \end{bmatrix} \\
 * \empty
 * \frac{\partial xy}{\partial \theta_d} &= \begin{bmatrix} \frac{x_n}{r} \\ \frac{y_n}{r} \end{bmatrix} \\
 * \empty
 * \frac{\partial \theta_d}{\partial \theta} &= \begin{bmatrix} 1 + 3k_1 \theta^2 + 5k_2 \theta^4 + 7k_3 \theta^6 + 9k_4
 * \theta^8\end{bmatrix} \\ \empty \frac{\partial \theta}{\partial r} &= \begin{bmatrix} \frac{1}{r^2+1} \end{bmatrix} \f}
 *
 * To equate this to one of Kalibr's models, this is what you would use for `pinhole-equi`.
 */
class CamEqui : public CamBase {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamEqui(int width, int height) : CamBase(width, height) {}

  // FORK CHANGE: the destructor is declared here and defined in CamEqui.cpp so it
  // acts as this class's *key function*. A class whose first non-pure virtual is
  // defined inline has no key function, so every TU that sees the header emits a
  // weak COMDAT vtable with default visibility -- including the executable, which
  // the dynamic loader always searches first. Anchoring the vtable in the library
  // keeps a stale or divergent copy in the binary from silently preempting it.
  ~CamEqui();

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override;

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override;

  /**
   * @brief Computes the derivative of raw distorted to normalized coordinate.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
   */
  void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override;
};

} // namespace ov_core

#endif /* OV_CORE_CAM_EQUI_H */