#pragma once

#include "../include/Config.h"

// General Solution namespace to for functions that involve running the problem set.

namespace solution {
/**
 * \brief Solves problem 1.a, which computes the matrix of calibration parameters M that describes
 * the mapping between 3D world points and 2D camera points.
 *
 * \param config Configuration settings as loaded from a YAML file.
 */
void runProblem1a(const Config& config);

/**
 * \brief Solves problems 1.b. and 1.c. First we compute the camera projection matrix with different
 * constraint sizes (8, 12, and 16 points), and picks the best projection by selecting the lowest
 * residual. Then, the camera center in the world frame is computed.
 *
 * \param config Configuration settings as loaded from a YAML file.
 */
void runProblem1bc(const Config& config);

/**
 * \brief Solves problem 2 (parts a, b, and c). This estimates the fundamental matrix F mapping
 * corresponding points between an image A and an image B. First, a least-squares estimate of F is
 * computed, which is then decomposed from rank 3 to rank 2. Finally, this matrix is used to compute
 * epipolar lines from the input points, and these lines are drawn on the input images.
 *
 * \param config Configuration settings as loaded from a YAML file.
 */
void runProblem2(const Config& config);

/**
 * \brief Solves the extra credit (problem 2, parts d and e). This computes fundamental matrix F by
 * first normalizing the input points, then draws the epipolar lines computed by this more accurate
 * F on the input images.
 *
 * \param config Configuration settings as loaded from a YAML file.
 */
void runExtraCredit(const Config& config);
}; // namespace solution