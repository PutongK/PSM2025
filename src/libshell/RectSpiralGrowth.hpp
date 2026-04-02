//
//  RectSpiralGrowth.hpp
//  RectSpiralGrowth for English Wheel process simulation
//
//  Created by Putong Kang on 3/21/26.
//  Copyright © 2026 Putong Kang. All rights reserved.
//

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace rect_spiral {

// ------------------------
// Config / CLI parameters
// ------------------------
enum class StartMode {
    LeftBottom_Up
};

struct Params {
    double Lx0_mm = 140.0;   // outer centerline envelope in x, mm
    double Ly0_mm = 140.0;   // outer centerline envelope in y, mm
    double gap_mm = 8.0;     // inward spacing between turns, mm
    double w_mm   = 8.0;     // active band width, mm
    double min_leg_mm = -1.0;   // if <= 0, fall back to w_mm

    bool last_wins = true;
    StartMode start_mode = StartMode::LeftBottom_Up;

    // If true: set all growth to zero first, then write only where pattern covers
    bool zero_outside = true;

    // Offset of spiral center relative to panel center, in mm
    double offset_dx_mm = 0.0;
    double offset_dy_mm = 0.0;
};

// ------------------------
// Small utilities
// ------------------------
inline double normalize_angle_pi(double a) {
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a >   M_PI) a -= 2.0 * M_PI;
    if (a < 0.0) a += M_PI;
    return a;
}

inline double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

inline double clamp_val(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// Distance from point p to segment [a,b] in 2D
inline double dist_point_segment_2d(const Eigen::Vector2d& p,
                                    const Eigen::Vector2d& a,
                                    const Eigen::Vector2d& b)
{
    const Eigen::Vector2d ab = b - a;
    const double ab2 = ab.squaredNorm();
    if (ab2 <= 1e-30) return (p - a).norm();

    const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / ab2));
    const Eigen::Vector2d proj = a + t * ab;
    return (p - proj).norm();
}

// ------------------------
// Segment definition
// ------------------------
struct SpiralSegment {
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    double width;       // meters
    double gtop;
    double gbot;
    double ortho;
    double angle_rad;   // orientation in [0, pi)
    int seg_id = -1;
};

struct SpiralBounds {
    double xmin_center_m = 0.0;
    double xmax_center_m = 0.0;
    double ymin_center_m = 0.0;
    double ymax_center_m = 0.0;

    double xmin_band_m = 0.0;
    double xmax_band_m = 0.0;
    double ymin_band_m = 0.0;
    double ymax_band_m = 0.0;
};

inline SpiralBounds getSpiralBounds(const Params& rs)
{
    SpiralBounds b;

    const double Lx0 = rs.Lx0_mm * 1e-3;
    const double Ly0 = rs.Ly0_mm * 1e-3;
    const double dx  = rs.offset_dx_mm * 1e-3;
    const double dy  = rs.offset_dy_mm * 1e-3;
    const double hw  = 0.5 * rs.w_mm * 1e-3;

    b.xmin_center_m = dx - 0.5 * Lx0;
    b.xmax_center_m = dx + 0.5 * Lx0;
    b.ymin_center_m = dy - 0.5 * Ly0;
    b.ymax_center_m = dy + 0.5 * Ly0;

    b.xmin_band_m = b.xmin_center_m - hw;
    b.xmax_band_m = b.xmax_center_m + hw;
    b.ymin_band_m = b.ymin_center_m - hw;
    b.ymax_band_m = b.ymax_center_m + hw;

    return b;
}

// Build rectangular spiral segments centered around the mesh-centered frame
inline std::vector<SpiralSegment> buildRectSpiralSegments(const Params& rs,
                                                          double gtop,
                                                          double gbot,
                                                          double ortho)
{
    if (rs.Lx0_mm <= 0.0) throw std::runtime_error("rect_spiral: spiral_Lx0_mm must be > 0");
    if (rs.Ly0_mm <= 0.0) throw std::runtime_error("rect_spiral: spiral_Ly0_mm must be > 0");
    if (rs.w_mm   <= 0.0) throw std::runtime_error("rect_spiral: spiral_w_mm must be > 0");
    if (rs.gap_mm <= 0.0) throw std::runtime_error("rect_spiral: spiral_gap_mm must be > 0");
    if (rs.gap_mm < rs.w_mm) {
        throw std::runtime_error("rect_spiral: spiral_gap_mm must be >= spiral_w_mm to avoid overlap between adjacent turns");
    }

    const double Lx0 = rs.Lx0_mm * 1e-3;
    const double Ly0 = rs.Ly0_mm * 1e-3;
    const double gap = rs.gap_mm * 1e-3;
    const double w   = rs.w_mm   * 1e-3;

    const double min_leg = (rs.min_leg_mm > 0.0 ? rs.min_leg_mm : rs.w_mm) * 1e-3;

    const double dx  = rs.offset_dx_mm * 1e-3;
    const double dy  = rs.offset_dy_mm * 1e-3;

    double x_left  = dx - 0.5 * Lx0;
    double x_right = dx + 0.5 * Lx0;
    double y_bot   = dy - 0.5 * Ly0;
    double y_top   = dy + 0.5 * Ly0;

    std::vector<SpiralSegment> segs;
    int seg_id = 0;

    auto make_seg = [&](const Eigen::Vector2d& a2, const Eigen::Vector2d& b2) {
        SpiralSegment s;
        s.a = a2;
        s.b = b2;
        s.width = w;
        s.gtop  = gtop;
        s.gbot  = gbot;
        s.ortho = ortho;
        s.seg_id = seg_id++;

        const Eigen::Vector2d d = b2 - a2;
        const double ang = std::atan2(d.y(), d.x());
        s.angle_rad = normalize_angle_pi(ang);
        return s;
    };

    while (true) {
        // 1) up along left side
        {
            const double leg = y_top - y_bot;
            if (leg >= min_leg) {
                segs.push_back(make_seg(Eigen::Vector2d(x_left, y_bot),
                                        Eigen::Vector2d(x_left, y_top)));
            } else {
                break;
            }
        }

        // 2) right along top
        {
            const double leg = x_right - x_left;
            if (leg >= min_leg) {
                segs.push_back(make_seg(Eigen::Vector2d(x_left, y_top),
                                        Eigen::Vector2d(x_right, y_top)));
            } else {
                break;
            }
        }

        // 3) down along right side
        {
            const double leg = y_top - (y_bot + gap);
            if (leg >= min_leg) {
                segs.push_back(make_seg(Eigen::Vector2d(x_right, y_top),
                                        Eigen::Vector2d(x_right, y_bot + gap)));
            } else {
                break;
            }
        }

        // 4) left along inner bottom
        {
            const double leg = x_right - (x_left + gap);
            if (leg >= min_leg) {
                segs.push_back(make_seg(Eigen::Vector2d(x_right, y_bot + gap),
                                        Eigen::Vector2d(x_left + gap, y_bot + gap)));
            } else {
                break;
            }
        }

        x_left  += gap;
        x_right -= gap;
        y_bot   += gap;
        y_top   -= gap;

        if (x_right < x_left || y_top < y_bot) break;
    }

    return segs;
}

// ------------------------
// Main application function
// ------------------------
template <typename MeshType>
inline void apply(const MeshType& mesh,
                  const Params& rs,
                  double gtop,
                  double gbot,
                  double ortho,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXd& orthoCoeffFaces,
                  Eigen::VectorXi* passCountFaces = nullptr)
{
    const int nFaces = mesh.getNumberOfFaces();
    if (growthRates_t.size() != nFaces || growthRates_b.size() != nFaces)
        throw std::runtime_error("rect_spiral::apply: growthRates size mismatch");
    if (growthAngles.size() != nFaces || orthoCoeffFaces.size() != nFaces)
        throw std::runtime_error("rect_spiral::apply: growthAngles/orthoCoeffFaces size mismatch");
    if (passCountFaces) {
        if (passCountFaces->size() != nFaces)
            throw std::runtime_error("rect_spiral::apply: passCountFaces size mismatch");
    }

    // Optional zeroing outside the pattern
    if (rs.zero_outside) {
        growthRates_t.setZero();
        growthRates_b.setZero();
        orthoCoeffFaces.setZero();
        // leave growthAngles as-is outside, consistent with zigzag
    }

    // Get vertices + faces
    const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
    const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();

    // Center mesh XY so pattern is defined relative to plate center
    const double xmin = V.col(0).minCoeff();
    const double xmax = V.col(0).maxCoeff();
    const double ymin = V.col(1).minCoeff();
    const double ymax = V.col(1).maxCoeff();
    const Eigen::Vector2d center(0.5 * (xmin + xmax), 0.5 * (ymin + ymax));

    const double panel_lx = xmax - xmin;
    const double panel_ly = ymax - ymin;

    // Boundary validation: spiral band footprint must stay inside panel
    const SpiralBounds b = getSpiralBounds(rs);
    const double panel_xmin = -0.5 * panel_lx;
    const double panel_xmax = +0.5 * panel_lx;
    const double panel_ymin = -0.5 * panel_ly;
    const double panel_ymax = +0.5 * panel_ly;

    if (b.xmin_band_m < panel_xmin || b.xmax_band_m > panel_xmax ||
        b.ymin_band_m < panel_ymin || b.ymax_band_m > panel_ymax) {
        std::ostringstream oss;
        oss << "rect_spiral: spiral band footprint exceeds panel bounds\n"
            << "  Spiral band x = [" << b.xmin_band_m << ", " << b.xmax_band_m << "] m\n"
            << "  Spiral band y = [" << b.ymin_band_m << ", " << b.ymax_band_m << "] m\n"
            << "  Panel      x = [" << panel_xmin << ", " << panel_xmax << "] m\n"
            << "  Panel      y = [" << panel_ymin << ", " << panel_ymax << "] m";
        throw std::runtime_error(oss.str());
    }

    // Build spiral segments
    const auto segs = buildRectSpiralSegments(rs, gtop, gbot, ortho);
    const double half_w = 0.5 * (rs.w_mm * 1e-3);

    // For each segment in order: later segment wins by overwrite
    for (int k = 0; k < (int)segs.size(); ++k) {
        const auto& s = segs[k];

        for (int i = 0; i < nFaces; ++i) {
            const int i0 = F(i,0);
            const int i1 = F(i,1);
            const int i2 = F(i,2);

            const Eigen::Vector2d c(
                (V(i0,0) + V(i1,0) + V(i2,0)) / 3.0,
                (V(i0,1) + V(i1,1) + V(i2,1)) / 3.0
            );

            const Eigen::Vector2d cc = c - center; // centered panel frame

            const double d = dist_point_segment_2d(cc, s.a, s.b);
            if (d <= half_w) {
                if (passCountFaces) (*passCountFaces)(i) += 1;

                growthRates_t(i)    = s.gtop;
                growthRates_b(i)    = s.gbot;
                growthAngles(i)     = s.angle_rad;
                orthoCoeffFaces(i)  = s.ortho;
            }
        }
    }

    // Debug print
    std::cout << "[rect_spiral] segments=" << segs.size()
              << " gap_mm=" << rs.gap_mm
              << " w_mm=" << rs.w_mm
              << " offset_dx_mm=" << rs.offset_dx_mm
              << " offset_dy_mm=" << rs.offset_dy_mm
              << "\n";

    std::cout << "[rect_spiral] band x=[" << b.xmin_band_m << "," << b.xmax_band_m
              << "] y=[" << b.ymin_band_m << "," << b.ymax_band_m << "] m\n";
}

} // namespace rect_spiral