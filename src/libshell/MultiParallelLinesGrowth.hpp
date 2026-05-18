#pragma once

#include "PatchGrowthUtils.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace multi_parallel_lines {

struct PatchParams {
    std::string name = "patch";
    bool enabled = true;
    patch_growth::Quad corners_m;

    double growth_angle_deg = 0.0;
    double width_mm = 1.0;
    double spacing_mm = 2.0;
    double offset_mm = 0.0;

    double exx_top = 0.0;
    double exx_bot = 0.0;
    double eyy_top = 0.0;
    double eyy_bot = 0.0;
};

enum class CombineMode {
    Sum,
    Override,
    ErrorOnOverlap
};

inline CombineMode parseCombineMode(const std::string& s)
{
    if (s == "sum") return CombineMode::Sum;
    if (s == "override") return CombineMode::Override;
    if (s == "error_on_overlap") return CombineMode::ErrorOnOverlap;
    throw std::runtime_error("multi_parallel_lines: combine_mode must be 'sum', 'override', or 'error_on_overlap'");
}

inline patch_growth::Quad readCornersMm(const nlohmann::json& jp, const std::string& name)
{
    if (!jp.contains("corners_mm")) {
        throw std::runtime_error("multi_parallel_lines: patch '" + name + "' missing corners_mm");
    }
    const auto& jc = jp.at("corners_mm");
    if (!jc.is_array() || jc.size() != 4) {
        throw std::runtime_error("multi_parallel_lines: patch '" + name + "' corners_mm must contain four [x,y] points");
    }

    patch_growth::Quad q;
    for (int i = 0; i < 4; ++i) {
        if (!jc[i].is_array() || jc[i].size() != 2) {
            throw std::runtime_error("multi_parallel_lines: each corner must be [x_mm, y_mm]");
        }
        q[i] = patch_growth::Vec2(jc[i][0].get<double>() * 1e-3,
                                  jc[i][1].get<double>() * 1e-3);
    }
    q = patch_growth::ensureCounterClockwise(q);
    if (!patch_growth::isConvexQuad(q)) {
        throw std::runtime_error("multi_parallel_lines: patch '" + name + "' is not a valid convex quadrilateral");
    }
    return q;
}

inline std::vector<PatchParams> loadPatches(const std::string& filename,
                                            CombineMode& combineMode)
{
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("multi_parallel_lines: cannot open patch file: " + filename);
    }

    nlohmann::json j;
    in >> j;

    const std::string units = j.value("units", "mm");
    if (units != "mm") {
        throw std::runtime_error("multi_parallel_lines: only units='mm' is supported in patch JSON");
    }

    combineMode = parseCombineMode(j.value("combine_mode", "sum"));

    if (!j.contains("patches") || !j.at("patches").is_array()) {
        throw std::runtime_error("multi_parallel_lines: patch JSON must contain a patches array");
    }

    std::vector<PatchParams> patches;
    int idx = 0;
    for (const auto& jp : j.at("patches")) {
        PatchParams p;
        p.name = jp.value("name", "patch_" + std::to_string(idx));
        p.enabled = jp.value("enabled", true);
        p.corners_m = readCornersMm(jp, p.name);

        p.growth_angle_deg = jp.value("growth_angle_deg", 0.0);
        p.width_mm   = jp.value("parline_width_mm",   jp.value("width_mm",   1.0));
        p.spacing_mm = jp.value("parline_spacing_mm", jp.value("spacing_mm", 2.0));
        p.offset_mm  = jp.value("parline_offset_mm",  jp.value("offset_mm",  0.0));

        p.exx_top = jp.value("exx_top", jp.value("parline_exx_top", 0.0));
        p.exx_bot = jp.value("exx_bot", jp.value("parline_exx_bot", 0.0));
        p.eyy_top = jp.value("eyy_top", jp.value("parline_eyy_top", 0.0));
        p.eyy_bot = jp.value("eyy_bot", jp.value("parline_eyy_bot", 0.0));

        if (p.spacing_mm <= 0.0) throw std::runtime_error("multi_parallel_lines: patch '" + p.name + "' spacing must be > 0");
        if (p.width_mm <= 0.0)   throw std::runtime_error("multi_parallel_lines: patch '" + p.name + "' width must be > 0");
        if (p.width_mm > p.spacing_mm) throw std::runtime_error("multi_parallel_lines: patch '" + p.name + "' width must be <= spacing");

        patches.push_back(p);
        ++idx;
    }

    if (patches.empty()) {
        throw std::runtime_error("multi_parallel_lines: no patches were found in patch JSON");
    }

    return patches;
}

inline bool pointInLineBand(double x, double y, const PatchParams& p)
{
    const double theta = patch_growth::deg2rad(p.growth_angle_deg);
    const double nx_dir = -std::sin(theta);
    const double ny_dir =  std::cos(theta);
    const double spacing = p.spacing_mm * 1e-3;
    const double width   = p.width_mm   * 1e-3;
    const double offset  = p.offset_mm  * 1e-3;

    const double s = x * nx_dir + y * ny_dir - offset;
    double s_mod = std::fmod(s, spacing);
    if (s_mod < 0.0) s_mod += spacing;
    if (s_mod > 0.5 * spacing) s_mod -= spacing;

    return std::abs(s_mod) <= 0.5 * width;
}

template <typename MeshType>
inline void apply(const MeshType& mesh,
                  const std::string& patch_file,
                  Eigen::VectorXd& growthRates_1_t,
                  Eigen::VectorXd& growthRates_1_b,
                  Eigen::VectorXd& growthRates_2_t,
                  Eigen::VectorXd& growthRates_2_b,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXi* patchHitCount = nullptr)
{
    CombineMode mode;
    const auto patches = loadPatches(patch_file, mode);

    const int nFaces = mesh.getNumberOfFaces();
    if (growthRates_1_t.size() != nFaces || growthRates_1_b.size() != nFaces ||
        growthRates_2_t.size() != nFaces || growthRates_2_b.size() != nFaces ||
        growthRates_t.size() != nFaces || growthRates_b.size() != nFaces ||
        growthAngles.size() != nFaces) {
        throw std::runtime_error("multi_parallel_lines::apply: growth vector size mismatch");
    }
    if (patchHitCount && patchHitCount->size() != nFaces) {
        throw std::runtime_error("multi_parallel_lines::apply: patchHitCount size mismatch");
    }

    const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
    const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();

    growthRates_1_t.setZero();
    growthRates_1_b.setZero();
    growthRates_2_t.setZero();
    growthRates_2_b.setZero();
    growthRates_t.setZero();
    growthRates_b.setZero();
    growthAngles.setZero();
    if (patchHitCount) patchHitCount->setZero();

    Eigen::VectorXi activeCount(nFaces);
    activeCount.setZero();

    int enabledCount = 0;
    for (const auto& p : patches) {
        if (!p.enabled) continue;
        ++enabledCount;

        const double theta = patch_growth::deg2rad(p.growth_angle_deg);
        int patchFaces = 0;
        int lineFaces = 0;

        for (int i = 0; i < nFaces; ++i) {
            const int v0 = F(i, 0);
            const int v1 = F(i, 1);
            const int v2 = F(i, 2);

            const double xc = (V(v0,0) + V(v1,0) + V(v2,0)) / 3.0;
            const double yc = (V(v0,1) + V(v1,1) + V(v2,1)) / 3.0;
            const patch_growth::Vec2 c(xc, yc);

            if (!patch_growth::pointInConvexQuad(c, p.corners_m)) continue;
            ++patchFaces;

            if (!pointInLineBand(xc, yc, p)) continue;
            ++lineFaces;

            if (activeCount(i) > 0) {
                if (mode == CombineMode::ErrorOnOverlap) {
                    std::ostringstream oss;
                    oss << "multi_parallel_lines: overlap detected at face " << i
                        << " while applying patch '" << p.name << "'.";
                    throw std::runtime_error(oss.str());
                }

                if (mode == CombineMode::Sum && std::abs(growthAngles(i) - theta) > 1e-10) {
                    std::ostringstream oss;
                    oss << "multi_parallel_lines: sum mode found overlapping patches with different angles at face " << i
                        << ". This is ambiguous for one orthotropic frame per face. Avoid overlap, use same angle, or use override/error_on_overlap.";
                    throw std::runtime_error(oss.str());
                }
            }

            if (mode == CombineMode::Override || activeCount(i) == 0) {
                if (mode == CombineMode::Override) {
                    growthRates_1_t(i) = 0.0;
                    growthRates_1_b(i) = 0.0;
                    growthRates_2_t(i) = 0.0;
                    growthRates_2_b(i) = 0.0;
                }
                growthAngles(i) = theta;
            }

            growthRates_1_t(i) += p.exx_top;
            growthRates_1_b(i) += p.exx_bot;
            growthRates_2_t(i) += p.eyy_top;
            growthRates_2_b(i) += p.eyy_bot;

            growthRates_t(i) = 0.5 * (growthRates_1_t(i) + growthRates_2_t(i));
            growthRates_b(i) = 0.5 * (growthRates_1_b(i) + growthRates_2_b(i));

            activeCount(i) += 1;
            if (patchHitCount) (*patchHitCount)(i) += 1;
        }

        std::cout << "[multi_parallel_lines] patch='" << p.name
                  << "' angle_deg=" << p.growth_angle_deg
                  << " width_mm=" << p.width_mm
                  << " spacing_mm=" << p.spacing_mm
                  << " patch_faces=" << patchFaces
                  << " line_faces=" << lineFaces
                  << " corners=" << patch_growth::quadToStringMm(p.corners_m)
                  << "\n";
    }

    if (enabledCount == 0) {
        throw std::runtime_error("multi_parallel_lines: all patches are disabled");
    }

    const int activeFaces = (activeCount.array() > 0).count();
    std::cout << "[multi_parallel_lines] enabled_patches=" << enabledCount
              << " active_faces=" << activeFaces
              << " combine_mode="
              << (mode == CombineMode::Sum ? "sum" : mode == CombineMode::Override ? "override" : "error_on_overlap")
              << "\n";
}

} // namespace multi_parallel_lines
