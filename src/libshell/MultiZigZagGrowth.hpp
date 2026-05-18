#pragma once

#include "PatchGrowthUtils.hpp"
#include "ZigZagGrowth.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace multi_zigzag {

using json = nlohmann::json;
using patch_growth::Vec2;
using patch_growth::Quad;

inline std::string getStringOrDefault(const json& j, const std::string& key, const std::string& def)
{
    if (!j.contains(key) || j.at(key).is_null()) return def;
    if (j.at(key).is_string()) return j.at(key).get<std::string>();
    std::ostringstream oss;
    oss << "multi_zigzag: key '" << key << "' must be a string";
    throw std::runtime_error(oss.str());
}

inline double getDoubleOrDefault(const json& j, const std::string& key, double def)
{
    if (!j.contains(key) || j.at(key).is_null()) return def;
    return j.at(key).get<double>();
}

inline int getIntOrDefault(const json& j, const std::string& key, int def)
{
    if (!j.contains(key) || j.at(key).is_null()) return def;
    return j.at(key).get<int>();
}

inline bool getBoolOrDefault(const json& j, const std::string& key, bool def)
{
    if (!j.contains(key) || j.at(key).is_null()) return def;
    return j.at(key).get<bool>();
}

inline Quad readCornersMm(const json& p)
{
    if (!p.contains("corners_mm")) {
        throw std::runtime_error("multi_zigzag: each patch must provide corners_mm");
    }
    const auto& c = p.at("corners_mm");
    if (!c.is_array() || c.size() != 4) {
        throw std::runtime_error("multi_zigzag: corners_mm must be an array of four [x,y] points");
    }

    Quad q;
    for (int i = 0; i < 4; ++i) {
        if (!c.at(i).is_array() || c.at(i).size() != 2) {
            throw std::runtime_error("multi_zigzag: each corners_mm entry must be [x,y]");
        }
        q[i] = Vec2(c.at(i).at(0).get<double>() * 1e-3,
                    c.at(i).at(1).get<double>() * 1e-3);
    }
    q = patch_growth::ensureCounterClockwise(q);
    return q;
}

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
    throw std::runtime_error("multi_zigzag: combine_mode must be 'sum', 'override', or 'error_on_overlap'");
}

inline std::vector<double> readNumberListOrUniform(const json& p,
                                                   const std::string& key,
                                                   int N,
                                                   double uniform_value)
{
    std::vector<double> out(N, uniform_value);
    if (!p.contains(key) || p.at(key).is_null()) return out;

    const auto& val = p.at(key);
    if (val.is_array()) {
        const int nread = std::min<int>(N, val.size());
        for (int i = 0; i < nread; ++i) out[i] = val.at(i).get<double>();
        return out;
    }

    if (val.is_string()) {
        return zigzag::parseCommaListReal(val.get<std::string>(), N, uniform_value);
    }

    throw std::runtime_error("multi_zigzag: optional list field '" + key + "' must be an array or comma-separated string");
}

struct Patch {
    std::string name;
    bool enabled = true;
    Quad corners;
    patch_growth::RectangleInfo rect;

    std::string coverage_mode = "cover_patch";
    std::string lv_mode = "auto_patch_height";
    std::string N_mode = "auto_cover_width";
    std::string rotation_mode = "auto_patch_angle";

    double Lv_mm = -1.0;
    double alpha_deg = 15.0;
    int N_total = -1;
    int N_min = 4;
    int N_max = 200;
    double w_mm = 2.0;
    double rotation_deg = 0.0;

    double offset_dx_mm = 0.0;
    double offset_dy_mm = 0.0;

    double growth_top = 0.0;
    double growth_bot = 0.0;
    double ortho_coeff = 0.0;

    zigzag::TopProfileMode top_profile_mode = zigzag::TopProfileMode::Uniform;
    double top_end_ratio = 1.0;
    double top_profile_power = 1.0;

    std::vector<double> gtop_list;
    std::vector<double> gbot_list;
    std::vector<double> ortho_list;
};

inline Patch parsePatch(const json& p, int patch_index)
{
    Patch z;
    z.name = getStringOrDefault(p, "name", "zigzag_patch_" + std::to_string(patch_index));
    z.enabled = getBoolOrDefault(p, "enabled", true);
    z.corners = readCornersMm(p);

    if (!patch_growth::isRectangle(z.corners, 1e-7, 1e-10)) {
        std::ostringstream oss;
        oss << "multi_zigzag: patch '" << z.name
            << "' is not rectangular. Version 1 only supports rectangular zigzag patches. corners="
            << patch_growth::quadToStringMm(z.corners);
        throw std::runtime_error(oss.str());
    }
    z.rect = patch_growth::rectangleInfo(z.corners);

    z.coverage_mode = getStringOrDefault(p, "zigzag_coverage_mode", "cover_patch");
    if (z.coverage_mode != "cover_patch") {
        throw std::runtime_error("multi_zigzag: version 1 only supports zigzag_coverage_mode='cover_patch'");
    }

    z.lv_mode = getStringOrDefault(p, "zigzag_lv_mode", "auto_patch_height");
    z.N_mode = getStringOrDefault(p, "zigzag_N_mode", "auto_cover_width");
    z.rotation_mode = getStringOrDefault(p, "zigzag_rotation_mode", "auto_patch_angle");

    z.alpha_deg = getDoubleOrDefault(p, "zigzag_alpha_deg", 15.0);
    z.w_mm = getDoubleOrDefault(p, "zigzag_w_mm", 2.0);
    z.N_min = getIntOrDefault(p, "zigzag_N_min", 4);
    z.N_max = getIntOrDefault(p, "zigzag_N_max", 200);

    z.offset_dx_mm = getDoubleOrDefault(p, "zigzag_offset_dx_mm", 0.0);
    z.offset_dy_mm = getDoubleOrDefault(p, "zigzag_offset_dy_mm", 0.0);

    z.growth_top = getDoubleOrDefault(p, "growth_top", 0.0);
    z.growth_bot = getDoubleOrDefault(p, "growth_bot", 0.0);
    z.ortho_coeff = getDoubleOrDefault(p, "ortho_coeff", 0.0);

    const std::string profile_mode = getStringOrDefault(p, "zigzag_profile_mode", "uniform");
    z.top_profile_mode = zigzag::parseTopProfileMode(profile_mode);
    z.top_end_ratio = getDoubleOrDefault(p, "zigzag_top_end_ratio", 1.0);
    z.top_profile_power = getDoubleOrDefault(p, "zigzag_top_profile_power", 1.0);

    if (z.lv_mode == "auto_patch_height") {
        z.Lv_mm = z.rect.height_m * 1e3;
    } else if (z.lv_mode == "manual") {
        z.Lv_mm = getDoubleOrDefault(p, "zigzag_lv_mm", -1.0);
        if (z.Lv_mm <= 0.0) {
            throw std::runtime_error("multi_zigzag: zigzag_lv_mode='manual' requires zigzag_lv_mm > 0");
        }
    } else {
        throw std::runtime_error("multi_zigzag: zigzag_lv_mode must be 'auto_patch_height' or 'manual'");
    }

    if (z.N_mode == "auto_cover_width") {
        z.N_total = patch_growth::autoEvenZigZagNForCover(z.rect.width_m,
                                                          z.Lv_mm * 1e-3,
                                                          z.alpha_deg,
                                                          z.w_mm * 1e-3,
                                                          z.N_min,
                                                          z.N_max);
    } else if (z.N_mode == "manual") {
        z.N_total = getIntOrDefault(p, "zigzag_N", -1);
        if (z.N_total < 2) {
            throw std::runtime_error("multi_zigzag: zigzag_N_mode='manual' requires zigzag_N >= 2");
        }
    } else {
        throw std::runtime_error("multi_zigzag: zigzag_N_mode must be 'auto_cover_width' or 'manual'");
    }

    if (z.rotation_mode == "auto_patch_angle") {
        z.rotation_deg = patch_growth::rad2deg(z.rect.angle_rad);
    } else if (z.rotation_mode == "manual") {
        z.rotation_deg = getDoubleOrDefault(p, "zigzag_rotation_deg", 0.0);
    } else {
        throw std::runtime_error("multi_zigzag: zigzag_rotation_mode must be 'auto_patch_angle' or 'manual'");
    }

    z.gtop_list = readNumberListOrUniform(p, "zigzag_gtop_list", z.N_total, z.growth_top);
    z.gbot_list = readNumberListOrUniform(p, "zigzag_gbot_list", z.N_total, z.growth_bot);
    z.ortho_list = readNumberListOrUniform(p, "zigzag_ortho_list", z.N_total, z.ortho_coeff);

    return z;
}

inline std::vector<Patch> loadPatches(const std::string& patch_file, CombineMode& combine_mode)
{
    std::ifstream in(patch_file);
    if (!in) {
        throw std::runtime_error("multi_zigzag: cannot open patch_file: " + patch_file);
    }

    json j;
    in >> j;

    const std::string mode = getStringOrDefault(j, "combine_mode", "error_on_overlap");
    combine_mode = parseCombineMode(mode);

    if (j.contains("growth_type")) {
        const std::string gt = j.at("growth_type").get<std::string>();
        if (gt != "multi_zigzag") {
            throw std::runtime_error("multi_zigzag: JSON growth_type must be 'multi_zigzag'");
        }
    }

    if (!j.contains("patches") || !j.at("patches").is_array()) {
        throw std::runtime_error("multi_zigzag: patch_file must contain a patches array");
    }

    std::vector<Patch> patches;
    int idx = 0;
    for (const auto& p : j.at("patches")) {
        Patch z = parsePatch(p, idx);
        if (z.enabled) patches.push_back(std::move(z));
        ++idx;
    }

    if (patches.empty()) {
        throw std::runtime_error("multi_zigzag: no enabled patches found");
    }

    return patches;
}

template <typename MeshType>
inline void applyPatch(const MeshType& mesh,
                       const Patch& zp,
                       CombineMode combine_mode,
                       Eigen::VectorXd& growthRates_t,
                       Eigen::VectorXd& growthRates_b,
                       Eigen::VectorXd& growthAngles,
                       Eigen::VectorXd& orthoCoeffFaces,
                       Eigen::VectorXi& patchHitCount,
                       Eigen::VectorXi* passCountFaces)
{
    const int nFaces = mesh.getNumberOfFaces();

    zigzag::Params zz;
    zz.Lv_mm = zp.Lv_mm;
    zz.alpha_deg = zp.alpha_deg;
    zz.N_total = zp.N_total;
    zz.w_mm = zp.w_mm;
    zz.offset_dx_mm = zp.offset_dx_mm;
    zz.offset_dy_mm = zp.offset_dy_mm;
    zz.rotation_deg = zp.rotation_deg;
    zz.last_wins = true;
    zz.zero_outside = false; // important: do not clear fields when applying each patch
    zz.start_mode = zigzag::StartMode::LeftBottom_Up;
    zz.top_profile_mode = zp.top_profile_mode;
    zz.top_end_ratio = zp.top_end_ratio;
    zz.top_profile_power = zp.top_profile_power;

    const auto strips = zigzag::buildZigZagStrips(zz, zp.gtop_list, zp.gbot_list, zp.ortho_list);
    const double half_w = 0.5 * (zz.w_mm * 1e-3);
    const double rot_rad = zigzag::deg2rad(zz.rotation_deg);
    const Eigen::Matrix2d Rinv = zigzag::rot2d(-rot_rad);
    const Eigen::Vector2d offset(zp.offset_dx_mm * 1e-3, zp.offset_dy_mm * 1e-3);

    const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
    const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();

    int patch_faces = 0;
    int strip_faces = 0;

    for (int i = 0; i < nFaces; ++i) {
        const int i0 = F(i,0);
        const int i1 = F(i,1);
        const int i2 = F(i,2);

        const Eigen::Vector2d c(
            (V(i0,0) + V(i1,0) + V(i2,0)) / 3.0,
            (V(i0,1) + V(i1,1) + V(i2,1)) / 3.0
        );

        if (!patch_growth::pointInConvexQuad(c, zp.corners)) continue;
        ++patch_faces;

        // Transform global face centroid into this patch's zigzag local frame.
        const Eigen::Vector2d q = c - zp.rect.center - offset;
        const Eigen::Vector2d p = Rinv * q;

        bool face_written_by_this_patch = false;

        // Last strip wins inside one zigzag patch, matching the existing single zigzag behavior.
        for (int k = 0; k < static_cast<int>(strips.size()); ++k) {
            const auto& s = strips[k];
            const double d = zigzag::dist_point_segment_2d(p, s.a, s.b);
            if (d > half_w) continue;

            double gtop_local = s.gtop;
            if (zz.top_profile_mode == zigzag::TopProfileMode::CenterPeak) {
                const double s01 = zigzag::path_coord_01_on_segment_2d(p, s.a, s.b);
                const double profile = zigzag::center_peak_profile_01(s01, zz.top_end_ratio, zz.top_profile_power);
                gtop_local *= profile;
            }

            const bool already_hit_by_other_patch = (patchHitCount(i) > 0) && !face_written_by_this_patch;
            if (already_hit_by_other_patch && combine_mode == CombineMode::ErrorOnOverlap) {
                std::ostringstream oss;
                oss << "multi_zigzag: overlap detected at face " << i
                    << " while applying patch '" << zp.name << "'. Use combine_mode='sum' or 'override', or adjust patch regions.";
                throw std::runtime_error(oss.str());
            }

            if (combine_mode == CombineMode::Sum && already_hit_by_other_patch) {
                growthRates_t(i) += gtop_local;
                growthRates_b(i) += s.gbot;
                // Angle/ortho are not uniquely defined for summed overlapping anisotropic fields.
                // Keep the latest values for diagnostics; overlap should usually be avoided.
                growthAngles(i) = zigzag::normalize_angle_pi(s.angle_rad + rot_rad);
                orthoCoeffFaces(i) = s.ortho;
            } else {
                growthRates_t(i) = gtop_local;
                growthRates_b(i) = s.gbot;
                growthAngles(i) = zigzag::normalize_angle_pi(s.angle_rad + rot_rad);
                orthoCoeffFaces(i) = s.ortho;
            }

            if (!face_written_by_this_patch) {
                ++strip_faces;
                face_written_by_this_patch = true;
            }
            if (passCountFaces) (*passCountFaces)(i) += 1;
        }

        if (face_written_by_this_patch) {
            patchHitCount(i) += 1;
        }
    }

    const double span_x_m = std::max(0, zp.N_total - 2) * (zp.Lv_mm * 1e-3) * std::tan(patch_growth::deg2rad(zp.alpha_deg));

    std::cout << "[multi_zigzag] patch='" << zp.name << "'"
              << " rect_width_mm=" << zp.rect.width_m * 1e3
              << " rect_height_mm=" << zp.rect.height_m * 1e3
              << " Lv_mm=" << zp.Lv_mm
              << " alpha_deg=" << zp.alpha_deg
              << " auto/manual_N=" << zp.N_total
              << " w_mm=" << zp.w_mm
              << " rotation_deg=" << zp.rotation_deg
              << " approx_span_x_mm=" << span_x_m * 1e3
              << " patch_faces=" << patch_faces
              << " strip_faces=" << strip_faces
              << "\n";
}

template <typename MeshType>
inline void apply(const MeshType& mesh,
                  const std::string& patch_file,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXd& orthoCoeffFaces,
                  Eigen::VectorXi* passCountFaces = nullptr)
{
    const int nFaces = mesh.getNumberOfFaces();
    if (growthRates_t.size() != nFaces || growthRates_b.size() != nFaces) {
        throw std::runtime_error("multi_zigzag::apply: growthRates size mismatch");
    }
    if (growthAngles.size() != nFaces || orthoCoeffFaces.size() != nFaces) {
        throw std::runtime_error("multi_zigzag::apply: growthAngles/orthoCoeffFaces size mismatch");
    }
    if (passCountFaces && passCountFaces->size() != nFaces) {
        throw std::runtime_error("multi_zigzag::apply: passCountFaces size mismatch");
    }

    CombineMode combine_mode;
    const auto patches = loadPatches(patch_file, combine_mode);

    growthRates_t.setZero();
    growthRates_b.setZero();
    orthoCoeffFaces.setZero();
    // Keep growthAngles initialized outside active patches.

    Eigen::VectorXi patchHitCount(nFaces);
    patchHitCount.setZero();
    if (passCountFaces) passCountFaces->setZero();

    std::cout << "[multi_zigzag] loaded enabled_patches=" << patches.size()
              << " from patch_file=" << patch_file << "\n";

    for (const auto& zp : patches) {
        applyPatch(mesh, zp, combine_mode,
                   growthRates_t, growthRates_b,
                   growthAngles, orthoCoeffFaces,
                   patchHitCount, passCountFaces);
    }

    int active_faces = 0;
    int overlap_faces = 0;
    for (int i = 0; i < nFaces; ++i) {
        if (patchHitCount(i) > 0) ++active_faces;
        if (patchHitCount(i) > 1) ++overlap_faces;
    }

    std::cout << "[multi_zigzag] active_faces=" << active_faces
              << " overlap_faces=" << overlap_faces
              << " top[min,max]=[" << growthRates_t.minCoeff() << "," << growthRates_t.maxCoeff() << "]"
              << " bot[min,max]=[" << growthRates_b.minCoeff() << "," << growthRates_b.maxCoeff() << "]"
              << "\n";
}

} // namespace multi_zigzag
