#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace zigzag {

// ------------------------
// Config / CLI parameters
// ------------------------
enum class StartMode {
    LeftBottom_Up
};

struct Params {
    double Lv_mm      = 140.0;   // vertical span in mm
    double alpha_deg  = 15.0;   // half-angle relative to vertical, deg
    int    N_total    = 6;      // total strips = (N_inclined + 2 vertical)
    double w_mm       = 8.0;    // strip width in mm

    bool last_wins    = true;
    StartMode start_mode = StartMode::LeftBottom_Up;

    // If true: set all growth to zero first, then write only where strips cover
    // This matches "growth only on toolpath".
    bool zero_outside = true;
};

// ------------------------
// Small utilities
// ------------------------
inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// Normalize orientation so opposite directions are equivalent (0..pi)
inline double normalize_angle_pi(double a) {
    // map to (-pi, pi]
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a >   M_PI) a -= 2.0 * M_PI;
    // identify opposite directions: map to [0, pi)
    if (a < 0.0) a += M_PI;
    return a;
}

// Parse comma list of Real; if fewer than N values provided, fill with default.
// If more than N, ignore extras.
inline std::vector<double> parseCommaListReal(const std::string& s, int N, double default_val)
{
    std::vector<double> out;
    out.reserve(N);

    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // trim spaces
        size_t b = item.find_first_not_of(" \t\r\n");
        size_t e = item.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        std::string token = item.substr(b, e - b + 1);
        out.push_back(std::stod(token));
        if ((int)out.size() >= N) break;
    }
    while ((int)out.size() < N) out.push_back(default_val);
    return out;
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

// A single strip = a segment centerline + width + parameters
struct StripSegment {
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    double width;          // meters
    double gtop;           // growth top
    double gbot;           // growth bot
    double ortho;          // ortho coeff
    double angle_rad;      // orientation in [0,pi)
};

// Build strips from parametric zigzag (centered at origin)
inline std::vector<StripSegment> buildZigZagStrips(const Params& zz,
                                                   const std::vector<double>& gtop_list,
                                                   const std::vector<double>& gbot_list,
                                                   const std::vector<double>& ortho_list)
{
    if (zz.N_total < 2) throw std::runtime_error("zigzag_N must be >= 2");
    const int N_incl = zz.N_total - 2;

    const double Lv = zz.Lv_mm * 1e-3;
    const double w  = zz.w_mm  * 1e-3;
    const double a  = deg2rad(zz.alpha_deg);

    // p = Lv * tan(alpha) (alpha relative to vertical)
    const double p = Lv * std::tan(a);

    const double y_bot = -0.5 * Lv;
    const double y_top = +0.5 * Lv;

    // total horizontal span covered by inclined strips
    const double W = (N_incl > 0) ? (N_incl * p) : 0.0;
    const double x_L = -0.5 * W;
    const double x_R = +0.5 * W;

    // Build polyline points (start left-bottom -> up)
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(2 + N_incl + 1);

    pts.emplace_back(x_L, y_bot);  // P0
    pts.emplace_back(x_L, y_top);  // P1  (left vertical)

    // inclined turning points: x increases by p each segment
    // j=1 -> first inclined goes from top to bottom at x_L+p
    for (int j = 1; j <= N_incl; ++j) {
        const double x = x_L + j * p;
        const double y = (j % 2 == 1) ? y_bot : y_top;
        pts.emplace_back(x, y);
    }

    // right vertical strip: go to opposite extreme at same x_R
    // If N_incl==0, pts.back() is P1 at (x_L,y_top) and x_R==x_L; still works.
    const Eigen::Vector2d plast = pts.back();
    const double y_other = (std::abs(plast.y() - y_top) < 1e-15) ? y_bot : y_top;
    pts.emplace_back(x_R, y_other);

    // Now construct strips from consecutive segments.
    // Segment index k corresponds to strip index k in your N_total lists:
    // k=0 left vertical, k=1..N_incl inclined, k=N_total-1 right vertical.
    std::vector<StripSegment> strips;
    strips.reserve(zz.N_total);

    auto make_strip = [&](int k, const Eigen::Vector2d& a2, const Eigen::Vector2d& b2) {
        StripSegment s;
        s.a = a2;
        s.b = b2;
        s.width = w;
        s.gtop  = gtop_list.at(k);
        s.gbot  = gbot_list.at(k);
        s.ortho = ortho_list.at(k);

        const Eigen::Vector2d d = b2 - a2;
        const double ang = std::atan2(d.y(), d.x());       // relative to +x
        s.angle_rad = normalize_angle_pi(ang);             // orientation only
        return s;
    };

    // pts layout:
    // 0->1 : left vertical
    // 1->2 : first inclined (if N_incl>=1) OR right vertical (if N_incl==0, because x_R==x_L)
    // ...
    // last-1 -> last : right vertical
    //
    // We want exactly N_total segments:
    // left vertical + N_incl inclined + right vertical = 2 + N_incl = N_total
    strips.push_back(make_strip(0, pts[0], pts[1]));

    for (int k = 1; k <= N_incl; ++k) {
        strips.push_back(make_strip(k, pts[k], pts[k+1]));
    }

    strips.push_back(make_strip(zz.N_total - 1, pts[pts.size()-2], pts[pts.size()-1]));

    return strips;
}

// ------------------------
// Main application function
// ------------------------
template <typename MeshType>
inline void apply(const MeshType& mesh,
                  const Params& zz,
                  const std::vector<double>& gtop_list,
                  const std::vector<double>& gbot_list,
                  const std::vector<double>& ortho_list,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXd& orthoCoeffFaces)
{
    const int nFaces = mesh.getNumberOfFaces();
    if (growthRates_t.size() != nFaces || growthRates_b.size() != nFaces)
        throw std::runtime_error("zigzag::apply: growthRates size mismatch");
    if (growthAngles.size() != nFaces || orthoCoeffFaces.size() != nFaces)
        throw std::runtime_error("zigzag::apply: growthAngles/orthoCoeffFaces size mismatch");
    if ((int)gtop_list.size() != zz.N_total || (int)gbot_list.size() != zz.N_total || (int)ortho_list.size() != zz.N_total)
        throw std::runtime_error("zigzag::apply: parameter lists must have size N_total");

    // Build strips (centered)
    const auto strips = buildZigZagStrips(zz, gtop_list, gbot_list, ortho_list);
    const double half_w = 0.5 * (zz.w_mm * 1e-3);

    // Optionally zero everything first
    if (zz.zero_outside) {
        growthRates_t.setZero();
        growthRates_b.setZero();
        // keep growthAngles as-is outside (could also set to 0); leaving as-is is fine
        // ortho outside should be 0 if no growth
        orthoCoeffFaces.setZero();
    }

    // Get vertices + faces
    const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
    const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();

    // Center mesh XY so zigzag strips (built around origin) hit the plate
    const double xmin = V.col(0).minCoeff();
    const double xmax = V.col(0).maxCoeff();
    const double ymin = V.col(1).minCoeff();
    const double ymax = V.col(1).maxCoeff();
    const Eigen::Vector2d center(0.5*(xmin + xmax), 0.5*(ymin + ymax));

    // For each strip in order: last wins = overwrite as we go
    for (int k = 0; k < (int)strips.size(); ++k) {
        const auto& s = strips[k];

        for (int i = 0; i < nFaces; ++i) {
            const int i0 = F(i,0);
            const int i1 = F(i,1);
            const int i2 = F(i,2);

            const Eigen::Vector2d c(
                (V(i0,0) + V(i1,0) + V(i2,0)) / 3.0,
                (V(i0,1) + V(i1,1) + V(i2,1)) / 3.0
            );

            const Eigen::Vector2d cc = c - center; // center XY, shift to zigzag frame

            const double d = dist_point_segment_2d(c, s.a, s.b);
            if (d <= half_w) {
                growthRates_t(i) = s.gtop;
                growthRates_b(i) = s.gbot;
                growthAngles(i)  = s.angle_rad;
                orthoCoeffFaces(i) = s.ortho;
            }
        }
    }
}

} // namespace zigzag

