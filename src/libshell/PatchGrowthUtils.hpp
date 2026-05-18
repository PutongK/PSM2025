#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

namespace patch_growth {

using Vec2 = Eigen::Vector2d;
using Quad = std::array<Vec2, 4>;

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

inline double cross2d(const Vec2& a, const Vec2& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

inline double signedArea2(const Quad& q)
{
    double a2 = 0.0;
    for (int i = 0; i < 4; ++i) {
        const Vec2& p = q[i];
        const Vec2& r = q[(i + 1) % 4];
        a2 += cross2d(p, r);
    }
    return a2;
}

inline double area(const Quad& q)
{
    return 0.5 * std::abs(signedArea2(q));
}

inline Vec2 centroid(const Quad& q)
{
    return 0.25 * (q[0] + q[1] + q[2] + q[3]);
}

inline Quad ensureCounterClockwise(Quad q)
{
    if (signedArea2(q) < 0.0) {
        std::swap(q[1], q[3]);
    }
    return q;
}

inline bool isConvexQuad(const Quad& q_in, double tol = 1e-14)
{
    const Quad q = ensureCounterClockwise(q_in);
    if (area(q) <= tol) return false;

    for (int i = 0; i < 4; ++i) {
        const Vec2 e0 = q[(i + 1) % 4] - q[i];
        const Vec2 e1 = q[(i + 2) % 4] - q[(i + 1) % 4];
        if (cross2d(e0, e1) < -tol) return false;
    }
    return true;
}

inline bool pointInConvexQuad(const Vec2& p, const Quad& q_in, double tol = 1e-14)
{
    const Quad q = ensureCounterClockwise(q_in);
    for (int i = 0; i < 4; ++i) {
        const Vec2 a = q[i];
        const Vec2 b = q[(i + 1) % 4];
        if (cross2d(b - a, p - a) < -tol) return false;
    }
    return true;
}

inline double edgeLength(const Quad& q, int i)
{
    return (q[(i + 1) % 4] - q[i]).norm();
}

inline bool nearZero(double x, double tol)
{
    return std::abs(x) <= tol;
}

inline bool nearlyEqual(double a, double b, double relTol = 1e-8, double absTol = 1e-12)
{
    return std::abs(a - b) <= std::max(absTol, relTol * std::max(std::abs(a), std::abs(b)));
}

inline bool isRectangle(const Quad& q_in, double relTol = 1e-8, double absTol = 1e-12)
{
    const Quad q = ensureCounterClockwise(q_in);
    if (!isConvexQuad(q, absTol)) return false;

    const Vec2 e0 = q[1] - q[0];
    const Vec2 e1 = q[2] - q[1];
    const Vec2 e2 = q[3] - q[2];
    const Vec2 e3 = q[0] - q[3];

    const double scale = std::max({e0.norm(), e1.norm(), e2.norm(), e3.norm(), 1.0});
    const double dotTol = absTol + relTol * scale * scale;

    if (!nearZero(e0.dot(e1), dotTol)) return false;
    if (!nearZero(e1.dot(e2), dotTol)) return false;
    if (!nearZero(e2.dot(e3), dotTol)) return false;
    if (!nearZero(e3.dot(e0), dotTol)) return false;

    if (!nearlyEqual(e0.norm(), e2.norm(), relTol, absTol)) return false;
    if (!nearlyEqual(e1.norm(), e3.norm(), relTol, absTol)) return false;

    const double d02 = (q[2] - q[0]).norm();
    const double d13 = (q[3] - q[1]).norm();
    if (!nearlyEqual(d02, d13, relTol, absTol)) return false;

    return true;
}

struct RectangleInfo {
    Vec2 center;
    double width_m = 0.0;     // along P0 -> P1
    double height_m = 0.0;    // along P1 -> P2
    double angle_rad = 0.0;   // orientation of P0 -> P1 relative to global +x
};

inline RectangleInfo rectangleInfo(const Quad& q_in)
{
    const Quad q = ensureCounterClockwise(q_in);
    RectangleInfo info;
    info.center = centroid(q);
    info.width_m = (q[1] - q[0]).norm();
    info.height_m = (q[2] - q[1]).norm();
    const Vec2 e0 = q[1] - q[0];
    info.angle_rad = std::atan2(e0.y(), e0.x());
    return info;
}

inline int nextEvenAtLeast(int n, int minVal = 4)
{
    n = std::max(n, minVal);
    if (n % 2 != 0) ++n;
    return n;
}

inline int autoEvenZigZagNForCover(double patch_width_m,
                                   double Lv_m,
                                   double alpha_deg,
                                   double w_m,
                                   int N_min = 4,
                                   int N_max = 200)
{
    const double alpha = deg2rad(alpha_deg);
    const double t = std::tan(alpha);
    if (Lv_m <= 0.0) throw std::runtime_error("autoEvenZigZagNForCover: Lv_m must be > 0");
    if (w_m <= 0.0) throw std::runtime_error("autoEvenZigZagNForCover: w_m must be > 0");
    if (std::abs(t) < 1e-12) {
        throw std::runtime_error("autoEvenZigZagNForCover: alpha_deg is too small for auto_cover_width mode");
    }

    const double required = (patch_width_m - w_m) / (Lv_m * std::abs(t)) + 2.0;
    int N = nextEvenAtLeast(static_cast<int>(std::ceil(required)), N_min);

    if (N > N_max) {
        std::ostringstream oss;
        oss << "autoEvenZigZagNForCover: required N_total=" << N
            << " exceeds N_max=" << N_max
            << ". Increase alpha_deg or Lv_m, reduce patch width, or increase zigzag_N_max.";
        throw std::runtime_error(oss.str());
    }
    return N;
}

inline std::string quadToStringMm(const Quad& q)
{
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < 4; ++i) {
        if (i) oss << ", ";
        oss << "(" << q[i].x() * 1e3 << ", " << q[i].y() * 1e3 << ")";
    }
    oss << "] mm";
    return oss.str();
}

} // namespace patch_growth
