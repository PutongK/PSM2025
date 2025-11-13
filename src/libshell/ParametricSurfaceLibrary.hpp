//
//  ParametricSurfaceLibrary.hpp
//  Elasticity
//
//  Created by Wim van Rees on 7/7/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef ParametricSurfaceLibrary_hpp
#define ParametricSurfaceLibrary_hpp

#include "common.hpp"

struct ParametricSurface
{
    virtual Eigen::Vector3d operator()(const Real uu, const Real vv) const = 0;
    virtual Eigen::Vector3d get_xu(const Real uu, const Real vv) const = 0;
    virtual Eigen::Vector3d get_xv(const Real uu, const Real vv) const = 0;
    virtual Real getMeanCurvature(const Real uu, const Real vv) const = 0;
    virtual Real getGaussCurvature(const Real uu, const Real vv) const = 0;

    virtual std::pair<Real,Real> getExtent_U() const = 0;
    virtual std::pair<Real,Real> getExtent_V() const = 0;

    std::tuple<Real, Real, Real> get_FFF(const Real uu, const Real vv, const Eigen::Vector3d & e1, const Eigen::Vector3d & e2) const
    {
        const Eigen::Vector3d xu = this->get_xu(uu, vv);
        const Eigen::Vector3d xv = this->get_xv(uu, vv);

        // [e1x e1y] [xu.xu xu.xv; xu.xv xv.xv] [e1x e1y]
        // [e1x e1y] [e1x * xu.xu + e1y*xu.xv ; e1x * xu.xv + e1y*xv.xv]
        // e1x*e1x * xu.xu + 2 * e1x*e1y * xu.xv + e1h
        const Real xu_dot_xu = xu.dot(xu);
        const Real xu_dot_xv = xu.dot(xv);
        const Real xv_dot_xv = xv.dot(xv);

        const Real I_E = xu_dot_xu*e1(0)*e1(0) + 2*xu_dot_xv*e1(0)*e1(1) + xv_dot_xv*e1(1)*e1(1);
        const Real I_F = xu_dot_xu*e1(0)*e2(0) + xu_dot_xv*(e1(0)*e2(1) + e1(1)*e2(0)) + xv_dot_xv*e1(1)*e2(1);
        const Real I_G = xu_dot_xu*e2(0)*e2(0) + 2*xu_dot_xv*e2(0)*e2(1) + xv_dot_xv*e2(1)*e2(1);

        return std::make_tuple(I_E, I_F, I_G);
    }

    virtual Eigen::Vector3d getNormal(const Real uu, const Real vv) const
    {
        const Eigen::Vector3d xu = get_xu(uu,vv);
        const Eigen::Vector3d xv = get_xv(uu,vv);
        const Eigen::Vector3d xu_cross_xv = xu.cross(xv);
        return xu_cross_xv.normalized();
    }
};

struct ParametricCylinder : ParametricSurface
{
    const Real radius, length, theta;

    ParametricCylinder(const Real R, const Real L, const Real th = 2.0*M_PI):
    radius(R),
    length(L),
    theta(th)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        Eigen::Vector3d retval;
        retval << vv, radius*std::cos(uu), radius*std::sin(uu);
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real ) const override
    {
        Eigen::Vector3d retval;
        retval << 0, -radius*std::sin(uu), radius*std::cos(uu);
        return retval;
    }

    Eigen::Vector3d get_xv(const Real , const Real ) const override
    {
        Eigen::Vector3d retval;
        retval << 1,0,0;
        return retval;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        return 0.5/radius;
    }

    Real getGaussCurvature(const Real, const Real ) const override
    {
        return 0.0;
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(0, theta);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(0, length);
    }
};


struct ParametricIsoScaledSurface : ParametricSurface
{
    const Real scaleFac;

    ParametricIsoScaledSurface(const Real scaleFac):
    scaleFac(scaleFac)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        Eigen::Vector3d retval;
        retval << scaleFac*uu, scaleFac*vv, 0;
        return retval;
    }

    Eigen::Vector3d get_xu(const Real , const Real ) const override
    {
        Eigen::Vector3d retval;
        retval << scaleFac, 0, 0;
        return retval;
    }

    Eigen::Vector3d get_xv(const Real , const Real ) const override
    {
        Eigen::Vector3d retval;
        retval << 0, scaleFac, 0;
        return retval;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        return 0.0;
    }

    Real getGaussCurvature(const Real, const Real ) const override
    {
        return 0.0;
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(-1, 1);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-1, 1);
    }
};


struct ParametricSphericalShell : ParametricSurface
{
    const Real radius;
    const Real halfOpeningAngle;
    const bool scaleBaseToOne;

    ParametricSphericalShell(const Real radius_in, const Real angle_in, const bool scaleBaseToOne_in = false):
    radius(radius_in),
    halfOpeningAngle(angle_in),
    scaleBaseToOne(scaleBaseToOne_in)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        // stereographic projection
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1

        const Real denum = 1 + urel*urel + vrel*vrel;
        const Real x = scaleFac * 2*urel/denum;
        const Real y = scaleFac * 2*vrel/denum;
        const Real z = scaleFac * ( -1 + urel*urel + vrel*vrel)/denum;

        Eigen::Vector3d retval;
        retval << x,y,z;
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real vv) const override
    {
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1

        const Real denum = std::pow(1.0 + urel*urel + vrel*vrel,2);

        const Real xu_x = scaleFac * 2*(1 - urel*urel + vrel*vrel) / denum;
        const Real xu_y = scaleFac * -4*urel*vrel / denum;
        const Real xu_z = scaleFac * 4*urel/ denum;

        const Eigen::Vector3d xu = (Eigen::Vector3d() << xu_x, xu_y, xu_z).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real uu, const Real vv) const override
    {
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1

        const Real denum = std::pow(1.0 + urel*urel + vrel*vrel,3);

        const Real xv_x = scaleFac * -4*urel*vrel / denum;
        const Real xv_y = scaleFac * 2*(1 + urel*urel - vrel*vrel) / denum;
        const Real xv_z = scaleFac * 4*vrel/ denum;

        const Eigen::Vector3d xv = (Eigen::Vector3d() << xv_x, xv_y, xv_z).finished();
        return xv;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        return 1.0/radius;
    }

    Real getGaussCurvature(const Real , const Real ) const override
    {
        return  1.0/(radius*radius);
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(-radius, radius);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-radius, radius);
    }
};



struct ParametricEnneper : ParametricSurface
{
    const int nFac;
    const Real rExtent;

    ParametricEnneper(const int nFac, const Real rExtent=1):
    nFac(nFac),
    rExtent(rExtent)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        // shortcuts
        const Real r = uu;
        const Real t = vv; // theta
        const int n = nFac;

        const Real r_pow_2n = std::pow(r,2*n);

        const Real x = r*std::cos(t) + (r_pow_2n*std::cos(t - 2*n*t))/(r - 2*n*r);
        const Real y = r*std::sin(t) + (r_pow_2n*std::sin(t - 2*n*t))/(r - 2*n*r);
        const Real z = 2*std::pow(r,n)*std::cos(n*t)/n;

        Eigen::Vector3d retval;
        retval << x,y,z;
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real vv) const override
    {
        // shortcuts
        const Real r = uu;
        const Real t = vv; // theta
        const int n = nFac;

        const Real dx = std::cos(t) - std::pow(r, 2*n-2)*std::cos(t - 2*n*t);
        const Real dy = std::sin(t) - std::pow(r, 2*n-2)*std::sin(t - 2*n*t);
        const Real dz = 2*std::pow(r, n-1)*std::cos(n*t);

        const Eigen::Vector3d xu = (Eigen::Vector3d() << dx, dy, dz ).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real uu, const Real vv) const override
    {
        // shortcuts
        const Real r = uu;
        const Real t = vv; // theta
        const int n = nFac;

        const Real dx = -r*std::sin(t) - std::pow(r, 2*n-1)*std::sin(t - 2*n*t);
        const Real dy =  r*std::cos(t) + std::pow(r, 2*n-1)*std::cos(t - 2*n*t);
        const Real dz = -2*std::pow(r, n)*std::sin(n*t);

        const Eigen::Vector3d xv = (Eigen::Vector3d() << dx, dy, dz ).finished();
        return xv;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        return 0.0; // minimal surface
    }

    Real getGaussCurvature(const Real , const Real ) const override
    {
        std::cout << "gaussian curvature for generalized enneper surfaces not yet implemented\n";
        return 0;
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(0, rExtent);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-M_PI, M_PI);
    }
};


struct ParametricSphere : ParametricSurface
{
    const Real radius;

    ParametricSphere(const Real radius_in):
    radius(radius_in)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        Eigen::Vector3d retval;
        const Real tx = radius*std::cos(uu)*std::sin(vv);
        const Real ty = radius*std::sin(uu)*std::sin(vv);
        const Real tz = radius*std::cos(vv);
        retval << tx, ty, tz;
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real vv) const override
    {
        const Eigen::Vector3d xu = (Eigen::Vector3d() <<
                                    -radius*std::sin(uu)*std::sin(vv),
                                     radius*std::cos(uu)*std::sin(vv),
                                    0.0
                                    ).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real uu, const Real vv) const override
    {
        const Eigen::Vector3d xv = (Eigen::Vector3d() <<
                                    radius*std::cos(uu)*std::cos(vv),
                                    radius*std::sin(uu)*std::cos(vv),
                                    -radius*std::sin(vv)
                                    ).finished();
        return xv;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        return 1.0/radius;
    }

    Real getGaussCurvature(const Real , const Real ) const override
    {
        return  1.0/(radius*radius);
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(0.0, 2.0*M_PI);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(0.0, M_PI);
    }
};



struct ParametricSinusoidalSurface : ParametricSurface
{
    const Real amplitude;

    ParametricSinusoidalSurface(const Real amplitude):
    amplitude(amplitude)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        const Real xx = std::sin(2.0*M_PI*uu);
        const Real yy = std::cos(4.0*M_PI*vv);
        const Real xx_yy = xx*yy;
        const Real expfac = std::exp(-(uu*uu + vv*vv)/0.5);

        Eigen::Vector3d retval;
        retval << uu, vv, amplitude*expfac*xx_yy;

        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real vv) const override
    {
        const Real expFac = std::exp(-(uu*uu + vv*vv)/0.5);
        const Real dzdu = amplitude*2.0*expFac*std::cos(4.0*M_PI*vv)*(M_PI*std::cos(2.0*M_PI*uu) - 2.0*uu*std::sin(2.0*M_PI*uu));

        const Eigen::Vector3d xu = (Eigen::Vector3d() <<
                                    1.0,
                                    0.0,
                                    dzdu
                                    ).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real uu, const Real vv) const override
    {
        const Real expFac = std::exp(-(uu*uu + vv*vv)/0.5);
        const Real dzdv = -amplitude*4.0*expFac*std::sin(2.0*M_PI*uu)*(M_PI*std::sin(4.0*M_PI*vv) + vv*std::cos(4.0*M_PI*vv));

        const Eigen::Vector3d xv = (Eigen::Vector3d() <<
                                    0.0,
                                    1.0,
                                    dzdv
                                    ).finished();
        return xv;
    }

    Real getMeanCurvature(const Real , const Real ) const override
    {
        std::cout << "mean curvature not implemented for parametric sinusoidal surface\n";
        return -1;
    }

    Real getGaussCurvature(const Real , const Real ) const override
    {
        std::cout << "gauss curvature not implemented for parametric sinusoidal surface\n";
        return -1;
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(-1.0, 1.0);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-1.0, 1.0);
    }
};



struct ParametricHyperbolicParaboloid : ParametricSurface
{
    const Real afac,bfac;

    ParametricHyperbolicParaboloid(const Real afac_in, const Real bfac_in):
    afac(afac_in),
    bfac(bfac_in)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        Eigen::Vector3d retval;
        retval << uu, vv, std::pow(uu/afac,2) - std::pow(vv/bfac,2);
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real ) const override
    {
        const Eigen::Vector3d xu = (Eigen::Vector3d() <<
                                    1.0,
                                    0.0,
                                    2.0*uu/(afac*afac)
                                    ).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real , const Real vv) const override
    {
        const Eigen::Vector3d xv = (Eigen::Vector3d() <<
                                    0.0,
                                    1.0,
                                    -2.0*vv/(bfac*bfac)
                                    ).finished();
        return xv;
    }

    Real getMeanCurvature(const Real uu, const Real vv) const override
    {
        const Real ufac = 4.0*uu*uu/std::pow(afac,4);
        const Real vfac = 4.0*vv*vv/std::pow(bfac,4);

        const Real asq = std::pow(afac,2);
        const Real bsq = std::pow(bfac,2);

        const Real denum = asq*bsq*std::pow(1.0 + ufac + vfac, 3.0/2.0);
        const Real num = -asq + bsq - 4.0*uu*uu/asq + 4.0*vv*vv/bsq;

        return num/denum;
    }

    Real getGaussCurvature(const Real uu, const Real vv) const override
    {
        const Real ufac = 4.0*uu*uu/std::pow(afac,4);
        const Real vfac = 4.0*vv*vv/std::pow(bfac,4);

        const Real denum = std::pow(afac * bfac * (1.0 + ufac + vfac), 2);
        return -4.0/denum;
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(-1.0, 1.0);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-1.0, 1.0);
    }
};



struct ParametricPartSphere : ParametricSurface
{
    const Real radius;

    ParametricPartSphere(const Real radius_in):
    radius(radius_in)
    {}

    Eigen::Vector3d operator()(const Real uu, const Real vv) const override
    {
        // stereographic projection
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        //const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1
        const Real scaleFac = (1+radius*radius)/(2*radius);

        const Real denum = 1 + urel*urel + vrel*vrel;
        const Real x = scaleFac * 2*urel/denum;
        const Real y = scaleFac * 2*vrel/denum;
        const Real z = scaleFac * (-1 + urel*urel + vrel*vrel)/denum + radius;

        Eigen::Vector3d retval;
        retval << x,y,z;
        return retval;
    }

    Eigen::Vector3d get_xu(const Real uu, const Real vv) const override
    {
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        //const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1
        const Real scaleFac = (1+radius*radius)/(2*radius);

        const Real denum = std::pow(1.0 + urel*urel + vrel*vrel,2);

        const Real xu_x = scaleFac * 2*(1 - urel*urel + vrel*vrel) / denum;
        const Real xu_y = scaleFac * -4*urel*vrel / denum;
        const Real xu_z = scaleFac * 4*urel/ denum;

        const Eigen::Vector3d xu = (Eigen::Vector3d() << xu_x, xu_y, xu_z).finished();
        return xu;
    }

    Eigen::Vector3d get_xv(const Real uu, const Real vv) const override
    {
        const Real urel = uu/radius;
        const Real vrel = vv/radius;

        //const Real scaleFac = scaleBaseToOne ? (1+radius*radius)/(2*radius) : 1.0; // so that xmax/xmin are within +1/-1
        const Real scaleFac = (1+radius*radius)/(2*radius);

        const Real denum = std::pow(1.0 + urel*urel + vrel*vrel,3);

        const Real xv_x = scaleFac * -4*urel*vrel / denum;
        const Real xv_y = scaleFac * 2*(1 + urel*urel - vrel*vrel) / denum;
        const Real xv_z = scaleFac * 4*vrel/ denum;

        const Eigen::Vector3d xv = (Eigen::Vector3d() << xv_x, xv_y, xv_z).finished();
        return xv;
    }

    Real getMeanCurvature(const Real uu, const Real vv) const override
    {
        return 1.0/radius;
    }

    Real getGaussCurvature(const Real uu, const Real vv) const override
    {
        return  1.0/(radius*radius);
    }

    std::pair<Real,Real> getExtent_U() const override
    {
        return std::make_pair(-1.0, 1.0);
    }

    std::pair<Real,Real> getExtent_V() const override
    {
        return std::make_pair(-1.0, 1.0);
    }
};
#endif /* ParametricSurfaceLibrary_hpp */
