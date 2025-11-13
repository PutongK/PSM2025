//
//  QuadraticFormOperator.cpp
//  Elasticity
//
//  Created by Wim van Rees on 1/30/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#include "QuadraticFormOperator.hpp"
#include "TriangleInfo.hpp"
#include "ExtendedTriangleInfo.hpp"
#include "MergePerFaceQuantities.hpp"

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

struct GradSurfaceDistance
{
    Eigen::Vector3d gradv0_norm;
    Eigen::Vector3d gradv1_norm;
    Eigen::Vector3d gradv2_norm;
    
    Eigen::Vector3d gradv_other_e0_norm;
    Eigen::Vector3d gradv_other_e1_norm;
    Eigen::Vector3d gradv_other_e2_norm;
    
    Real gradphi_e0_norm;
    Real gradphi_e1_norm;
    Real gradphi_e2_norm;
};

struct SurfaceDistanceHelper
{
    const Eigen::Matrix2d & firstFF_target, secondFF_target;
    const Eigen::Matrix2d & firstFF, secondFF;
    
    Eigen::Matrix2d shapeOp_target, shapeOp, firstFF_inv;
    
    SurfaceDistanceHelper(const Eigen::Matrix2d & firstFF_target, const Eigen::Matrix2d & secondFF_target, const Eigen::Matrix2d & firstFF, const Eigen::Matrix2d & secondFF):
    firstFF_target(firstFF_target),
    secondFF_target(secondFF_target),
    firstFF(firstFF),
    secondFF(secondFF)
    {
    }
    
    Real computeDistance()
    {
        shapeOp_target = firstFF_target.inverse() * secondFF_target;
        
        firstFF_inv = firstFF.inverse();
        shapeOp =  firstFF_inv * secondFF;
        
        return (shapeOp - shapeOp_target).squaredNorm();
    }
    
    GradSurfaceDistance computeGradient(const QuadraticFormGradientData_Verts & grad_firstFF, const QuadraticFormGradientData & grad_secondFF) const
    {
        const Eigen::Matrix2d shapeOp_diff = shapeOp - shapeOp_target;
        GradSurfaceDistance retval;
        
        for(int d=0;d<3;++d)
        {
            const Eigen::Matrix2d gradv0_firstFF = (Eigen::Matrix2d() << grad_firstFF.gradv0_11(d), grad_firstFF.gradv0_12(d), grad_firstFF.gradv0_12(d), grad_firstFF.gradv0_22(d)).finished();
            const Eigen::Matrix2d gradv1_firstFF = (Eigen::Matrix2d() << grad_firstFF.gradv1_11(d), grad_firstFF.gradv1_12(d), grad_firstFF.gradv1_12(d), grad_firstFF.gradv1_22(d)).finished();
            const Eigen::Matrix2d gradv2_firstFF = (Eigen::Matrix2d() << grad_firstFF.gradv2_11(d), grad_firstFF.gradv2_12(d), grad_firstFF.gradv2_12(d), grad_firstFF.gradv2_22(d)).finished();

            const Eigen::Matrix2d gradv0_firstFF_inv = - firstFF_inv * gradv0_firstFF * firstFF_inv;
            const Eigen::Matrix2d gradv1_firstFF_inv = - firstFF_inv * gradv1_firstFF * firstFF_inv;
            const Eigen::Matrix2d gradv2_firstFF_inv = - firstFF_inv * gradv2_firstFF * firstFF_inv;
            
            const Eigen::Matrix2d gradv0_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv0_11(d), grad_secondFF.gradv0_12(d), grad_secondFF.gradv0_12(d), grad_secondFF.gradv0_22(d)).finished();
            const Eigen::Matrix2d gradv1_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv1_11(d), grad_secondFF.gradv1_12(d), grad_secondFF.gradv1_12(d), grad_secondFF.gradv1_22(d)).finished();
            const Eigen::Matrix2d gradv2_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv2_11(d), grad_secondFF.gradv2_12(d), grad_secondFF.gradv2_12(d), grad_secondFF.gradv2_22(d)).finished();
            
            const Eigen::Matrix2d gradv0_shapeOp = gradv0_firstFF_inv * secondFF + firstFF_inv * gradv0_secondFF;
            const Eigen::Matrix2d gradv1_shapeOp = gradv1_firstFF_inv * secondFF + firstFF_inv * gradv1_secondFF;
            const Eigen::Matrix2d gradv2_shapeOp = gradv2_firstFF_inv * secondFF + firstFF_inv * gradv2_secondFF;
            
            retval.gradv0_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv0_shapeOp(0,0) + shapeOp_diff(0,1) * gradv0_shapeOp(0,1) + shapeOp_diff(1,0) * gradv0_shapeOp(1,0) + shapeOp_diff(1,1) * gradv0_shapeOp(1,1) );
            retval.gradv1_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv1_shapeOp(0,0) + shapeOp_diff(0,1) * gradv1_shapeOp(0,1) + shapeOp_diff(1,0) * gradv1_shapeOp(1,0) + shapeOp_diff(1,1) * gradv1_shapeOp(1,1) );
            retval.gradv2_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv2_shapeOp(0,0) + shapeOp_diff(0,1) * gradv2_shapeOp(0,1) + shapeOp_diff(1,0) * gradv2_shapeOp(1,0) + shapeOp_diff(1,1) * gradv2_shapeOp(1,1) );
            
            // other vertices
            const Eigen::Matrix2d gradv_other_e0_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv_other_e0_11(d), grad_secondFF.gradv_other_e0_12(d), grad_secondFF.gradv_other_e0_12(d), grad_secondFF.gradv_other_e0_22(d)).finished();
            const Eigen::Matrix2d gradv_other_e1_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv_other_e1_11(d), grad_secondFF.gradv_other_e1_12(d), grad_secondFF.gradv_other_e1_12(d), grad_secondFF.gradv_other_e1_22(d)).finished();
            const Eigen::Matrix2d gradv_other_e2_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradv_other_e2_11(d), grad_secondFF.gradv_other_e2_12(d), grad_secondFF.gradv_other_e2_12(d), grad_secondFF.gradv_other_e2_22(d)).finished();
            
            // no derivative of firstFF wrt these guys
            const Eigen::Matrix2d gradv_other_e0_shapeOp = firstFF_inv * gradv_other_e0_secondFF;
            const Eigen::Matrix2d gradv_other_e1_shapeOp = firstFF_inv * gradv_other_e1_secondFF;
            const Eigen::Matrix2d gradv_other_e2_shapeOp = firstFF_inv * gradv_other_e2_secondFF;
            
            retval.gradv_other_e0_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv_other_e0_shapeOp(0,0) + shapeOp_diff(0,1) * gradv_other_e0_shapeOp(0,1) + shapeOp_diff(1,0) * gradv_other_e0_shapeOp(1,0) + shapeOp_diff(1,1) * gradv_other_e0_shapeOp(1,1) );
            retval.gradv_other_e1_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv_other_e1_shapeOp(0,0) + shapeOp_diff(0,1) * gradv_other_e1_shapeOp(0,1) + shapeOp_diff(1,0) * gradv_other_e1_shapeOp(1,0) + shapeOp_diff(1,1) * gradv_other_e1_shapeOp(1,1) );
            retval.gradv_other_e2_norm(d) = 2.0 * (shapeOp_diff(0,0) * gradv_other_e2_shapeOp(0,0) + shapeOp_diff(0,1) * gradv_other_e2_shapeOp(0,1) + shapeOp_diff(1,0) * gradv_other_e2_shapeOp(1,0) + shapeOp_diff(1,1) * gradv_other_e2_shapeOp(1,1) );
        }
        
        const Eigen::Matrix2d gradphi_e0_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradphi_e0_11, grad_secondFF.gradphi_e0_12, grad_secondFF.gradphi_e0_12, grad_secondFF.gradphi_e0_22).finished();
        const Eigen::Matrix2d gradphi_e1_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradphi_e1_11, grad_secondFF.gradphi_e1_12, grad_secondFF.gradphi_e1_12, grad_secondFF.gradphi_e1_22).finished();
        const Eigen::Matrix2d gradphi_e2_secondFF = (Eigen::Matrix2d() << grad_secondFF.gradphi_e2_11, grad_secondFF.gradphi_e2_12, grad_secondFF.gradphi_e2_12, grad_secondFF.gradphi_e2_22).finished();
        
        // no derivative of firstFF wrt these guys
        const Eigen::Matrix2d gradphi_e0_shapeOp = firstFF_inv * gradphi_e0_secondFF;
        const Eigen::Matrix2d gradphi_e1_shapeOp = firstFF_inv * gradphi_e1_secondFF;
        const Eigen::Matrix2d gradphi_e2_shapeOp = firstFF_inv * gradphi_e2_secondFF;
        
        retval.gradphi_e0_norm = 2.0 * (shapeOp_diff(0,0) * gradphi_e0_shapeOp(0,0) + shapeOp_diff(0,1) * gradphi_e0_shapeOp(0,1) + shapeOp_diff(1,0) * gradphi_e0_shapeOp(1,0) + shapeOp_diff(1,1) * gradphi_e0_shapeOp(1,1) );
        retval.gradphi_e1_norm = 2.0 * (shapeOp_diff(0,0) * gradphi_e1_shapeOp(0,0) + shapeOp_diff(0,1) * gradphi_e1_shapeOp(0,1) + shapeOp_diff(1,0) * gradphi_e1_shapeOp(1,0) + shapeOp_diff(1,1) * gradphi_e1_shapeOp(1,1) );
        retval.gradphi_e2_norm = 2.0 * (shapeOp_diff(0,0) * gradphi_e2_shapeOp(0,0) + shapeOp_diff(0,1) * gradphi_e2_shapeOp(0,1) + shapeOp_diff(1,0) * gradphi_e2_shapeOp(1,0) + shapeOp_diff(1,1) * gradphi_e2_shapeOp(1,1) );
        
        return retval;
    }
};

struct GradGrowthRate_Verts
{
    Eigen::Vector3d gradv0_rate;
    Eigen::Vector3d gradv1_rate;
    Eigen::Vector3d gradv2_rate;
};

struct GradGrowthRate : GradGrowthRate_Verts
{
    Eigen::Vector3d gradv_other_e0_rate;
    Eigen::Vector3d gradv_other_e1_rate;
    Eigen::Vector3d gradv_other_e2_rate;
    
    Real gradphi_e0_rate;
    Real gradphi_e1_rate;
    Real gradphi_e2_rate;
    
};

struct GrowthRateEnergyHelper
{
    const Real hfac;
    const Eigen::Matrix2d & firstFF, secondFF;
    const Eigen::Matrix2d aform_bar;
    const Eigen::Matrix2d aform_bar_inv;
    
    Eigen::Matrix2d aform;
    Eigen::Matrix2d delta;
    Real Dfac;
    
    GrowthRateEnergyHelper(const Real hfac, const Eigen::Matrix2d & firstFF, const Eigen::Matrix2d & secondFF, const Eigen::Matrix2d & aform_bar):
    hfac(hfac),
    firstFF(firstFF),
    secondFF(secondFF),
    aform_bar(aform_bar),
    aform_bar_inv(aform_bar.inverse())
    {}
    
    std::pair<Real, Real> computeRates()
    {
        aform = firstFF + hfac * secondFF;
        
        // compute the growth tensor
        delta = aform * aform_bar_inv;
        
        // compute the growth rates
        // eigenvalues are
        // 1/2 * [ (a11 + a22) pm sqrt( (a11 - a22)^2 + 4 a12^2 )
        // have to be larger than zero
        
        const Real Dfac_arg = std::pow(delta(0,0) - delta(1,1), 2) + 4 * delta(0,1) * delta(1,0) + std::numeric_limits<Real>::epsilon();
        Dfac = std::sqrt( Dfac_arg );

        const Real rate1 = 0.5 * (delta(0,0) + delta(1,1) + Dfac);
        const Real rate2 = 0.5 * (delta(0,0) + delta(1,1) - Dfac);

//        if(Dfac_arg < 0)
//            std::cout << "PROBLEM : EIGENVALUES ARE COMPLEX " << Dfac_arg << "\t" << rate1 << "\t" << rate2 << std::endl;
//        
//        if(rate1 < 0 or rate2 < 0)
//            std::cout << "PROBLEM : EIGENVALUES ARE NEGATIVE " << Dfac << "\t" << rate1 << "\t" << rate2 << std::endl;
        
        return std::make_pair(rate1, rate2);
    }
    
    std::pair<GradGrowthRate, GradGrowthRate> computeGradient(const QuadraticFormGradientData_Verts & grad_firstFF, const QuadraticFormGradientData & grad_secondFF) const
    {
        // compute the gradients : gradients of current guys wrt vertices
        const Eigen::Vector3d gradv0_aform_11 = grad_firstFF.gradv0_11 + hfac*grad_secondFF.gradv0_11;
        const Eigen::Vector3d gradv0_aform_12 = grad_firstFF.gradv0_12 + hfac*grad_secondFF.gradv0_12;
        const Eigen::Vector3d gradv0_aform_22 = grad_firstFF.gradv0_22 + hfac*grad_secondFF.gradv0_22;
        
        const Eigen::Vector3d gradv1_aform_11 = grad_firstFF.gradv1_11 + hfac*grad_secondFF.gradv1_11;
        const Eigen::Vector3d gradv1_aform_12 = grad_firstFF.gradv1_12 + hfac*grad_secondFF.gradv1_12;
        const Eigen::Vector3d gradv1_aform_22 = grad_firstFF.gradv1_22 + hfac*grad_secondFF.gradv1_22;
        
        const Eigen::Vector3d gradv2_aform_11 = grad_firstFF.gradv2_11 + hfac*grad_secondFF.gradv2_11;
        const Eigen::Vector3d gradv2_aform_12 = grad_firstFF.gradv2_12 + hfac*grad_secondFF.gradv2_12;
        const Eigen::Vector3d gradv2_aform_22 = grad_firstFF.gradv2_22 + hfac*grad_secondFF.gradv2_22;
        
        const Eigen::Vector3d gradv_other_e0_aform_11 = hfac*grad_secondFF.gradv_other_e0_11;
        const Eigen::Vector3d gradv_other_e0_aform_12 = hfac*grad_secondFF.gradv_other_e0_12;
        const Eigen::Vector3d gradv_other_e0_aform_22 = hfac*grad_secondFF.gradv_other_e0_22;
        
        const Eigen::Vector3d gradv_other_e1_aform_11 = hfac*grad_secondFF.gradv_other_e1_11;
        const Eigen::Vector3d gradv_other_e1_aform_12 = hfac*grad_secondFF.gradv_other_e1_12;
        const Eigen::Vector3d gradv_other_e1_aform_22 = hfac*grad_secondFF.gradv_other_e1_22;
        
        const Eigen::Vector3d gradv_other_e2_aform_11 = hfac*grad_secondFF.gradv_other_e2_11;
        const Eigen::Vector3d gradv_other_e2_aform_12 = hfac*grad_secondFF.gradv_other_e2_12;
        const Eigen::Vector3d gradv_other_e2_aform_22 = hfac*grad_secondFF.gradv_other_e2_22;
        
        const Real gradphi_e0_aform_11 = hfac*grad_secondFF.gradphi_e0_11;
        const Real gradphi_e0_aform_12 = hfac*grad_secondFF.gradphi_e0_12;
        const Real gradphi_e0_aform_22 = hfac*grad_secondFF.gradphi_e0_22;
        
        const Real gradphi_e1_aform_11 = hfac*grad_secondFF.gradphi_e1_11;
        const Real gradphi_e1_aform_12 = hfac*grad_secondFF.gradphi_e1_12;
        const Real gradphi_e1_aform_22 = hfac*grad_secondFF.gradphi_e1_22;
        
        const Real gradphi_e2_aform_11 = hfac*grad_secondFF.gradphi_e2_11;
        const Real gradphi_e2_aform_12 = hfac*grad_secondFF.gradphi_e2_12;
        const Real gradphi_e2_aform_22 = hfac*grad_secondFF.gradphi_e2_22;
        
        // then gradients of growth rates
        GradGrowthRate grad_rate1, grad_rate2;
        
        for(int d=0;d<3;++d)
        {
            const Eigen::Matrix2d gradv0_delta = (Eigen::Matrix2d() << gradv0_aform_11(d), gradv0_aform_12(d), gradv0_aform_12(d), gradv0_aform_22(d)).finished() * aform_bar_inv;
            const Eigen::Matrix2d gradv1_delta = (Eigen::Matrix2d() << gradv1_aform_11(d), gradv1_aform_12(d), gradv1_aform_12(d), gradv1_aform_22(d)).finished() * aform_bar_inv;
            const Eigen::Matrix2d gradv2_delta = (Eigen::Matrix2d() << gradv2_aform_11(d), gradv2_aform_12(d), gradv2_aform_12(d), gradv2_aform_22(d)).finished() * aform_bar_inv;
            
            const Real gradv0_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv0_delta(0,0) - gradv0_delta(1,1)) + 4.0 * (delta(0,1) * gradv0_delta(1,0) + gradv0_delta(0,1) * delta(1,0)) );
            const Real gradv1_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv1_delta(0,0) - gradv1_delta(1,1)) + 4.0 * (delta(0,1) * gradv1_delta(1,0) + gradv1_delta(0,1) * delta(1,0)) );
            const Real gradv2_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv2_delta(0,0) - gradv2_delta(1,1)) + 4.0 * (delta(0,1) * gradv2_delta(1,0) + gradv2_delta(0,1) * delta(1,0)) );
            
            grad_rate1.gradv0_rate(d) = 0.5 * (gradv0_delta(0,0) + gradv0_delta(1,1) + gradv0_Dfac);
            grad_rate2.gradv0_rate(d) = 0.5 * (gradv0_delta(0,0) + gradv0_delta(1,1) - gradv0_Dfac);
            
            grad_rate1.gradv1_rate(d) = 0.5 * (gradv1_delta(0,0) + gradv1_delta(1,1) + gradv1_Dfac);
            grad_rate2.gradv1_rate(d) = 0.5 * (gradv1_delta(0,0) + gradv1_delta(1,1) - gradv1_Dfac);
            
            grad_rate1.gradv2_rate(d) = 0.5 * (gradv2_delta(0,0) + gradv2_delta(1,1) + gradv2_Dfac);
            grad_rate2.gradv2_rate(d) = 0.5 * (gradv2_delta(0,0) + gradv2_delta(1,1) - gradv2_Dfac);
            
            const Eigen::Matrix2d gradv_other_e0_delta = (Eigen::Matrix2d() << gradv_other_e0_aform_11(d), gradv_other_e0_aform_12(d), gradv_other_e0_aform_12(d), gradv_other_e0_aform_22(d)).finished() * aform_bar_inv;
            const Eigen::Matrix2d gradv_other_e1_delta = (Eigen::Matrix2d() << gradv_other_e1_aform_11(d), gradv_other_e1_aform_12(d), gradv_other_e1_aform_12(d), gradv_other_e1_aform_22(d)).finished() * aform_bar_inv;
            const Eigen::Matrix2d gradv_other_e2_delta = (Eigen::Matrix2d() << gradv_other_e2_aform_11(d), gradv_other_e2_aform_12(d), gradv_other_e2_aform_12(d), gradv_other_e2_aform_22(d)).finished() * aform_bar_inv;
            
            const Real gradv_other_e0_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv_other_e0_delta(0,0) - gradv_other_e0_delta(1,1)) + 4.0 * (delta(0,1) * gradv_other_e0_delta(1,0) + gradv_other_e0_delta(0,1) * delta(1,0)) );
            const Real gradv_other_e1_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv_other_e1_delta(0,0) - gradv_other_e1_delta(1,1)) + 4.0 * (delta(0,1) * gradv_other_e1_delta(1,0) + gradv_other_e1_delta(0,1) * delta(1,0)) );
            const Real gradv_other_e2_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv_other_e2_delta(0,0) - gradv_other_e2_delta(1,1)) + 4.0 * (delta(0,1) * gradv_other_e2_delta(1,0) + gradv_other_e2_delta(0,1) * delta(1,0)) );
            
            grad_rate1.gradv_other_e0_rate(d) = 0.5 * (gradv_other_e0_delta(0,0) + gradv_other_e0_delta(1,1) + gradv_other_e0_Dfac);
            grad_rate2.gradv_other_e0_rate(d) = 0.5 * (gradv_other_e0_delta(0,0) + gradv_other_e0_delta(1,1) - gradv_other_e0_Dfac);
            
            grad_rate1.gradv_other_e1_rate(d) = 0.5 * (gradv_other_e1_delta(0,0) + gradv_other_e1_delta(1,1) + gradv_other_e1_Dfac);
            grad_rate2.gradv_other_e1_rate(d) = 0.5 * (gradv_other_e1_delta(0,0) + gradv_other_e1_delta(1,1) - gradv_other_e1_Dfac);
            
            grad_rate1.gradv_other_e2_rate(d) = 0.5 * (gradv_other_e2_delta(0,0) + gradv_other_e2_delta(1,1) + gradv_other_e2_Dfac);
            grad_rate2.gradv_other_e2_rate(d) = 0.5 * (gradv_other_e2_delta(0,0) + gradv_other_e2_delta(1,1) - gradv_other_e2_Dfac);
        }
        
        const Eigen::Matrix2d gradphi_e0_delta = (Eigen::Matrix2d() << gradphi_e0_aform_11, gradphi_e0_aform_12, gradphi_e0_aform_12, gradphi_e0_aform_22).finished() * aform_bar_inv;
        const Eigen::Matrix2d gradphi_e1_delta = (Eigen::Matrix2d() << gradphi_e1_aform_11, gradphi_e1_aform_12, gradphi_e1_aform_12, gradphi_e1_aform_22).finished() * aform_bar_inv;
        const Eigen::Matrix2d gradphi_e2_delta = (Eigen::Matrix2d() << gradphi_e2_aform_11, gradphi_e2_aform_12, gradphi_e2_aform_12, gradphi_e2_aform_22).finished() * aform_bar_inv;
        
        const Real gradphi_e0_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradphi_e0_delta(0,0) - gradphi_e0_delta(1,1)) + 4.0 * (delta(0,1) * gradphi_e0_delta(1,0) + gradphi_e0_delta(0,1) * delta(1,0)) );
        const Real gradphi_e1_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradphi_e1_delta(0,0) - gradphi_e1_delta(1,1)) + 4.0 * (delta(0,1) * gradphi_e1_delta(1,0) + gradphi_e1_delta(0,1) * delta(1,0)) );
        const Real gradphi_e2_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradphi_e2_delta(0,0) - gradphi_e2_delta(1,1)) + 4.0 * (delta(0,1) * gradphi_e2_delta(1,0) + gradphi_e2_delta(0,1) * delta(1,0)) );
        
        grad_rate1.gradphi_e0_rate = 0.5 * (gradphi_e0_delta(0,0) + gradphi_e0_delta(1,1) + gradphi_e0_Dfac);
        grad_rate2.gradphi_e0_rate = 0.5 * (gradphi_e0_delta(0,0) + gradphi_e0_delta(1,1) - gradphi_e0_Dfac);
        
        grad_rate1.gradphi_e1_rate = 0.5 * (gradphi_e1_delta(0,0) + gradphi_e1_delta(1,1) + gradphi_e1_Dfac);
        grad_rate2.gradphi_e1_rate = 0.5 * (gradphi_e1_delta(0,0) + gradphi_e1_delta(1,1) - gradphi_e1_Dfac);
        
        grad_rate1.gradphi_e2_rate = 0.5 * (gradphi_e2_delta(0,0) + gradphi_e2_delta(1,1) + gradphi_e2_Dfac);
        grad_rate2.gradphi_e2_rate = 0.5 * (gradphi_e2_delta(0,0) + gradphi_e2_delta(1,1) - gradphi_e2_Dfac);
        
        return std::make_pair(grad_rate1, grad_rate2);
    }
    
    
    std::pair<GradGrowthRate_Verts, GradGrowthRate_Verts> computeGradient_abar(const QuadraticFormGradientData_Verts & grad_abar) const
    {
        // then gradients of growth rates
        GradGrowthRate grad_rate1, grad_rate2;
        
        for(int d=0;d<3;++d)
        {
            const Eigen::Matrix2d gradv0_delta = aform * (- aform_bar_inv * (Eigen::Matrix2d() << grad_abar.gradv0_11(d), grad_abar.gradv0_12(d), grad_abar.gradv0_12(d), grad_abar.gradv0_22(d)).finished() * aform_bar_inv);
            const Eigen::Matrix2d gradv1_delta = aform * (- aform_bar_inv * (Eigen::Matrix2d() << grad_abar.gradv1_11(d), grad_abar.gradv1_12(d), grad_abar.gradv1_12(d), grad_abar.gradv1_22(d)).finished() * aform_bar_inv);
            const Eigen::Matrix2d gradv2_delta = aform * (- aform_bar_inv * (Eigen::Matrix2d() << grad_abar.gradv2_11(d), grad_abar.gradv2_12(d), grad_abar.gradv2_12(d), grad_abar.gradv2_22(d)).finished() * aform_bar_inv);
            
            const Real gradv0_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv0_delta(0,0) - gradv0_delta(1,1)) + 4.0 * (delta(0,1) * gradv0_delta(1,0) + gradv0_delta(0,1) * delta(1,0)) );
            const Real gradv1_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv1_delta(0,0) - gradv1_delta(1,1)) + 4.0 * (delta(0,1) * gradv1_delta(1,0) + gradv1_delta(0,1) * delta(1,0)) );
            const Real gradv2_Dfac = 0.5 / Dfac * (2.0 * (delta(0,0) - delta(1,1)) * (gradv2_delta(0,0) - gradv2_delta(1,1)) + 4.0 * (delta(0,1) * gradv2_delta(1,0) + gradv2_delta(0,1) * delta(1,0)) );
            
            grad_rate1.gradv0_rate(d) = 0.5 * (gradv0_delta(0,0) + gradv0_delta(1,1) + gradv0_Dfac);
            grad_rate2.gradv0_rate(d) = 0.5 * (gradv0_delta(0,0) + gradv0_delta(1,1) - gradv0_Dfac);
            
            grad_rate1.gradv1_rate(d) = 0.5 * (gradv1_delta(0,0) + gradv1_delta(1,1) + gradv1_Dfac);
            grad_rate2.gradv1_rate(d) = 0.5 * (gradv1_delta(0,0) + gradv1_delta(1,1) - gradv1_Dfac);
            
            grad_rate1.gradv2_rate(d) = 0.5 * (gradv2_delta(0,0) + gradv2_delta(1,1) + gradv2_Dfac);
            grad_rate2.gradv2_rate(d) = 0.5 * (gradv2_delta(0,0) + gradv2_delta(1,1) - gradv2_Dfac);
        }

        return std::make_pair(grad_rate1, grad_rate2);
    }

};


struct QuadraticFormMetricParameters
{
    const int exponent;
    const Real minBound;
    const Real maxBound;
    const Real weight_surface;
    const Real weight_rates;
    
    QuadraticFormMetricParameters(const int exp, const Real minB, const Real maxB, const Real wS, const Real wR):
    exponent(exp),
    minBound(minB),
    maxBound(maxB),
    weight_surface(wS),
    weight_rates(wR)
    {}
};


template<typename tMesh, typename tMaterialType, bool withRestConfig, bool withGradient>
struct ComputeQuadraticFormMetrics
{
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;
    typedef typename tMesh::tCurrentConfigData tCurrentConfigData;
    
    const MaterialProperties<tMaterialType>  & material_properties;
    const TopologyData & topology;
    const tCurrentConfigData & currentState;
    const tReferenceConfigData & restState;
    
    const QuadraticFormMetricParameters & params;
    const tVecMat2d & target_firstFF, target_secondFF;
    
    Eigen::Ref<Eigen::MatrixXd> per_face_gradients;
    
    Real energySum_surface;
    Real energySum_rates;
    
    
    ComputeQuadraticFormMetrics(const MaterialProperties<tMaterialType>  & mat_props_in, const TopologyData & topo_in, const tCurrentConfigData & currentState_in, const tReferenceConfigData & restState_in, const QuadraticFormMetricParameters & params_in, const tVecMat2d & target_firstFF_in, const tVecMat2d & target_secondFF_in, Eigen::Ref<Eigen::MatrixXd> per_face_gradients_in):
    material_properties(mat_props_in),
    topology(topo_in),
    currentState(currentState_in),
    restState(restState_in),
    params(params_in),
    target_firstFF(target_firstFF_in),
    target_secondFF(target_secondFF_in),
    per_face_gradients(per_face_gradients_in),
    energySum_surface(0.0),
    energySum_rates(0.0)
    {}
    
    // split constructor (dont need copy constructor)
    ComputeQuadraticFormMetrics(const ComputeQuadraticFormMetrics & c, tbb::split):
    material_properties(c.material_properties),
    topology(c.topology),
    currentState(c.currentState),
    restState(c.restState),
    params(c.params),
    target_firstFF(c.target_firstFF),
    target_secondFF(c.target_secondFF),
    per_face_gradients(c.per_face_gradients),
    energySum_surface(0.0),
    energySum_rates(0.0)
    {}
    
    void join(const ComputeQuadraticFormMetrics & j)
    {
        // join the energy
        energySum_surface += j.energySum_surface;
        energySum_rates += j.energySum_rates;
    }
    
    template<bool U = withGradient> typename std::enable_if<U, void>::type
    addGradients(const ExtendedTriangleInfo & info, const SurfaceDistanceHelper & surfacedist, const GrowthRateEnergyHelper & growthrates_bot, const GrowthRateEnergyHelper & growthrates_top, const Real rate1_bot_norm, const Real rate2_bot_norm, const Real rate1_top_norm, const Real rate2_top_norm, const Real targetArea, const Real areaFac)
    {
        const int expm = this->params.exponent - 1;
        const int i = info.face_idx;
        
        
        const QuadraticFormGradientData_Verts grad_firstFF = info.computeGradFirstFundamentalForm();
        const QuadraticFormGradientData grad_secondFF = info.computeGradSecondFundamentalForm();
        
        // surface distance gradients
        const GradSurfaceDistance gradSurfdist = surfacedist.computeGradient(grad_firstFF, grad_secondFF);
        
        for(int j=0;j<3;++j)
        {
            per_face_gradients(i, 3*0+j) += this->params.weight_surface * gradSurfdist.gradv0_norm(j) * targetArea;
            per_face_gradients(i, 3*1+j) += this->params.weight_surface * gradSurfdist.gradv1_norm(j) * targetArea;
            per_face_gradients(i, 3*2+j) += this->params.weight_surface * gradSurfdist.gradv2_norm(j) * targetArea;
        }
        
        // opposite vertex to each edge (if it exists)
        if(info.other_faces[0] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*3+j) += this->params.weight_surface * gradSurfdist.gradv_other_e0_norm(j) * targetArea;
        
        if(info.other_faces[1] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*4+j) += this->params.weight_surface * gradSurfdist.gradv_other_e1_norm(j) * targetArea;
        
        if(info.other_faces[2] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*5+j) += this->params.weight_surface * gradSurfdist.gradv_other_e2_norm(j) * targetArea;
        
        // edges
        per_face_gradients(i, 3*6+0) += this->params.weight_surface * gradSurfdist.gradphi_e0_norm * targetArea;
        per_face_gradients(i, 3*6+1) += this->params.weight_surface * gradSurfdist.gradphi_e1_norm * targetArea;
        per_face_gradients(i, 3*6+2) += this->params.weight_surface * gradSurfdist.gradphi_e2_norm * targetArea;
        
        // growth rate gradients
        const std::pair<GradGrowthRate, GradGrowthRate> gradRates_bot = growthrates_bot.computeGradient(grad_firstFF, grad_secondFF);
        const std::pair<GradGrowthRate, GradGrowthRate> gradRates_top = growthrates_top.computeGradient(grad_firstFF, grad_secondFF);
        
        const Real preFac = 2.0 / (this->params.maxBound - this->params.minBound) * this->params.exponent * this->params.weight_rates * areaFac;
        
        for(int j=0;j<3;++j)
        {
            per_face_gradients(i, 3*0+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv0_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv0_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv0_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv0_rate(j) );
            per_face_gradients(i, 3*1+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv1_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv1_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv1_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv1_rate(j) );
            per_face_gradients(i, 3*2+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv2_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv2_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv2_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv2_rate(j) );
        }
        
        // opposite vertex to each edge (if it exists)
        if(info.other_faces[0] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*3+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv_other_e0_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv_other_e0_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv_other_e0_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv_other_e0_rate(j) );
        
        if(info.other_faces[1] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*4+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv_other_e1_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv_other_e1_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv_other_e1_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv_other_e1_rate(j) );
        
        if(info.other_faces[2] != nullptr)
            for(int j=0;j<3;++j)
                per_face_gradients(i, 3*5+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradv_other_e2_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradv_other_e2_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradv_other_e2_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradv_other_e2_rate(j) );
        
        per_face_gradients(i, 3*6+0) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradphi_e0_rate + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradphi_e0_rate + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradphi_e0_rate + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradphi_e0_rate );
        per_face_gradients(i, 3*6+1) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradphi_e1_rate + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradphi_e1_rate + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradphi_e1_rate + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradphi_e1_rate );
        per_face_gradients(i, 3*6+2) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_bot.first.gradphi_e2_rate + std::pow(rate2_bot_norm, expm)*gradRates_bot.second.gradphi_e2_rate + std::pow(rate1_top_norm, expm)*gradRates_top.first.gradphi_e2_rate + std::pow(rate2_top_norm, expm)*gradRates_top.second.gradphi_e2_rate );
        
        if(withRestConfig)
        {
            const TriangleInfo rinfo = restState.getTriangleInfoLite(topology, i);
            
            // gradient wrt abar
            QuadraticFormGradientData_Verts gradabar;
            
            // gradient with respect to vertices
            
            // derivatives of e1.dot(e1)
            gradabar.gradv0_11.setZero();
            gradabar.gradv1_11 = -2.0*rinfo.e1;
            gradabar.gradv2_11 = +2.0*rinfo.e1;
            
            // derivatives of e1.dot(e2)
            gradabar.gradv0_12 =  rinfo.e1;
            gradabar.gradv1_12 = -rinfo.e2;
            gradabar.gradv2_12 =  rinfo.e2 - rinfo.e1;
            
            // derivatives of e2.dot(e2)
            gradabar.gradv0_22 = +2.0*rinfo.e2;
            gradabar.gradv1_22.setZero();
            gradabar.gradv2_22 = -2.0*rinfo.e2;

            const std::pair<GradGrowthRate_Verts, GradGrowthRate_Verts> gradRates_abar_bot = growthrates_bot.computeGradient_abar(gradabar);
            const std::pair<GradGrowthRate_Verts, GradGrowthRate_Verts> gradRates_abar_top = growthrates_top.computeGradient_abar(gradabar);
            
            // derivative wrt rates metric (appears both in area and the actual metric)
            //energy_rates += this->params.weight_rates * ( (std::pow(rate1_bot_norm, this->params.exponent) + std::pow(rate2_bot_norm, this->params.exponent))*area_bot + (std::pow(rate1_top_norm, this->params.exponent) + std::pow(rate2_top_norm, this->params.exponent)) );

            // in fact express everything in terms of abar_bot (should be consistent with the vertex-computed first fundamental form)
            const Eigen::Matrix2d aform_bar_bot = this->restState.template getFirstFundamentalForm<bottom>(i);
            
            const Eigen::Vector3d gradv0_area = 0.125 / areaFac * (aform_bar_bot(0,0) * gradabar.gradv0_22 + aform_bar_bot(1,1) * gradabar.gradv0_11 - 2.0 * aform_bar_bot(0,1) * gradabar.gradv0_12);
            const Eigen::Vector3d gradv1_area = 0.125 / areaFac * (aform_bar_bot(0,0) * gradabar.gradv1_22 + aform_bar_bot(1,1) * gradabar.gradv1_11 - 2.0 * aform_bar_bot(0,1) * gradabar.gradv1_12);
            const Eigen::Vector3d gradv2_area = 0.125 / areaFac * (aform_bar_bot(0,0) * gradabar.gradv2_22 + aform_bar_bot(1,1) * gradabar.gradv2_11 - 2.0 * aform_bar_bot(0,1) * gradabar.gradv2_12);
            
            const Real rate_area_prefac = this->params.weight_rates * ( std::pow(rate1_bot_norm, this->params.exponent) + std::pow(rate2_bot_norm, this->params.exponent) + std::pow(rate1_top_norm, this->params.exponent) + std::pow(rate2_top_norm, this->params.exponent) );
            
            // wrt area
            for(int j=0;j<3;++j)
            {
                per_face_gradients(i, 3*7+j) += rate_area_prefac * gradv0_area(j);
                per_face_gradients(i, 3*8+j) += rate_area_prefac * gradv1_area(j);
                per_face_gradients(i, 3*9+j) += rate_area_prefac * gradv2_area(j);
            }

            // wrt metric
            for(int j=0;j<3;++j)
            {
                per_face_gradients(i, 3*7+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_abar_bot.first.gradv0_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_abar_bot.second.gradv0_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_abar_top.first.gradv0_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_abar_top.second.gradv0_rate(j) );
                per_face_gradients(i, 3*8+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_abar_bot.first.gradv1_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_abar_bot.second.gradv1_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_abar_top.first.gradv1_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_abar_top.second.gradv1_rate(j) );
                per_face_gradients(i, 3*9+j) += preFac * (std::pow(rate1_bot_norm, expm)*gradRates_abar_bot.first.gradv2_rate(j) + std::pow(rate2_bot_norm, expm)*gradRates_abar_bot.second.gradv2_rate(j) + std::pow(rate1_top_norm, expm)*gradRates_abar_top.first.gradv2_rate(j) + std::pow(rate2_top_norm, expm)*gradRates_abar_top.second.gradv2_rate(j) );
            }
        }
    }
    
    template<bool U = withGradient> typename std::enable_if<!U, void>::type
    addGradients(const ExtendedTriangleInfo & , const SurfaceDistanceHelper & , const GrowthRateEnergyHelper & , const GrowthRateEnergyHelper & , const Real , const Real , const Real , const Real , const Real , const Real  )
    {
        // do nothing
    }
        
    virtual void operator()(const tbb::blocked_range<int> & face_range)
    {
        Real energy_surface = 0.0;
        Real energy_rates = 0.0;

        for (int i=face_range.begin(); i != face_range.end(); ++i)
        {
            const ExtendedTriangleInfo & info = this->currentState.getTriangleInfo(i);

            const Real thickness = this->material_properties.getFaceMaterial(i).getThickness();
            const Real hfac = 2.0 * thickness / 3.0;
            
            // compute the quadratic forms of this mesh
            const Eigen::Matrix2d firstFF = info.computeFirstFundamentalForm();
            const Eigen::Matrix2d secondFF = info.computeSecondFundamentalForm();
            
            // compute those for the rest configuration
            const Eigen::Matrix2d aform_bar_bot = this->restState.template getFirstFundamentalForm<bottom>(i);
            const Eigen::Matrix2d aform_bar_top = this->restState.template getFirstFundamentalForm<top>(i);
            
            const Real areaFac = 0.5*std::sqrt(aform_bar_bot.determinant());
            //const Real area_bot = 0.5*std::sqrt(aform_bar_bot.determinant());
            //const Real area_top = 0.5*std::sqrt(aform_bar_top.determinant());
            //const Real area_avg = 0.5*(area_bot + area_top);
            
            // surface distance
            SurfaceDistanceHelper surfacedist(target_firstFF[i], target_secondFF[i], firstFF, secondFF);
            const Real surfacedist_metric = surfacedist.computeDistance();
            const Real targetArea = 0.5*std::sqrt(target_firstFF[i].determinant());
            
            energy_surface += this->params.weight_surface * surfacedist_metric * targetArea;
            
            // growth rate
            GrowthRateEnergyHelper growthrates_bot(-hfac, firstFF, secondFF, aform_bar_bot);
            GrowthRateEnergyHelper growthrates_top(+hfac, firstFF, secondFF, aform_bar_top);
            
            const std::pair<Real, Real> rates_bot = growthrates_bot.computeRates();
            const std::pair<Real, Real> rates_top = growthrates_top.computeRates();
            
            // normalize rates so that min and max fall on -1 and 1
            const Real rate1_bot_norm = 2.0 * (rates_bot.first  - this->params.minBound) / (this->params.maxBound - this->params.minBound) - 1;
            const Real rate2_bot_norm = 2.0 * (rates_bot.second - this->params.minBound) / (this->params.maxBound - this->params.minBound) - 1;
            
            const Real rate1_top_norm = 2.0 * (rates_top.first  - this->params.minBound) / (this->params.maxBound - this->params.minBound) - 1;
            const Real rate2_top_norm = 2.0 * (rates_top.second - this->params.minBound) / (this->params.maxBound - this->params.minBound) - 1;
            
            energy_rates += this->params.weight_rates * ( std::pow(rate1_bot_norm, this->params.exponent) + std::pow(rate2_bot_norm, this->params.exponent) + std::pow(rate1_top_norm, this->params.exponent) + std::pow(rate2_top_norm, this->params.exponent) ) * areaFac;
            
            addGradients(info, surfacedist, growthrates_bot, growthrates_top, rate1_bot_norm, rate2_bot_norm, rate1_top_norm, rate2_top_norm, targetArea, areaFac);

        }
        
        this->energySum_surface += energy_surface;
        this->energySum_rates += energy_rates;
    }
};


struct MergeGradVertices_Rest
{
    const TopologyData & topo;
    const Eigen::Ref<const Eigen::MatrixXd> per_face_gradients;
    Eigen::Ref<Eigen::MatrixXd> gradient_vertices;
    
    MergeGradVertices_Rest(const TopologyData & topo, const Eigen::Ref<const Eigen::MatrixXd> pfg, Eigen::Ref<Eigen::MatrixXd> g):
    topo(topo),
    per_face_gradients(pfg),
    gradient_vertices(g)
    {}
    
    MergeGradVertices_Rest(const MergeGradVertices_Rest & c, tbb::split):
    topo(c.topo),
    per_face_gradients(c.per_face_gradients),
    gradient_vertices(c.gradient_vertices)
    {}
    
    void join(const MergeGradVertices_Rest & )
    {
        //        nothing here : but it has to be here to allow parallel_reduce, which in turn has to be to allow a non-const operator() (we technically are const but we change a const & in the form of const Eigen::Ref<const Eigen::...> ). Same for MergeGradEdges below.
    }
    
    void operator () (const tbb::blocked_range<int>& vertex_range)
    {
        const auto face2vertices = topo.getFace2Vertices();
        const auto & vertex2faces = topo.getVertex2Faces();
        
        for (int i=vertex_range.begin(); i != vertex_range.end(); ++i)
        {
            const std::vector<int> & faces = vertex2faces[i];
            
            for(size_t f=0;f<faces.size();++f)
            {
                const int face_idx = faces[f];
                
                // figure out which vertex I am
                const int idx_v_rel = face2vertices(face_idx,0) == i ? 0 : (face2vertices(face_idx,1) == i ? 1 : 2);
                
                // ONLY DO THE XY-COORDINATES (IGNORE Z)
                for(int j=0;j<2;++j)
                    gradient_vertices(i,j) += per_face_gradients(face_idx, 3*(idx_v_rel + 7)+j); // add +7 because thats where the rest gradients are
            }
        }
    }
};


template<typename tMesh, typename tMaterialType>
Real QuadraticFormOperator<tMesh, tMaterialType>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, Eigen::Ref<Eigen::MatrixXd> gradient_restvertices, const bool computeGradient) const
{
    const auto & topology = mesh.getTopology();
    const auto & currentState = mesh.getCurrentConfiguration();
    const auto & restState = mesh.getRestConfiguration();
    
    const int nFaces = mesh.getNumberOfFaces();
    const QuadraticFormMetricParameters params(exponent, minBound, maxBound, weight_surface, weight_rates);
    Eigen::MatrixXd per_face_gradients;
    
    if(not computeGradient)
    {
        this->profiler.push_start("compute energy / gradient");

        ComputeQuadraticFormMetrics<tMesh, tMaterialType, false, false> compute_tbb(material_properties, topology, currentState, restState, params, target_firstFF, target_secondFF, per_face_gradients);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces), compute_tbb, tbb::auto_partitioner());
        
        this->profiler.pop_stop();
        lastEnergy = compute_tbb.energySum_surface + compute_tbb.energySum_rates;
    }
    else
    {
        const int nCols = restConfigGradients ? 30 : 21;
        per_face_gradients.resize(nFaces, nCols);
        per_face_gradients.setZero();
        
        this->profiler.push_start("compute energy / gradient");
        if(not restConfigGradients)
        {
            ComputeQuadraticFormMetrics<tMesh, tMaterialType, false, true> compute_tbb(material_properties, topology, currentState, restState, params, target_firstFF, target_secondFF, per_face_gradients);
            tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces), compute_tbb, tbb::auto_partitioner());
            
            lastEnergy = compute_tbb.energySum_surface + compute_tbb.energySum_rates;
        }
        else
        {
            ComputeQuadraticFormMetrics<tMesh, tMaterialType, true, true> compute_tbb(material_properties, topology, currentState, restState, params, target_firstFF, target_secondFF, per_face_gradients);
            tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces), compute_tbb, tbb::auto_partitioner());
            
            lastEnergy = compute_tbb.energySum_surface + compute_tbb.energySum_rates;
        }
        this->profiler.pop_stop();
        

        
        // merge
        const auto & topo = mesh.getTopology();
        const auto & boundaryConditions = mesh.getBoundaryConditions();
        
        const int nVertices = currentState.getNumberOfVertices();
        this->profiler.push_start("merge vertices");
        MergeGradVertices mergevertices(topo, boundaryConditions, per_face_gradients, gradient_vertices);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nVertices),mergevertices, tbb::auto_partitioner());
        this->profiler.pop_stop();
        
        const int nEdges = topo.getNumberOfEdges();
        this->profiler.push_start("merge edges");
        MergeGradEdges mergeedges(topo, boundaryConditions, per_face_gradients, gradient_edges);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nEdges),mergeedges, tbb::auto_partitioner());
        this->profiler.pop_stop();
        
        if(restConfigGradients)
        {
            this->profiler.push_start("merge rest vertices");
            MergeGradVertices_Rest mergevertices_rest(topo, per_face_gradients, gradient_restvertices);
            tbb::parallel_reduce(tbb::blocked_range<int>(0,nVertices), mergevertices_rest, tbb::auto_partitioner());
            this->profiler.pop_stop();
            
        }
    }
    
    return lastEnergy;
}

// explicit instantiations
#include "Mesh.hpp"
template class QuadraticFormOperator<BilayerMesh, Material_Isotropic>;
template class ComputeQuadraticFormMetrics<BilayerMesh, Material_Isotropic, false, false>;
template class ComputeQuadraticFormMetrics<BilayerMesh, Material_Isotropic, false, true>;
template class ComputeQuadraticFormMetrics<BilayerMesh, Material_Isotropic, true, true>;

