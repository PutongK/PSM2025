//
//  EnergyHelper_Parametric_Inverse.hpp
//  Elasticity
//
//  Created by Wim van Rees on 7/6/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef EnergyHelper_Parametric_Inverse_hpp
#define EnergyHelper_Parametric_Inverse_hpp

#include "common.hpp"
#include "EnergyHelper_Parametric.hpp"

struct EnergyHelper_Parametric_InverseGrowth : EnergyHelper_Parametric
{
    std::array<Real, 3> grad_area_fac;
    std::array<Eigen::Matrix2d, 3> aform_bar_inv_fac;
    
    EnergyHelper_Parametric_InverseGrowth(const Material_Isotropic & matprop, const Eigen::Matrix2d & aform_bar):
    EnergyHelper_Parametric(matprop, aform_bar)
    {
        const Eigen::Matrix2d grad_aform_bar_11 = (Eigen::Matrix2d() << 1,0,0,0).finished();
        const Eigen::Matrix2d grad_aform_bar_12 = (Eigen::Matrix2d() << 0,1,1,0).finished();
        const Eigen::Matrix2d grad_aform_bar_22 = (Eigen::Matrix2d() << 0,0,0,1).finished();
        
        // leave out the last aform^{-1} because we will do
        // if A = abar^{-1} B, then grad_{abar}A = -abar^{-1} grad_{abar}(abar) abar^{-1} B = -abar^{-1} grad_{abar}(abar) A
        // so here we store only the -abar^{-1} grad_{abar}(abar) part
        aform_bar_inv_fac[0] = -aform_bar_inv * grad_aform_bar_11;
        aform_bar_inv_fac[1] = -aform_bar_inv * grad_aform_bar_12;
        aform_bar_inv_fac[2] = -aform_bar_inv * grad_aform_bar_22;
        
        
        // grad determinant. full expression =
        //        const Real grad_det =
        //        aform_bar(0,0)*grad_aform_bar(1,1) + aform_bar(1,1)*grad_aform_bar(0,0) -
        //        aform_bar(0,1)*grad_aform_bar(1,0) - aform_bar(1,0)*grad_aform_bar(0,1);
        // but we can simply take out the components we want given the above definitions of grad_aform_bar_ij
        const Eigen::Vector3d grad_det = (Eigen::Vector3d() << aform_bar(1,1), -aform_bar(0,1) - aform_bar(1,0), aform_bar(0,0)).finished();
        
        //        grad_area = 1/4sqrt[x] d/dx = 1/8 * 1/(1/2 sqrt[x]) d/dx = 1/8 * 1/area d/dx
        const Real grad11_area = 0.125/area * grad_det(0);
        const Real grad12_area = 0.125/area * grad_det(1);
        const Real grad22_area = 0.125/area * grad_det(2);
        
        // area fac : we divide by area (to cancel out the original contribution) and multiply by gradarea
        grad_area_fac[0] = grad11_area / area;
        grad_area_fac[1] = grad12_area / area;
        grad_area_fac[2] = grad22_area / area;
    }
};


struct HessianHelper
{
    const EnergyHelper_Parametric_InverseGrowth & helper;
    
    const Real hfac;
    const StrainData & straindata;
    const EnergyNormData & energydata;
    const std::array<Eigen::Matrix2d, 3> gradabar_strain_all;
    std::array<Real, 3> gradabar_strain_trace_all;
    
    HessianHelper(const EnergyHelper_Parametric_InverseGrowth & helper_in, const Real hfac_in, const StrainData & straindata_in, const EnergyNormData & energydata_in, const std::array<Eigen::Matrix2d, 3> gradabar_strain_all_in):
    helper(helper_in),
    hfac(hfac_in),
    straindata(straindata_in),
    energydata(energydata_in),
    gradabar_strain_all(gradabar_strain_all_in)
    {
        for(int d=0;d<3;++d)
            gradabar_strain_trace_all[d] = gradabar_strain_all[d].trace(); // grad_a[trace]
        
    }
    
    template<int idx_abar>
    Eigen::Vector3d computeGradientsPerVertexPerAbar(const std::array<Eigen::Matrix2d, 3> & grad_strain_v, const Eigen::Vector3d & grad_trace_v, const Eigen::Vector3d & gradeng_v) const
    {
        /*
         We define strain as: strain = abar^{-1} A where A is an arbitrary matrix
         further we have trace = strain.trace, and trace_sq = (strain * strain).trace()
         and we have an energy density defined as E = hfac * (matfac_1 * trace^2 + matfac_2 * trace_sq) * area
         
         now the gradient of the strain wrt vertices is already computed and stored in grad_strain_v for each vertex
         in particular, this is defined so that grad_strain_v[0] is the matrix corresponding to the first (x) coordinate of that vertex position, where the matrix contains the derivative of each matrix entry wrt that coordinate of the vertex (symmetric matrix) --> grad_strain_v[2][1] = d/dz (strain_12) (if z is vertex coordinate 2)
         the gradient of the trace wrt this particular vertex is stored in grad_trace_v (each row d is the trace of grad_strain_v[d])
         
         finally we want to take the gradient wrt one component of abar (could be 11, 12 or 22) of the gradient of the energy density
         we have
         grad_v[E] = hfac * (matfac_1 * 2.0 * trace * grad_v[trace] + matfac_2 * grad_v[trace_sq]) * area
         taking the derivative wrt one of the components of abar gives
         
         grad_a[grad_v[E]] = hfac * (matfac_1 * 2.0 * grad_a[ trace * grad_v[trace]] + matfac_2 * grad_a[grad_v[trace_sq]]) * area + hfac * (matfac_1 * trace + matfac_2 * trace_sq) * grad_a[area] (a)
         
         the complex terms occur in the first part of expression (a), so we treat them one-by-one
         1) grad_a [ trace * grad_v[trace]] = grad_a[trace] * grad_v[trace] + trace * grad_a[grad_v[trace]]
         since we take the trace only at the end we can write
         grad_a[trace] = Trace(grad_a[strain])
         grad_a[grad_v[trace]] = Trace(grad_a[grad_v[strain]])
         --> this part is being computed in computeGradientsPerVertexPerAbar_Term_a
         
         2) grad_a[grad_v[trace_sq]] = grad_a[ (strain * grad_strain_v + grad_strain_v * strain).trace ]
         if we defer taking the trace until the end we can write:
         grad_a[ strain * grad_strain_v + grad_strain_v * strain ] = grad_a[strain] * grad_strain_v + strain * grad_a[grad_strain_v] + grad_a[grad_strain_v] * strain + grad_strain_v * grad_a[strain]
         //we already have grad_a[strain] and grad_a[grad_strain_v] from (1) so we can just plug these in
         --> this part is also being computed in computeGradientsPerVertexPerAbar_Term_a
         
         The second part of expression (a) can be computed directly from the energy gradient, by multiplying with 1/area * grad_area (1/area cancels out original area, grad_area then gives the gradient wrt each component of abar)
         --> this part is also being computed in computeGradientsPerVertexPerAbar_Term_b
         
         finally : to prevent double work (since we call this routine 9 times at least) we precompute and store the vertex-independent things such as grad_a[strain] and grad_a[trace] in the constructor
         */
        
        return computeGradientsPerVertexPerAbar_Term_a<idx_abar>(grad_strain_v, grad_trace_v) + computeGradientsPerVertexPerAbar_Term_b<idx_abar>(gradeng_v);
    }
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_scalar(const Eigen::Matrix2d & grad_strain_e, const Real grad_trace_e, const Real gradeng_e) const
    {
        return computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_e, grad_trace_e) + computeGradientsPerVertexPerAbar_Term_b_scalar<idx_abar>(gradeng_e);
    }
    
    
    template<int idx_abar>
    Eigen::Vector3d computeGradientsPerVertexPerAbar_Term_a(const std::array<Eigen::Matrix2d, 3> & grad_strain_v, const Eigen::Vector3d & grad_trace_v) const
    {
        // here we compute hfac * (matfac_1 * 2.0 * grad_a[ trace * grad_v[trace]] + matfac_2 * grad_a[grad_v[trace_sq]]) * area
        
        //        // extract the relevant quantities
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Eigen::Matrix2d & aform_bar_inv_fac = helper.aform_bar_inv_fac[idx_abar];
        const Eigen::Matrix2d & gradabar_strain = gradabar_strain_all[idx_abar];
        const Real gradabar_trace = gradabar_strain_trace_all[idx_abar];
        
        // grad_abar (abar^{-1} grad_Q mat) = (grad_abar abar^{-1}) * grad_Q mat
        std::array<Eigen::Matrix2d, 3> gradabar_grad_strain_v; // grad_a[grad_v[strain]]
        
        for(int d=0;d<3;++d) gradabar_grad_strain_v[d] = aform_bar_inv_fac * grad_strain_v[d];
        const Eigen::Vector3d gradabar_grad_trace_v = (Eigen::Vector3d() << gradabar_grad_strain_v[0].trace(), gradabar_grad_strain_v[1].trace(), gradabar_grad_strain_v[2].trace()).finished(); // grad_a[grad_v[trace]]
        
        // Derivatives of first term of SV energy
        const Eigen::Vector3d gradabar_grad_term1 = 2.0 * (energydata.trace * gradabar_grad_trace_v + gradabar_trace * grad_trace_v);
        
        // grad_trace_sq (already known) and gradabar_grad_trace_sq
        std::array<Eigen::Matrix2d, 3> gradabar_grad_strain_sq;
        for(int d=0;d<3;++d) gradabar_grad_strain_sq[d] = gradabar_strain * grad_strain_v[d] + straindata.strain * gradabar_grad_strain_v[d] + gradabar_grad_strain_v[d] * straindata.strain + grad_strain_v[d] * gradabar_strain;
        const Eigen::Vector3d gradabar_grad_trace_sq = (Eigen::Vector3d() << gradabar_grad_strain_sq[0].trace(), gradabar_grad_strain_sq[1].trace(), gradabar_grad_strain_sq[2].trace()).finished();
        
        // Derivatives of second term of SV energy
        const Eigen::Vector3d gradabar_grad_term2 = gradabar_grad_trace_sq;
        
        // finally we use the gradabar_energy and combine the whole thing
        const Eigen::Vector3d gradabar_gradv = hfac * (helper.material_prefac_1 * gradabar_grad_term1 + helper.material_prefac_2 * gradabar_grad_term2)*helper.area;
        
        return gradabar_gradv;
        
        // NOTE : we could also over the components of the vertex, but might be bit slower
        //        Eigen::Vector3d retval;
        //        for(int d=0;d<3;++d)
        //            retval(d) = computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_v[d], grad_trace_v(d));
        //        return retval;
        
    }
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_Term_a_scalar(const Eigen::Matrix2d & grad_strain_e, const Real grad_trace_e) const
    {
        // here we compute hfac * (matfac_1 * 2.0 * grad_a[ trace * grad_v[trace]] + matfac_2 * grad_a[grad_v[trace_sq]]) * area
        
        // extract the relevant quantities
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Eigen::Matrix2d & aform_bar_inv_fac = helper.aform_bar_inv_fac[idx_abar];
        const Eigen::Matrix2d & gradabar_strain = gradabar_strain_all[idx_abar];
        const Real gradabar_trace = gradabar_strain_trace_all[idx_abar];
        
        // grad_abar (abar^{-1} grad_Q mat) = (grad_abar abar^{-1}) * grad_Q mat
        const Eigen::Matrix2d gradabar_grad_strain_e = aform_bar_inv_fac * grad_strain_e; // grad_a[grad_v[strain]]
        const Real gradabar_grad_trace_e = gradabar_grad_strain_e.trace();
        
        // Derivatives of first term of SV energy
        const Real gradabar_grad_term1 = 2.0 * (energydata.trace * gradabar_grad_trace_e + gradabar_trace * grad_trace_e);
        
        // grad_trace_sq (already known) and gradabar_grad_trace_sq
        const Eigen::Matrix2d gradabar_grad_strain_sq = gradabar_strain * grad_strain_e + straindata.strain * gradabar_grad_strain_e + gradabar_grad_strain_e * straindata.strain + grad_strain_e * gradabar_strain;
        const Real gradabar_grad_trace_sq = gradabar_grad_strain_sq.trace();
        
        // Derivatives of second term of SV energy
        const Real gradabar_grad_term2 = gradabar_grad_trace_sq;
        
        // finally we use the gradabar_energy and combine the whole thing
        const Real gradabar_grade = hfac * (helper.material_prefac_1 * gradabar_grad_term1 + helper.material_prefac_2 * gradabar_grad_term2)*helper.area;
        
        return gradabar_grade;
    }
    
    template<int idx_abar>
    Eigen::Vector3d computeGradientsPerVertexPerAbar_Term_b(const Eigen::Vector3d & gradeng_v) const
    {
        // here we compute  hfac * (matfac_1 * trace + matfac_2 * trace_sq) * grad_a[area] = energy / area * grad_a[area] = energy * grad_area_fac
        
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Real grad_area_fac = helper.grad_area_fac[idx_abar];
        return gradeng_v * grad_area_fac;
    }
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_Term_b_scalar(const Real gradeng_v) const
    {
        // here we compute  hfac * (matfac_1 * trace + matfac_2 * trace_sq) * grad_a[area] = energy / area * grad_a[area] = energy * grad_area_fac
        
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Real grad_area_fac = helper.grad_area_fac[idx_abar];
        return gradeng_v * grad_area_fac;
    }
};


struct HessianHelper_Bilayer
{
    // constant terms
    const Real hfac;
    const HessianHelper & hesshelper_1;
    const HessianHelper & hesshelper_2;
    
    HessianHelper_Bilayer(const Real hfac_in, const HessianHelper & hesshelper_1_in, const HessianHelper & hesshelper_2_in):
    hfac(hfac_in),
    hesshelper_1(hesshelper_1_in),
    hesshelper_2(hesshelper_2_in)
    {
    }
    
    template<int idx_abar>
    Eigen::Vector3d computeGradientsPerVertexPerAbar(const std::array<Eigen::Matrix2d, 3> & grad_strain_1, const std::array<Eigen::Matrix2d, 3> & grad_strain_2, const Eigen::Vector3d & grad_trace_1, const Eigen::Vector3d & grad_trace_2, const Eigen::Vector3d & gradeng_v) const
    {
        Eigen::Vector3d retval;
        // first term
        for(int d=0;d<3;++d)
            retval(d) = computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_1[d], grad_strain_2[d], grad_trace_1(d), grad_trace_2(d));
        // second term
        retval += hesshelper_1.computeGradientsPerVertexPerAbar_Term_b<idx_abar>(gradeng_v);
        
        return retval;
    }
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_scalar(const Eigen::Matrix2d & grad_strain_1, const Eigen::Matrix2d & grad_strain_2, const Real grad_trace_1, const Real grad_trace_2, const Real gradeng_e) const
    {
        return computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_1, grad_strain_2, grad_trace_1, grad_trace_2) + hesshelper_1.computeGradientsPerVertexPerAbar_Term_b_scalar<idx_abar>(gradeng_e);
    }
    
    // specialized method for when strain_1 has zero gradient
    template<int idx_abar>
    Eigen::Vector3d computeGradientsPerVertexPerAbar(const std::array<Eigen::Matrix2d, 3> & grad_strain_2, const Eigen::Vector3d & grad_trace_2, const Eigen::Vector3d & gradeng_v) const
    {
        Eigen::Vector3d retval;
        // first term
        for(int d=0;d<3;++d)
            retval(d) = computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_2[d], grad_trace_2(d));
        // second term
        retval += hesshelper_1.computeGradientsPerVertexPerAbar_Term_b<idx_abar>(gradeng_v);
        
        return retval;
    }
    
    // specialized method for when strain_1 has zero gradient
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_scalar(const Eigen::Matrix2d & grad_strain_2, const Real grad_trace_2, const Real gradeng_e) const
    {
        return computeGradientsPerVertexPerAbar_Term_a_scalar<idx_abar>(grad_strain_2, grad_trace_2) + hesshelper_1.computeGradientsPerVertexPerAbar_Term_b_scalar<idx_abar>(gradeng_e);
    }
    
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_Term_a_scalar(const Eigen::Matrix2d & grad_strain_1, const Eigen::Matrix2d & grad_strain_2, const Real grad_trace_1, const Real grad_trace_2) const
    {
        // here we compute hfac * (matfac_1 * 2.0 * grad_a[ trace * grad_v[trace]] + matfac_2 * grad_a[grad_v[trace_sq]]) * area
        
        const EnergyHelper_Parametric_InverseGrowth & helper = hesshelper_1.helper; // does not matter if pick helper_1 or helper_2
        
        // extract the relevant quantities
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Eigen::Matrix2d & aform_bar_inv_fac = helper.aform_bar_inv_fac[idx_abar];
        const Eigen::Matrix2d & gradabar_strain_1 = hesshelper_1.gradabar_strain_all[idx_abar];
        const Eigen::Matrix2d & gradabar_strain_2 = hesshelper_2.gradabar_strain_all[idx_abar];
        const Real gradabar_trace_1 = hesshelper_1.gradabar_strain_trace_all[idx_abar];
        const Real gradabar_trace_2 = hesshelper_2.gradabar_strain_trace_all[idx_abar];
        
        // grad_abar (abar^{-1} grad_Q mat) = (grad_abar abar^{-1}) * grad_Q mat
        const Eigen::Matrix2d gradabar_grad_strain_1 = aform_bar_inv_fac * grad_strain_1; // grad_a[grad_v[strain]]
        const Eigen::Matrix2d gradabar_grad_strain_2 = aform_bar_inv_fac * grad_strain_2; // grad_a[grad_v[strain]]
        const Real gradabar_grad_trace_1 = gradabar_grad_strain_1.trace();
        const Real gradabar_grad_trace_2 = gradabar_grad_strain_2.trace();
        
        // Derivatives of first term of SV energy
        const Real gradabar_grad_term1 = hesshelper_1.energydata.trace * gradabar_grad_trace_2 + gradabar_trace_1 * grad_trace_2 + hesshelper_2.energydata.trace * gradabar_grad_trace_1 + gradabar_trace_2 * grad_trace_1;
        
        // gradabar_grad_trace_12
        const Eigen::Matrix2d gradabar_grad_strain_12 = gradabar_strain_1 * grad_strain_2 + hesshelper_1.straindata.strain * gradabar_grad_strain_2 + gradabar_grad_strain_1 * hesshelper_2.straindata.strain + grad_strain_1 * gradabar_strain_2;
        const Real gradabar_grad_trace_12 = gradabar_grad_strain_12.trace();
        
        // Derivatives of second term of SV energy
        const Real gradabar_grad_term2 = gradabar_grad_trace_12;
        
        // finally we use the gradabar_energy and combine the whole thing
        const Real gradabar_grade = hfac * (helper.material_prefac_1 * gradabar_grad_term1 + helper.material_prefac_2 * gradabar_grad_term2)*helper.area;
        
        return gradabar_grade;
    }
    
    template<int idx_abar>
    Real computeGradientsPerVertexPerAbar_Term_a_scalar(const Eigen::Matrix2d & grad_strain_2, const Real grad_trace_2) const // method where the firstFF gradients are zero!
    {
        // here we compute hfac * (matfac_1 * 2.0 * grad_a[ trace * grad_v[trace]] + matfac_2 * grad_a[grad_v[trace_sq]]) * area
        
        const EnergyHelper_Parametric_InverseGrowth & helper = hesshelper_1.helper; // does not matter if pick helper_1 or helper_2
        
        // extract the relevant quantities
        static_assert((idx_abar >= 0 && idx_abar < 3), "abar components can only be 0 (11), 1 (12), or 2 (22)");
        const Eigen::Matrix2d & aform_bar_inv_fac = helper.aform_bar_inv_fac[idx_abar];
        const Eigen::Matrix2d & gradabar_strain_1 = hesshelper_1.gradabar_strain_all[idx_abar];
        const Real gradabar_trace_1 = hesshelper_1.gradabar_strain_trace_all[idx_abar];
        
        // grad_abar (abar^{-1} grad_Q mat) = (grad_abar abar^{-1}) * grad_Q mat
        const Eigen::Matrix2d gradabar_grad_strain_2 = aform_bar_inv_fac * grad_strain_2; // grad_a[grad_v[strain]]
        const Real gradabar_grad_trace_2 = gradabar_grad_strain_2.trace();
        
        // Derivatives of first term of SV energy
        const Real gradabar_grad_term1 = hesshelper_1.energydata.trace * gradabar_grad_trace_2 + gradabar_trace_1 * grad_trace_2;
        
        // gradabar_grad_trace_12
        const Eigen::Matrix2d gradabar_grad_strain_12 = gradabar_strain_1 * grad_strain_2 + hesshelper_1.straindata.strain * gradabar_grad_strain_2;
        const Real gradabar_grad_trace_12 = gradabar_grad_strain_12.trace();
        
        // Derivatives of second term of SV energy
        const Real gradabar_grad_term2 = gradabar_grad_trace_12;
        
        // finally we use the gradabar_energy and combine the whole thing
        const Real gradabar_grade = hfac * (helper.material_prefac_1 * gradabar_grad_term1 + helper.material_prefac_2 * gradabar_grad_term2)*helper.area;
        
        return gradabar_grade;
    }
};

template<typename tMaterialType, MeshLayer layer>
struct HessianHelper_Stretching : HessianHelper
{
    const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy;
    Eigen::Matrix3d hessv0_aa, hessv1_aa, hessv2_aa;
    
    static std::array<Eigen::Matrix2d, 3> computeGradAbarStrain(const EnergyHelper_Parametric_InverseGrowth & helper_in, const StrainData_Stretching<true> & straindata_stretching)
    {
        std::array<Eigen::Matrix2d, 3> retval;
        for(int d=0;d<3;++d)
        {
            // d/dabar[abar^{-1} a - I] = -abar^{-1} d/dabar[abar] abar^{-1} a = aform_bar_inv_fac * abar^{-1} a = aform_bar_inv_fac * (strain + I)
            retval[d] = helper_in.aform_bar_inv_fac[d]*(straindata_stretching.strain + Eigen::Matrix2d::Identity());
        }
        return retval;
    }
    
    HessianHelper_Stretching(const EnergyHelper_Parametric_InverseGrowth & helper_in, const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy_in):
    HessianHelper(helper_in, SVenergy_in.h_aa, SVenergy_in.straindata_stretching, SVenergy_in.energydata_stretching, computeGradAbarStrain(helper_in, SVenergy_in.straindata_stretching)),
    SVenergy(SVenergy_in)
    {}
    
    void compute_hessian()
    {
        const StrainData_Stretching<true> & straindata_stretching = SVenergy.straindata_stretching;
        const EnergyNormData_Stretching<true> & energydata_stretching = SVenergy.energydata_stretching;
        
        hessv0_aa.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), SVenergy.gradv0_aa);
        hessv0_aa.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), SVenergy.gradv0_aa);
        hessv0_aa.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), SVenergy.gradv0_aa);
        
        hessv1_aa.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), SVenergy.gradv1_aa);
        hessv1_aa.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), SVenergy.gradv1_aa);
        hessv1_aa.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), SVenergy.gradv1_aa);
        
        hessv2_aa.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), SVenergy.gradv2_aa);
        hessv2_aa.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), SVenergy.gradv2_aa);
        hessv2_aa.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), SVenergy.gradv2_aa);
    }
};

template<typename tMaterialType, MeshLayer layer>
struct HessianHelper_Bending : HessianHelper
{
    const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy;
    Eigen::Matrix3d hessv0_bb, hessv1_bb, hessv2_bb;
    Eigen::Matrix3d hessv_other_e0_bb, hessv_other_e1_bb, hessv_other_e2_bb;
    Eigen::Vector3d hessphi_e0_bb, hessphi_e1_bb, hessphi_e2_bb;
    
    static std::array<Eigen::Matrix2d, 3> computeGradAbarStrain(const EnergyHelper_Parametric_InverseGrowth & helper_in, const StrainData_Bending<true> & straindata_bending)
    {
        std::array<Eigen::Matrix2d, 3> retval;
        for(int d=0;d<3;++d)
        {
            // d/dabar[abar^{-1} b] = -abar^{-1} d/dabar[abar] abar^{-1} b = aform_bar_inv_fac * abar^{-1} b = aform_bar_inv_fac * strain
            retval[d] = helper_in.aform_bar_inv_fac[d]*straindata_bending.strain;
        }
        return retval;
    }
    
    HessianHelper_Bending(const EnergyHelper_Parametric_InverseGrowth & helper_in, const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy_in):
    HessianHelper(helper_in, SVenergy_in.h_bb, SVenergy_in.straindata_bending, SVenergy_in.energydata_bending, computeGradAbarStrain(helper_in, SVenergy_in.straindata_bending)),
    SVenergy(SVenergy_in)
    {}
    
    void compute_hessian()
    {
        const StrainData_Bending<true> & straindata_bending = SVenergy.straindata_bending;
        const EnergyNormData_Bending<true> & energydata_bending = SVenergy.energydata_bending;
        
        // our vertices
        hessv0_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v0, energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_bb);
        hessv0_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v0, energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_bb);
        hessv0_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v0, energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_bb);
        
        hessv1_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v1, energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_bb);
        hessv1_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v1, energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_bb);
        hessv1_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v1, energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_bb);
        
        hessv2_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v2, energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_bb);
        hessv2_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v2, energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_bb);
        hessv2_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v2, energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_bb);
        
        // other vertices
        hessv_other_e0_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_bb);
        hessv_other_e0_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_bb);
        hessv_other_e0_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_bb);
        
        hessv_other_e1_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_bb);
        hessv_other_e1_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_bb);
        hessv_other_e1_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_bb);
        
        hessv_other_e2_bb.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_bb);
        hessv_other_e2_bb.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_bb);
        hessv_other_e2_bb.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_bb);
        
        // edges
        hessphi_e0_bb(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_bb);
        hessphi_e0_bb(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_bb);
        hessphi_e0_bb(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_bb);
        
        hessphi_e1_bb(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_bb);
        hessphi_e1_bb(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_bb);
        hessphi_e1_bb(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_bb);
        
        hessphi_e2_bb(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_bb);
        hessphi_e2_bb(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_bb);
        hessphi_e2_bb(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_bb);
    }
};



template<typename tMaterialType, MeshLayer layer>
struct BilayerTerm_HessianHelper : HessianHelper_Bilayer
{
    const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy;
    
    Eigen::Matrix3d hessv0_ab, hessv1_ab, hessv2_ab;
    Eigen::Matrix3d hessv_other_e0_ab, hessv_other_e1_ab, hessv_other_e2_ab;
    Eigen::Vector3d hessphi_e0_ab, hessphi_e1_ab, hessphi_e2_ab;
    
    BilayerTerm_HessianHelper(const HessianHelper_Stretching<tMaterialType, layer> & hessian_stretching_in, const HessianHelper_Bending<tMaterialType, layer> & hessian_bending_in, const SaintVenantEnergy<true, tMaterialType, layer> & SVenergy_in):
    HessianHelper_Bilayer(SVenergy_in.h_ab, hessian_stretching_in, hessian_bending_in),
    SVenergy(SVenergy_in)
    {}
    
    void compute_hessian()
    {
        const StrainData_Stretching<true> & straindata_stretching = SVenergy.straindata_stretching;
        const EnergyNormData_Stretching<true> & energydata_stretching = SVenergy.energydata_stretching;
        
        const StrainData_Bending<true> & straindata_bending = SVenergy.straindata_bending;
        const EnergyNormData_Bending<true> & energydata_bending = SVenergy.energydata_bending;
        
        
        // our vertices
        hessv0_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v0, straindata_bending.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_ab);
        hessv0_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v0, straindata_bending.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_ab);
        hessv0_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v0, straindata_bending.grad_strain_v0, energydata_stretching.grad_trace_v.col(0), energydata_bending.grad_trace_v.col(0), SVenergy.gradv0_ab);
        
        hessv1_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v1, straindata_bending.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_ab);
        hessv1_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v1, straindata_bending.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_ab);
        hessv1_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v1, straindata_bending.grad_strain_v1, energydata_stretching.grad_trace_v.col(1), energydata_bending.grad_trace_v.col(1), SVenergy.gradv1_ab);
        
        hessv2_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_stretching.grad_strain_v2, straindata_bending.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_ab);
        hessv2_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_stretching.grad_strain_v2, straindata_bending.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_ab);
        hessv2_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_stretching.grad_strain_v2, straindata_bending.grad_strain_v2, energydata_stretching.grad_trace_v.col(2), energydata_bending.grad_trace_v.col(2), SVenergy.gradv2_ab);
        
        // other vertices
        hessv_other_e0_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_ab);
        hessv_other_e0_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_ab);
        hessv_other_e0_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e0, energydata_bending.grad_trace_v_other_e.col(0), SVenergy.gradv_other_e0_ab);
        
        hessv_other_e1_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_ab);
        hessv_other_e1_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_ab);
        hessv_other_e1_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e1, energydata_bending.grad_trace_v_other_e.col(1), SVenergy.gradv_other_e1_ab);
        
        hessv_other_e2_ab.col(0) = computeGradientsPerVertexPerAbar<0>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_ab);
        hessv_other_e2_ab.col(1) = computeGradientsPerVertexPerAbar<1>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_ab);
        hessv_other_e2_ab.col(2) = computeGradientsPerVertexPerAbar<2>(straindata_bending.grad_strain_v_other_e2, energydata_bending.grad_trace_v_other_e.col(2), SVenergy.gradv_other_e2_ab);
        
        // edges
        hessphi_e0_ab(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_ab);
        hessphi_e0_ab(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_ab);
        hessphi_e0_ab(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e0, energydata_bending.grad_trace_e(0), SVenergy.gradphi_e0_ab);
        
        hessphi_e1_ab(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_ab);
        hessphi_e1_ab(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_ab);
        hessphi_e1_ab(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e1, energydata_bending.grad_trace_e(1), SVenergy.gradphi_e1_ab);
        
        hessphi_e2_ab(0) = computeGradientsPerVertexPerAbar_scalar<0>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_ab);
        hessphi_e2_ab(1) = computeGradientsPerVertexPerAbar_scalar<1>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_ab);
        hessphi_e2_ab(2) = computeGradientsPerVertexPerAbar_scalar<2>(straindata_bending.grad_strain_e2, energydata_bending.grad_trace_e(2), SVenergy.gradphi_e2_ab);
    }
};

struct BaseData_Hessian
{
    const EnergyHelper_Parametric_InverseGrowth helper;
    
    Eigen::Vector3d energy_v0, energy_v1, energy_v2;
    Eigen::Vector3d energy_v_other_e0, energy_v_other_e1, energy_v_other_e2;
    Real energy_e0, energy_e1, energy_e2;
    
    BaseData_Hessian(const Material_Isotropic & mat_prop_in, const Eigen::Matrix2d & aform_bar):
    helper(mat_prop_in, aform_bar)
    {}
};

struct BaseData_Hessian_WithGradient : BaseData_Hessian
{
    Eigen::Matrix3d hessv0, hessv1, hessv2; // col 0 is grad_abar11_grad_v0, col 1 is grad_abar12_grad_v0, col 2 is grad_abar22_grad_v0
    Eigen::Matrix3d hessv_other_e0, hessv_other_e1, hessv_other_e2;
    Eigen::Vector3d hessphi_e0, hessphi_e1, hessphi_e2;
    
    BaseData_Hessian_WithGradient(const Material_Isotropic & mat_prop_in, const Eigen::Matrix2d & aform_bar):
    BaseData_Hessian(mat_prop_in, aform_bar)
    {}
};


template<bool withGradient, typename tMaterialType, MeshLayer layer>
struct SaintVenantEnergy_Inverse : std::conditional<withGradient, BaseData_Hessian_WithGradient, BaseData_Hessian>::type

{
};

template<bool withGradient, MeshLayer layer>
struct SaintVenantEnergy_Inverse<withGradient, Material_Isotropic, layer> : std::conditional<withGradient, BaseData_Hessian_WithGradient, BaseData_Hessian>::type
{
    typedef typename std::conditional<withGradient, BaseData_Hessian_WithGradient, BaseData_Hessian>::type HessianDataBase;
    typedef Material_Isotropic tMaterialType;
    
    SaintVenantEnergy<true, tMaterialType, layer> SVenergy;
    
    SaintVenantEnergy_Inverse(const Material_Isotropic & mat_prop_in, const Eigen::Matrix2d & aform_bar_in, const ExtendedTriangleInfo & info_in):
    HessianDataBase(mat_prop_in, aform_bar_in),
    SVenergy(mat_prop_in, aform_bar_in, info_in)
    {}
    
    
    SaintVenantEnergy_Inverse(const Material_Isotropic & mat_prop_in, const Eigen::Matrix2d & aform_bar_in, const Eigen::Matrix2d bform_bar_in, const ExtendedTriangleInfo & info_in):
    HessianDataBase(mat_prop_in, aform_bar_in),
    SVenergy(mat_prop_in, aform_bar_in, bform_bar_in, info_in)
    {}
    
    
    // ========== ENERGY COMPUTATION METHODS =========== //
    void compute_energy_single()
    {
        SVenergy.compute();
        SVenergy.compute_gradients();
        
        this->energy_v0 = SVenergy.gradv0_aa + SVenergy.gradv0_bb;
        this->energy_v1 = SVenergy.gradv1_aa + SVenergy.gradv1_bb;
        this->energy_v2 = SVenergy.gradv2_aa + SVenergy.gradv2_bb;
        
        this->energy_v_other_e0 = SVenergy.gradv_other_e0_bb;
        this->energy_v_other_e1 = SVenergy.gradv_other_e1_bb;
        this->energy_v_other_e2 = SVenergy.gradv_other_e2_bb;
        
        this->energy_e0 = SVenergy.gradphi_e0_bb;
        this->energy_e1 = SVenergy.gradphi_e1_bb;
        this->energy_e2 = SVenergy.gradphi_e2_bb;
    }
    
    template<MeshLayer L = layer> typename std::enable_if<L==single, void>::type
    compute()
    {
        compute_energy_single();
    }
    
    template<MeshLayer L = layer> typename std::enable_if<L!=single, void>::type
    compute()
    {
        compute_energy_single();
        
        this->energy_v0 += SVenergy.gradv0_ab;
        this->energy_v1 += SVenergy.gradv1_ab;
        this->energy_v2 += SVenergy.gradv2_ab;
        
        this->energy_v_other_e0 += SVenergy.gradv_other_e0_ab;
        this->energy_v_other_e1 += SVenergy.gradv_other_e1_ab;
        this->energy_v_other_e2 += SVenergy.gradv_other_e2_ab;
        
        this->energy_e0 += SVenergy.gradphi_e0_ab;
        this->energy_e1 += SVenergy.gradphi_e1_ab;
        this->energy_e2 += SVenergy.gradphi_e2_ab;
    }
    
    // ========== GRADIENT COMPUTATION METHODS =========== //
    template<bool U = withGradient, MeshLayer L = layer> typename std::enable_if<U && L==single, void>::type
    compute_gradients()
    {
        HessianHelper_Stretching<tMaterialType, L> hessian_stretching(this->helper, SVenergy);
        HessianHelper_Bending<tMaterialType, L> hessian_bending(this->helper, SVenergy);
        
        hessian_stretching.compute_hessian();
        hessian_bending.compute_hessian();
        
        this->hessv0 = hessian_stretching.hessv0_aa + hessian_bending.hessv0_bb;
        this->hessv1 = hessian_stretching.hessv1_aa + hessian_bending.hessv1_bb;
        this->hessv2 = hessian_stretching.hessv2_aa + hessian_bending.hessv2_bb;
        
        this->hessv_other_e0 = hessian_bending.hessv_other_e0_bb;
        this->hessv_other_e1 = hessian_bending.hessv_other_e1_bb;
        this->hessv_other_e2 = hessian_bending.hessv_other_e2_bb;
        
        this->hessphi_e0 = hessian_bending.hessphi_e0_bb;
        this->hessphi_e1 = hessian_bending.hessphi_e1_bb;
        this->hessphi_e2 = hessian_bending.hessphi_e2_bb;
    }
    
    template<bool U = withGradient, MeshLayer L = layer> typename std::enable_if<U && L!=single, void>::type
    compute_gradients()
    {
        HessianHelper_Stretching<tMaterialType, L> hessian_stretching(this->helper, SVenergy);
        HessianHelper_Bending<tMaterialType, L> hessian_bending(this->helper, SVenergy);
        BilayerTerm_HessianHelper<tMaterialType, L> hessian_bilayer(hessian_stretching, hessian_bending, SVenergy);
        
        hessian_stretching.compute_hessian();
        hessian_bending.compute_hessian();
        hessian_bilayer.compute_hessian();
        
        this->hessv0 = hessian_stretching.hessv0_aa + hessian_bending.hessv0_bb + hessian_bilayer.hessv0_ab;
        this->hessv1 = hessian_stretching.hessv1_aa + hessian_bending.hessv1_bb + hessian_bilayer.hessv1_ab;
        this->hessv2 = hessian_stretching.hessv2_aa + hessian_bending.hessv2_bb + hessian_bilayer.hessv2_ab;
        
        this->hessv_other_e0 = hessian_bending.hessv_other_e0_bb + hessian_bilayer.hessv_other_e0_ab;
        this->hessv_other_e1 = hessian_bending.hessv_other_e1_bb + hessian_bilayer.hessv_other_e1_ab;
        this->hessv_other_e2 = hessian_bending.hessv_other_e2_bb + hessian_bilayer.hessv_other_e2_ab;
        
        this->hessphi_e0 = hessian_bending.hessphi_e0_bb + hessian_bilayer.hessphi_e0_ab;
        this->hessphi_e1 = hessian_bending.hessphi_e1_bb + hessian_bilayer.hessphi_e1_ab;
        this->hessphi_e2 = hessian_bending.hessphi_e2_bb + hessian_bilayer.hessphi_e2_ab;
    }
};



#endif /* EnergyHelper_Parametric_Inverse_hpp */
