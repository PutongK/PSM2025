//
//  DirichletEnergyOperator_Inverse_v2.cpp
//  Elasticity
//
//  Created by Wim van Rees on 9/14/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "DirichletEnergyOperator_Inverse.hpp"
#include "TriangleInfo.hpp"
#include "TopologyData.hpp"
#include "BoundaryConditionsData.hpp"

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

#define INTEGRATE_OVER_MESH_AREA

template<typename tMesh, MeshLayer layer, bool withGradient>
struct ComputeDirichletEnergy_Inverse
{
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;
    typedef typename tMesh::tCurrentConfigData tCurrentConfigData;
    
    const Real prefactor;
    const TopologyData & topo;
    const tReferenceConfigData & restState;
    const Eigen::MatrixXd edgevectors;
    Eigen::Ref<Eigen::MatrixXd> gradient_abar;
    Real energy;
    
    ComputeDirichletEnergy_Inverse(const Real prefactor_in, const TopologyData & topo_in, const tReferenceConfigData & restState_in, Eigen::Ref<Eigen::MatrixXd> gradient_abar_in):
    prefactor(prefactor_in),
    topo(topo_in),
    restState(restState_in),
    edgevectors((Eigen::MatrixXd(2,3) <<
                                         1, 1, 0,
                                         1, 0, 1
                                         ).finished()),
    gradient_abar(gradient_abar_in),
    energy(0)
    {}
    
    // split constructor (dont need copy constructor)
    ComputeDirichletEnergy_Inverse(const ComputeDirichletEnergy_Inverse & c, tbb::split):
    prefactor(c.prefactor),
    topo(c.topo),
    restState(c.restState),
    edgevectors(c.edgevectors),
    gradient_abar(c.gradient_abar),
    energy(0)
    {}

    void join(const ComputeDirichletEnergy_Inverse & j)
    {
        // join the energy
        energy += j.energy;
    }
    
    template<MeshLayer L = layer> typename std::enable_if<L==single, Eigen::Matrix2d>::type
    getFirstFundamentalForm(const int i) const
    {
        return this->restState.getFirstFundamentalForm(i);
    }
    
    template<MeshLayer L = layer> typename std::enable_if<L!=single, Eigen::Matrix2d>::type
    getFirstFundamentalForm(const int i) const
    {
        return this->restState.template getFirstFundamentalForm<L>(i);
    }
    
    
    void operator()(const tbb::blocked_range<int> & edge_range)
    {
        const auto edge2faces = topo.getEdge2Faces();
        const auto face2edges = topo.getFace2Edges();
        
        Real energy_tmp = 0;
        
        for (int i=edge_range.begin(); i != edge_range.end(); ++i)
        {
            const int idx_f0 = edge2faces(i,0);
            const int idx_f1 = edge2faces(i,1);
            
            if(idx_f0 < 0 or idx_f1 < 0) continue; // dirichlet energy zero on boundary
            
            // get edgelengths from neighboring abars
            const int eidx_0 = (face2edges(idx_f0,0) == i ? 0 : (face2edges(idx_f0,1) == i ? 1 : 2));
            const int eidx_1 = (face2edges(idx_f1,0) == i ? 0 : (face2edges(idx_f1,1) == i ? 1 : 2));
            
            // get the edge vectors in the reference unit triangle frame
            const Eigen::Vector2d evec_0 = edgevectors.col(eidx_0);
            const Eigen::Vector2d evec_1 = edgevectors.col(eidx_1);
            
            // compute the edge lengths of the edge across the triangle
            const Eigen::Matrix2d & aform_bar_f0 = getFirstFundamentalForm(idx_f0);
            const Eigen::Matrix2d & aform_bar_f1 = getFirstFundamentalForm(idx_f1);
            
            const Real le0_sq = evec_0.transpose() * (aform_bar_f0 * evec_0);
            const Real le1_sq = evec_1.transpose() * (aform_bar_f1 * evec_1);
            
            // the following should hold if aform_bar is positive definite
            assert(le0_sq > 0);
            assert(le1_sq > 0);
            
            const Real le0 = std::sqrt(le0_sq);
            const Real le1 = std::sqrt(le1_sq);
            
            // compute the dirichlet metric
            //        const Real eng = prefactor * std::pow(le1 - le0, 2) / std::pow(le0 + le1, 2); // should never be zero unless both edges have zero length
            
            const Real nExp = 2;
            
            assert(nExp >= 1); // else kink when le0 == le1 --> derivative ~ 1/0
            
            const Real engFac = (le1 / le0 + le0 / le1 - 2); // only compare ratios
            const Real eng = prefactor * std::pow(engFac, nExp);
            
            
#ifdef INTEGRATE_OVER_MESH_AREA
            // multiply by the area of this edge
            const TriangleInfo info_f0 = restState.getTriangleInfoLite(topo, idx_f0);
            const TriangleInfo info_f1 = restState.getTriangleInfoLite(topo, idx_f1);
            
            const Real area_f0 = 0.5 * std::sqrt( (info_f0.e1).dot(info_f0.e1) * (info_f0.e2).dot(info_f0.e2) - std::pow( (info_f0.e1).dot(info_f0.e2) ,2) );
            const Real area_f1 = 0.5 * std::sqrt( (info_f1.e1).dot(info_f1.e1) * (info_f1.e2).dot(info_f1.e2) - std::pow( (info_f1.e1).dot(info_f1.e2) ,2) );
#else
            const Real aform_bar_f0_sqrtdet = std::sqrt(aform_bar_f0.determinant());
            const Real aform_bar_f1_sqrtdet = std::sqrt(aform_bar_f1.determinant());
            
            const Real area_f0 = 0.5*aform_bar_f0_sqrtdet;
            const Real area_f1 = 0.5*aform_bar_f1_sqrtdet;
#endif
            const Real area_prefac = 0.5*(area_f0 + area_f1);
            
            energy_tmp += area_prefac * eng;
            
            if(not withGradient) continue;
            
            
            // e^T (a11 a12 ; a12 a22) e = a11 e1^2 + 2 a12 e1 e2 + a22 e2^2
            // d/da11 = e1^2
            // d/da12 = 2 e1 e2
            // d/da22 = e2^2
            
            //        const Real grad_le0_a11 = 0.5/le0_sq * evec_0(0) * evec_0(0);
            //        const Real grad_le0_a12 = 0.5/le0_sq * 2.0 * evec_0(0) * evec_0(1);
            //        const Real grad_le0_a22 = 0.5/le0_sq * evec_0(1) * evec_0(1);
            //
            //        const Real grad_le1_a11 = 0.5/le1_sq * evec_1(0) * evec_1(0);
            //        const Real grad_le1_a12 = 0.5/le1_sq * 2.0 * evec_1(0) * evec_1(1);
            //        const Real grad_le1_a22 = 0.5/le1_sq * evec_1(1) * evec_1(1);
            //
            //        const Real grad_prefac = prefactor * 4*(le1 - le0)/std::pow(le0 + le1, 3);
            
            const Real grad_le0_a11 = (1.0/le1 - le1/le0_sq) * evec_0(0) * evec_0(0);
            const Real grad_le0_a12 = (1.0/le1 - le1/le0_sq) * 2.0 * evec_0(0) * evec_0(1);
            const Real grad_le0_a22 = (1.0/le1 - le1/le0_sq) * evec_0(1) * evec_0(1);
            
            const Real grad_le1_a11 = (1.0/le0 - le0/le1_sq) * evec_1(0);
            const Real grad_le1_a12 = (1.0/le0 - le0/le1_sq) * 2.0 * evec_1(0) * evec_1(1);
            const Real grad_le1_a22 = (1.0/le0 - le0/le1_sq) * evec_1(1) * evec_1(1);
            
            const Real grad_prefac = prefactor * nExp * std::pow(engFac, nExp-1);
            
#ifdef INTEGRATE_OVER_MESH_AREA
            gradient_abar(idx_f0, 0) += area_prefac * grad_prefac * grad_le0_a11;
            gradient_abar(idx_f0, 1) += area_prefac * grad_prefac * grad_le0_a12;
            gradient_abar(idx_f0, 2) += area_prefac * grad_prefac * grad_le0_a22;
            
            gradient_abar(idx_f1, 0) += area_prefac * grad_prefac * grad_le1_a11;
            gradient_abar(idx_f1, 1) += area_prefac * grad_prefac * grad_le1_a12;
            gradient_abar(idx_f1, 2) += area_prefac * grad_prefac * grad_le1_a22;
#else
            const Real grad_det_f0_11 = aform_bar_f0(1,1);
            const Real grad_det_f0_12 = - aform_bar_f0(0,1) - aform_bar_f0(1,0);
            const Real grad_det_f0_22 = aform_bar_f0(0,0);
            
            const Real grad_det_f1_11 = aform_bar_f1(1,1);
            const Real grad_det_f1_12 = - aform_bar_f1(0,1) - aform_bar_f1(1,0);
            const Real grad_det_f1_22 = aform_bar_f1(0,0);
            
            // area gradients
            const Real grad_area_f0_11 = 0.25/aform_bar_f0_sqrtdet * grad_det_f0_11;
            const Real grad_area_f0_12 = 0.25/aform_bar_f0_sqrtdet * grad_det_f0_12;
            const Real grad_area_f0_22 = 0.25/aform_bar_f0_sqrtdet * grad_det_f0_22;
            
            const Real grad_area_f1_11 = 0.25/aform_bar_f1_sqrtdet * grad_det_f1_11;
            const Real grad_area_f1_12 = 0.25/aform_bar_f1_sqrtdet * grad_det_f1_12;
            const Real grad_area_f1_22 = 0.25/aform_bar_f1_sqrtdet * grad_det_f1_22;
            
            const Real grad_area_prefac_11 = 0.5*(grad_area_f0_11 + grad_area_f1_11);
            const Real grad_area_prefac_12 = 0.5*(grad_area_f0_12 + grad_area_f1_12);
            const Real grad_area_prefac_22 = 0.5*(grad_area_f0_22 + grad_area_f1_22);
            
            gradient_abar(idx_f0, 0) +=  area_prefac * grad_prefac * grad_le0_a11 + eng * grad_area_prefac_11;
            gradient_abar(idx_f0, 1) +=  area_prefac * grad_prefac * grad_le0_a12 + eng * grad_area_prefac_12;
            gradient_abar(idx_f0, 2) +=  area_prefac * grad_prefac * grad_le0_a22 + eng * grad_area_prefac_22;
            
            gradient_abar(idx_f1, 0) +=  area_prefac * grad_prefac * grad_le1_a11 + eng * grad_area_prefac_11;
            gradient_abar(idx_f1, 1) +=  area_prefac * grad_prefac * grad_le1_a12 + eng * grad_area_prefac_12;
            gradient_abar(idx_f1, 2) +=  area_prefac * grad_prefac * grad_le1_a22 + eng * grad_area_prefac_22;
#endif
        }
        
        energy += energy_tmp;
    }
};


template<typename tMesh, MeshLayer layer>
Real DirichletEnergyOperator_Inverse<tMesh, layer>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient, const bool computeGradient) const
{
    if(prefactor < std::numeric_limits<Real>::epsilon()) return 0;
    
    const int nEdges = mesh.getNumberOfEdges();
    
    const auto & topo = mesh.getTopology();
    const auto & restState = mesh.getRestConfiguration();
    
    // map the gradient into abar
    const int nFaces = mesh.getNumberOfFaces();
    Eigen::Map<Eigen::MatrixXd> gradient_abar(gradient.data(), nFaces, 3);

    Real energy;
    
    this->profiler.push_start("dirichlet energy");
    if(not computeGradient)
    {
        ComputeDirichletEnergy_Inverse<tMesh, layer, false> compute_tbb(prefactor, topo, restState, gradient_abar);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nEdges), compute_tbb, tbb::auto_partitioner());
        energy = compute_tbb.energy;
    }
    else
    {
        ComputeDirichletEnergy_Inverse<tMesh, layer, true> compute_tbb(prefactor, topo, restState, gradient_abar);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nEdges), compute_tbb, tbb::auto_partitioner());
        energy = compute_tbb.energy;
    }
    this->profiler.pop_stop();
    
    return energy;
}

#include "Mesh.hpp"
template class DirichletEnergyOperator_Inverse<Mesh, single>;
template class ComputeDirichletEnergy_Inverse<Mesh, single, true>;
template class ComputeDirichletEnergy_Inverse<Mesh, single, false>;

template class DirichletEnergyOperator_Inverse<BilayerMesh, bottom>;
template class ComputeDirichletEnergy_Inverse<BilayerMesh, bottom, true>;
template class ComputeDirichletEnergy_Inverse<BilayerMesh, bottom, false>;

template class DirichletEnergyOperator_Inverse<BilayerMesh, top>;
template class ComputeDirichletEnergy_Inverse<BilayerMesh, top, true>;
template class ComputeDirichletEnergy_Inverse<BilayerMesh, top, false>;
