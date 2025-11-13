//
//  ComputeCombined_Parametric_InverseGrowth.hpp
//  Elasticity
//
//  Created by Wim M. van Rees on 8/15/18.
//  Copyright © 2018 Wim van Rees. All rights reserved.
//

#ifndef ComputeCombined_Parametric_InverseGrowth_hpp
#define ComputeCombined_Parametric_InverseGrowth_hpp

#include "EnergyHelper_Parametric_Inverse.hpp"
#include "MergePerFaceQuantities.hpp"
#include "TriangleInfo.hpp"

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"

template<typename tMesh, typename tMaterialType, MeshLayer layer, bool withGradient>
struct ComputeCombined_Parametric_InverseGrowth
{
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;
    typedef typename tMesh::tCurrentConfigData tCurrentConfigData;

    const MaterialProperties<tMaterialType>  & material_properties;
    const tCurrentConfigData & currentState;
    const tReferenceConfigData & restState;
    Eigen::Ref<Eigen::MatrixXd> per_face_gradients; // for the cost function computation (gradient of the elastic energy)
    Eigen::Ref<Eigen::MatrixXd> per_face_hessians; // for the gradient of the cost function (mixed hessian of the elastic energy)
    const int nDims;
    const bool computeHessianYoung;

    ComputeCombined_Parametric_InverseGrowth(const MaterialProperties<tMaterialType> & mat_props_in, const tCurrentConfigData & currentState_in, const tReferenceConfigData & restState_in, Eigen::Ref<Eigen::MatrixXd> per_face_gradients_in, Eigen::Ref<Eigen::MatrixXd> per_face_hessians_in):
    material_properties(mat_props_in),
    currentState(currentState_in),
    restState(restState_in),
    per_face_gradients(per_face_gradients_in),
    per_face_hessians(per_face_hessians_in),
    nDims(per_face_gradients.cols()),
    computeHessianYoung((per_face_hessians.cols() > 3*nDims))
    {}

    // split constructor (dont need copy constructor)
    ComputeCombined_Parametric_InverseGrowth(const ComputeCombined_Parametric_InverseGrowth & c, tbb::split):
    material_properties(c.material_properties),
    currentState(c.currentState),
    restState(c.restState),
    per_face_gradients(c.per_face_gradients),
    per_face_hessians(c.per_face_hessians),
    nDims(c.nDims),
    computeHessianYoung(c.computeHessianYoung)
    {}

    void join(const ComputeCombined_Parametric_InverseGrowth & )
    {
        // no need for join at this point : everything is completely serial (arrays need to be merged later)
    }

    void addGradients(const ExtendedTriangleInfo & info, const SaintVenantEnergy_Inverse<withGradient, tMaterialType, layer> & SV_energy_inverse)
    {
        const int i = info.face_idx;

        for(int j=0;j<3;++j)
        {
            per_face_gradients(i, 3*0+j) += SV_energy_inverse.energy_v0(j);
            per_face_gradients(i, 3*1+j) += SV_energy_inverse.energy_v1(j);
            per_face_gradients(i, 3*2+j) += SV_energy_inverse.energy_v2(j);
        }

        // opposite vertex to each edge (if it exists)
        if(info.other_faces[0] != nullptr)
            for(int j=0;j<3;++j) per_face_gradients(i, 3*3+j) += SV_energy_inverse.energy_v_other_e0(j);

        if(info.other_faces[1] != nullptr)
            for(int j=0;j<3;++j) per_face_gradients(i, 3*4+j) += SV_energy_inverse.energy_v_other_e1(j);

        if(info.other_faces[2] != nullptr)
            for(int j=0;j<3;++j) per_face_gradients(i, 3*5+j) += SV_energy_inverse.energy_v_other_e2(j);

        // theta
        {
            per_face_gradients(i, 3*6+0) += SV_energy_inverse.energy_e0;
            per_face_gradients(i, 3*6+1) += SV_energy_inverse.energy_e1;
            per_face_gradients(i, 3*6+2) += SV_energy_inverse.energy_e2;
        }
    }

    template<bool U = withGradient> typename std::enable_if<U, void>::type
    addHessians(const ExtendedTriangleInfo & info, SaintVenantEnergy_Inverse<withGradient, tMaterialType, layer> & SV_energy_inverse)
    {
        SV_energy_inverse.compute_gradients();

        const int i = info.face_idx;
        for(int d=0;d<3;++d)
        {
            // vertex v0
            per_face_hessians(i, nDims*0 + 3*0 + d) += SV_energy_inverse.hessv0(d,0);
            per_face_hessians(i, nDims*1 + 3*0 + d) += SV_energy_inverse.hessv0(d,1);
            per_face_hessians(i, nDims*2 + 3*0 + d) += SV_energy_inverse.hessv0(d,2);

            // vertex v1
            per_face_hessians(i, nDims*0 + 3*1 + d) += SV_energy_inverse.hessv1(d,0);
            per_face_hessians(i, nDims*1 + 3*1 + d) += SV_energy_inverse.hessv1(d,1);
            per_face_hessians(i, nDims*2 + 3*1 + d) += SV_energy_inverse.hessv1(d,2);

            // vertex v2
            per_face_hessians(i, nDims*0 + 3*2 + d) += SV_energy_inverse.hessv2(d,0);
            per_face_hessians(i, nDims*1 + 3*2 + d) += SV_energy_inverse.hessv2(d,1);
            per_face_hessians(i, nDims*2 + 3*2 + d) += SV_energy_inverse.hessv2(d,2);

            // vertex v_other_e0
            per_face_hessians(i, nDims*0 + 3*3 + d) += SV_energy_inverse.hessv_other_e0(d,0);
            per_face_hessians(i, nDims*1 + 3*3 + d) += SV_energy_inverse.hessv_other_e0(d,1);
            per_face_hessians(i, nDims*2 + 3*3 + d) += SV_energy_inverse.hessv_other_e0(d,2);

            // vertex v_other_e1
            per_face_hessians(i, nDims*0 + 3*4 + d) += SV_energy_inverse.hessv_other_e1(d,0);
            per_face_hessians(i, nDims*1 + 3*4 + d) += SV_energy_inverse.hessv_other_e1(d,1);
            per_face_hessians(i, nDims*2 + 3*4 + d) += SV_energy_inverse.hessv_other_e1(d,2);

            // vertex v_other_e2
            per_face_hessians(i, nDims*0 + 3*5 + d) += SV_energy_inverse.hessv_other_e2(d,0);
            per_face_hessians(i, nDims*1 + 3*5 + d) += SV_energy_inverse.hessv_other_e2(d,1);
            per_face_hessians(i, nDims*2 + 3*5 + d) += SV_energy_inverse.hessv_other_e2(d,2);
        }

        per_face_hessians(i, nDims*0 + 3*6 + 0) += SV_energy_inverse.hessphi_e0(0);
        per_face_hessians(i, nDims*1 + 3*6 + 0) += SV_energy_inverse.hessphi_e0(1);
        per_face_hessians(i, nDims*2 + 3*6 + 0) += SV_energy_inverse.hessphi_e0(2);

        per_face_hessians(i, nDims*0 + 3*6 + 1) += SV_energy_inverse.hessphi_e1(0);
        per_face_hessians(i, nDims*1 + 3*6 + 1) += SV_energy_inverse.hessphi_e1(1);
        per_face_hessians(i, nDims*2 + 3*6 + 1) += SV_energy_inverse.hessphi_e1(2);

        per_face_hessians(i, nDims*0 + 3*6 + 2) += SV_energy_inverse.hessphi_e2(0);
        per_face_hessians(i, nDims*1 + 3*6 + 2) += SV_energy_inverse.hessphi_e2(1);
        per_face_hessians(i, nDims*2 + 3*6 + 2) += SV_energy_inverse.hessphi_e2(2);

        if(computeHessianYoung) addHessians_Young(info);
    }

    template<bool U = withGradient> typename std::enable_if<U, void>::type
    addHessians_Young(const ExtendedTriangleInfo & info)
    {
        const int i = info.face_idx;
        const Real invYoung = 1.0 / this->material_properties.getFaceMaterial(i).getYoung();

        for(int d=0;d<3;++d)
        {
            // vertex v0
            per_face_hessians(i, nDims*3 + 3*0 + d) += per_face_gradients(i, 3*0 + d) * invYoung;

            // vertex v1
            per_face_hessians(i, nDims*3 + 3*1 + d) += per_face_gradients(i, 3*1 + d) * invYoung;

            // vertex v2
            per_face_hessians(i, nDims*3 + 3*2 + d) += per_face_gradients(i, 3*2 + d) * invYoung;

            // vertex v_other_e0
            per_face_hessians(i, nDims*3 + 3*3 + d) += per_face_gradients(i, 3*3 + d) * invYoung;

            // vertex v_other_e1
            per_face_hessians(i, nDims*3 + 3*4 + d) += per_face_gradients(i, 3*4 + d) * invYoung;

            // vertex v_other_e2
            per_face_hessians(i, nDims*3 + 3*5 + d) += per_face_gradients(i, 3*5 + d) * invYoung;
        }

        per_face_hessians(i, nDims*3 + 3*6 + 0) += per_face_gradients(i, 3*6 + 0) * invYoung;

        per_face_hessians(i, nDims*3 + 3*6 + 1) += per_face_gradients(i, 3*6 + 1) * invYoung;

        per_face_hessians(i, nDims*3 + 3*6 + 2) += per_face_gradients(i, 3*6 + 2) * invYoung;
    }

    template<bool U = withGradient> typename std::enable_if<!U, void>::type
    addHessians(const ExtendedTriangleInfo & , SaintVenantEnergy_Inverse<withGradient, tMaterialType, layer> & )
    {
        // do nothing
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


    void operator()(const tbb::blocked_range<int> & face_range)
    {
        for (int i=face_range.begin(); i != face_range.end(); ++i)
        {
            const ExtendedTriangleInfo & info = this->currentState.getTriangleInfo(i);

            const tMaterialType matprop = this->material_properties.getFaceMaterial(i);

            const Eigen::Matrix2d aform_bar = getFirstFundamentalForm(i);
            const Eigen::Matrix2d bform_bar = this->restState.getSecondFundamentalForm(i);

            // create an inverse energy operator, compute energy gradient, and store
            SaintVenantEnergy_Inverse<withGradient, tMaterialType, layer> SV_energy_inverse(matprop, aform_bar, bform_bar, info);
            SV_energy_inverse.compute();
            addGradients(info, SV_energy_inverse);

            // deal with the hessian
            addHessians(info, SV_energy_inverse);
        }
    }

};

struct MultiplyAndComputeAreas
{
    const TopologyData & topo;
    const tVecMat2d & aforms;
    const Eigen::Ref<const Eigen::VectorXd> gradients;
    Eigen::Ref<Eigen::VectorXd> areas;
    enum class RunCase { vertices, edges };
    const RunCase runCase;

    Real energySum;

    MultiplyAndComputeAreas(const TopologyData & topo, const tVecMat2d & aforms, const Eigen::Ref<const Eigen::VectorXd>  gradients, Eigen::Ref<Eigen::VectorXd> areas, const RunCase runCase):
    topo(topo),
    aforms(aforms),
    gradients(gradients),
    areas(areas),
    runCase(runCase),
    energySum(0.0)
    {}

    MultiplyAndComputeAreas(const MultiplyAndComputeAreas & c, tbb::split):
    topo(c.topo),
    aforms(c.aforms),
    gradients(c.gradients),
    areas(c.areas),
    runCase(c.runCase),
    energySum(0.0)
    {}

    void join(const MultiplyAndComputeAreas & j)
    {
        energySum += j.energySum;
    }

    void doVertices(const tbb::blocked_range<int> & vertex_range)
    {
        const auto & vertex2faces = topo.getVertex2Faces();
        assert(gradients.rows()%3==0);
        const int nVertices = gradients.rows() / 3;
        Eigen::Map<const Eigen::MatrixXd> per_vertex_gradient(gradients.data(), nVertices, 3);

        for (int i=vertex_range.begin(); i != vertex_range.end(); ++i)
        {
            // compute the area of this vertex
            Real vertex_area = 0.0;
            const size_t nFacesV = vertex2faces[i].size();
            for(size_t j=0;j<nFacesV;++j)
            {
                const int face_idx = vertex2faces[i][j];
                const Real face_area = 0.5*std::sqrt(aforms[face_idx].determinant());
                vertex_area += face_area / 3.0;
            }
            // store area
            areas(i) = vertex_area;

            // compute energy
            for(int d=0;d<3;++d)
                energySum += std::pow(per_vertex_gradient(i,d),2) / vertex_area;
                //printf("OLOLOAREA_Vert = %10.10e \n", vertex_area);}

            //printf("OLOLENERGY_Vert = %10.10e \n", energySum);
        }
    }

    void doEdges(const tbb::blocked_range<int> & edge_range)
    {
        const auto edge2faces = topo.getEdge2Faces();
        for (int i=edge_range.begin(); i != edge_range.end(); ++i)
        {
            const int face_idx_0 = edge2faces(i,0);
            const int face_idx_1 = edge2faces(i,1);
            const Real face_area_0 = (face_idx_0 < 0 ? 0.0 : 0.5*std::sqrt(aforms[face_idx_0].determinant()));
            const Real face_area_1 = (face_idx_1 < 0 ? 0.0 : 0.5*std::sqrt(aforms[face_idx_1].determinant()));
            const Real edge_area = 0.5*(face_area_0 + face_area_1);

            // store area
            areas(i) = edge_area;

            // compute energy
            energySum += std::pow(gradients(i),2) / edge_area;
        }
        //printf("OLOLENERGY_Edge = %10.10e \n", energySum);
    }

    void operator () (const tbb::blocked_range<int>& var_range)
    {
        switch(runCase)
        {
            case RunCase::vertices:
                doVertices(var_range);
                break;
            case RunCase::edges:
                doEdges(var_range);
                break;
            default:
                break;
        }
    }
};

#endif /* ComputeCombined_Parametric_InverseGrowth_hpp */
