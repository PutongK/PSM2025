//
//  EdgeDistortionOperator.cpp
//  Elasticity
//
//  Created by Wim van Rees on 6/9/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#include "EdgeDistortionOperator.hpp"

template<typename tMesh>
Real EdgeDistortionOperator<tMesh>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> , const bool computeGradient) const
{
    // get the mesh features
    const auto & topo = mesh.getTopology();
    const auto & currentState = mesh.getCurrentConfiguration();
    const auto & boundaryConditions = mesh.getBoundaryConditions();

    // get mesh quantities
    const auto vertices = currentState.getVertices();

    const auto vertices_bc = boundaryConditions.getVertexBoundaryConditions();
    const auto edge2vertices = topo.getEdge2Vertices();

    Real maxval_num = 0, maxval_denum = 0;
    Real minval_num = 0, minval_denum = 0;

    const int nEdges = mesh.getNumberOfEdges();

    for(int i=0;i<nEdges;++i)
    {
        const int idx_v0 = edge2vertices(i,0);
        const int idx_v1 = edge2vertices(i,1);

        const Eigen::Vector3d v0 = vertices.row(idx_v0);
        const Eigen::Vector3d v1 = vertices.row(idx_v1);
        const Eigen::Vector3d diff_v =  v1 - v0;
        const Real target_length = targetEdgeLengths(i);
        const Real growth = diff_v.norm()/target_length; // between 0 and infinity

        const Real expval_max = std::exp( alpha * growth);
        const Real expval_min = std::exp(-alpha * growth);

        maxval_num += growth * expval_max;
        minval_num += growth * expval_min;

        maxval_denum += expval_max;
        minval_denum += expval_min;
    }
    const Real energy = maxval_num / minval_num * (minval_denum / maxval_denum);

    if(computeGradient)
    {
        for(int i=0;i<nEdges;++i)
        {
            const int idx_v0 = edge2vertices(i,0);
            const int idx_v1 = edge2vertices(i,1);

            const Eigen::Vector3d v0 = vertices.row(idx_v0);
            const Eigen::Vector3d v1 = vertices.row(idx_v1);
            const Eigen::Vector3d diff_v =  v1 - v0;
            const Real target_length = targetEdgeLengths(i);
            const Real growth = diff_v.norm()/target_length; // between 0 and infinity

            const Real expval_max = std::exp( alpha * growth);
            const Real expval_min = std::exp(-alpha * growth);

            const Eigen::Vector3d gradv0_growth = -1.0/target_length * diff_v;
            const Eigen::Vector3d gradv1_growth =  1.0/target_length * diff_v;

            const Eigen::Vector3d gradv0_expval_max =  alpha * expval_max * gradv0_growth;
            const Eigen::Vector3d gradv0_expval_min = -alpha * expval_min * gradv0_growth;

            const Eigen::Vector3d gradv1_expval_max =  alpha * expval_max * gradv1_growth;
            const Eigen::Vector3d gradv1_expval_min = -alpha * expval_min * gradv1_growth;

            const Eigen::Vector3d gradv0_maxval_num = gradv0_growth * expval_max + growth * gradv0_expval_max;
            const Eigen::Vector3d gradv0_minval_num = gradv0_growth * expval_min + growth * gradv0_expval_min;

            const Eigen::Vector3d gradv1_maxval_num = gradv1_growth * expval_max + growth * gradv1_expval_max;
            const Eigen::Vector3d gradv1_minval_num = gradv1_growth * expval_min + growth * gradv1_expval_min;

            const Eigen::Vector3d gradv0_maxval_denum = gradv0_expval_max;
            const Eigen::Vector3d gradv0_minval_denum = gradv0_expval_min;

            const Eigen::Vector3d gradv1_maxval_denum = gradv1_expval_max;
            const Eigen::Vector3d gradv1_minval_denum = gradv1_expval_min;

            //(maxval_num / minval_num) * (minval_denum / maxval_denum);
            const Eigen::Vector3d gradv0_eng =
            (gradv0_maxval_num / minval_num) * (minval_denum / maxval_denum) +
            (maxval_num / minval_num) * (gradv0_minval_denum / maxval_denum) -
            (maxval_num / minval_num) * (gradv0_minval_num / minval_num) * (minval_denum / maxval_denum) -
            (maxval_num / minval_num) * (minval_denum / maxval_denum) * (gradv0_maxval_denum / maxval_denum);

            const Eigen::Vector3d gradv1_eng =
            (gradv1_maxval_num / minval_num) * (minval_denum / maxval_denum) +
            (maxval_num / minval_num) * (gradv1_minval_denum / maxval_denum) -
            (maxval_num / minval_num) * (gradv1_minval_num / minval_num) * (minval_denum / maxval_denum) -
            (maxval_num / minval_num) * (minval_denum / maxval_denum) * (gradv1_maxval_denum / maxval_denum);

            for(int j=0;j<3;++j)
            {
                if(not vertices_bc(idx_v0,j)) gradient_vertices(idx_v0,j) += gradv0_eng(j);
                if(not vertices_bc(idx_v1,j)) gradient_vertices(idx_v1,j) += gradv1_eng(j);
            }
        }

    }


    lastEnergy = energy;
    return energy;
}


// explicit instantiations
#include "Mesh.hpp"
template class EdgeDistortionOperator<Mesh>;
template class EdgeDistortionOperator<BilayerMesh>;
