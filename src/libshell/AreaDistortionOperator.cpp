//
//  AreaDistortionOperator.cpp
//  Elasticity
//
//  Created by Wim van Rees on 6/9/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#include "AreaDistortionOperator.hpp"
#include "ExtendedTriangleInfo.hpp"

template<typename tMesh>
Real AreaDistortionOperator<tMesh>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> , const bool computeGradient) const
{
    // get the mesh features
    const auto & topo = mesh.getTopology();
    const auto & currentState = mesh.getCurrentConfiguration();
    const auto & boundaryConditions = mesh.getBoundaryConditions();

    // get mesh quantities
    const auto vertices = currentState.getVertices();

    const auto vertices_bc = boundaryConditions.getVertexBoundaryConditions();
    const auto face2vertices = topo.getFace2Vertices();

    Real energy = 0.0;

    const int nFaces = mesh.getNumberOfFaces();

    for(int i=0;i<nFaces;++i)
    {
        // vertex indices
        const int idx_v0 = face2vertices(i,0);
        const int idx_v1 = face2vertices(i,1);
        const int idx_v2 = face2vertices(i,2);

        // vertex locations
        const Eigen::Vector3d v0 = vertices.row(idx_v0);
        const Eigen::Vector3d v1 = vertices.row(idx_v1);
        const Eigen::Vector3d v2 = vertices.row(idx_v2);

        // computed signed area
        const Real area = 0.5*(v0(0)*v1(1) - v1(0)*v0(1) + v1(0)*v2(1) - v2(0)*v1(1) + v2(0)*v0(1) - v0(0)*v2(1));

        const Real target_area = targetFaceAreas(i);
        const Real diff = (area - target_area) / target_area;
        energy += 0.5 * diff * diff;

        if(not computeGradient) continue;

        const Eigen::Vector3d gradv0_area = (Eigen::Vector3d() <<
                                             v1(1) - v2(1),
                                            -v1(0) + v2(0),
                                             0
                                             ).finished();

        const Eigen::Vector3d gradv1_area = (Eigen::Vector3d() <<
                                            -v0(1) + v2(1),
                                             v0(0) - v2(0),
                                             0
                                             ).finished();

        const Eigen::Vector3d gradv2_area = (Eigen::Vector3d() <<
                                             -v1(1) + v0(1),
                                              v1(0) - v0(0),
                                             0
                                             ).finished();

        for(int j=0;j<3;++j)
        {
            if(not vertices_bc(idx_v0,j)) gradient_vertices(idx_v0,j) += eng_fac * diff * gradv0_area(j);
            if(not vertices_bc(idx_v1,j)) gradient_vertices(idx_v1,j) += eng_fac * diff * gradv1_area(j);
            if(not vertices_bc(idx_v2,j)) gradient_vertices(idx_v2,j) += eng_fac * diff * gradv2_area(j);
        }
    }

    energy *= eng_fac;

    lastEnergy = energy;
    return energy;
}


// explicit instantiations
#include "Mesh.hpp"
template class AreaDistortionOperator<Mesh>;
template class AreaDistortionOperator<BilayerMesh>;
