//
//  ConformalMappingBoundary.cpp
//  Elasticity
//
//  Created by Wim van Rees on 11/30/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "ConformalMappingBoundary.hpp"
#include "NonEuclideanConformalMapping.hpp"

#include <igl/boundary_loop.h>
#include <igl/lscm.h>


template<typename tMesh, MeshLayer layer>
void ConformalMappingBoundary<tMesh, layer>::applyFreeConformalMap(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> confmap_result) const
{
    NonEuclideanConformalMapping<tMesh, layer> conformalMappingOp(1e-8);
    Eigen::VectorXd eigenvals;
    Eigen::MatrixXd eigenvecs;

    conformalMappingOp.compute(mesh, eigenvals, eigenvecs, true);

    const int lastIdx = eigenvals.size()-1;

    const int nVertices = mesh.getNumberOfVertices();

    for(int i=0;i<nVertices;++i)
    {
        confmap_result(i,0) = eigenvecs(i, lastIdx);
        confmap_result(i,1) = eigenvecs(i + nVertices, lastIdx);
        confmap_result(i,2) = 0.0;
    }
}

template<typename tMesh, MeshLayer layer>
void ConformalMappingBoundary<tMesh, layer>::prepareBoundaryVertices_disk(const tMesh & mesh, const Eigen::Ref<const Eigen::MatrixXd> vertices, Eigen::VectorXi & boundary_verts, Eigen::MatrixXd & boundary_verts_vals) const
{
    const auto face2vertices = mesh.getTopology().getFace2Vertices();

    // get longest boundary loop
    std::vector<int> boundary_loop;
    const Eigen::MatrixXi & ref_face2vertices = face2vertices;
    igl::boundary_loop(ref_face2vertices, boundary_loop); // pick the longest loop
    const size_t nBoundaryVertices = boundary_loop.size();

    // we will constraint the outer perimeter of the boundary in a circle
    // get largest distance from center to compute the circle radius (origin will be at the center)

    Real maxRadiusSq = 0;
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vidx = boundary_loop[i];
        maxRadiusSq = std::max(maxRadiusSq, std::pow(vertices(vidx,0),2) + std::pow(vertices(vidx,1),2));
    }
    const Real circleRadius = std::sqrt(maxRadiusSq);

    // prepare data structures for the boundary conditions
    boundary_verts.resize(nBoundaryVertices);
    boundary_verts_vals.resize(nBoundaryVertices,2); // note: the IGL code documentation says this should have 3 columns, but it segfaults with 3 - instead the tutorial uses 2 columns which works

    // fill them up
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vidx = boundary_loop[i];
        boundary_verts(i) = vidx;

        const Real theta = std::atan2(vertices(vidx,1),vertices(vidx,0));
        boundary_verts_vals(i,0) = circleRadius * std::cos(theta);
        boundary_verts_vals(i,1) = circleRadius * std::sin(theta);
    }
}


template<typename tMesh, MeshLayer layer>
bool ConformalMappingBoundary<tMesh, layer>::doMapping(const tMesh & mesh, const Eigen::Ref<const Eigen::VectorXi> boundary_verts, const Eigen::Ref<const Eigen::MatrixXd> boundary_verts_vals, Eigen::Ref<Eigen::MatrixXd> vertices) const
{
    const auto face2vertices = mesh.getTopology().getFace2Vertices();

    // do the LSCM routine from IGL
    Eigen::MatrixXd lscm_result;
    const bool lscm_status = igl::lscm(vertices, face2vertices, boundary_verts, boundary_verts_vals, lscm_result);

    const int nVertices = mesh.getNumberOfVertices();
    for(int i=0;i<nVertices;++i)
    {
        vertices(i,0) = lscm_result(i,0);
        vertices(i,1) = lscm_result(i,1);
    }
    return lscm_status;
}

template<typename tMesh, MeshLayer layer>
bool ConformalMappingBoundary<tMesh, layer>::compute_disk(const tMesh & mesh, Eigen::MatrixXd & mapped_vertices) const
{
    const int nVertices = mesh.getNumberOfVertices();
    mapped_vertices.resize(nVertices, 3);
    mapped_vertices.setZero();

    // prepare the boundary conditions : first we map everything to a plane, then constrain the perimeter to a disk
    Eigen::VectorXi boundary_verts;
    Eigen::MatrixXd boundary_verts_vals;
    applyFreeConformalMap(mesh, mapped_vertices);
    prepareBoundaryVertices_disk(mesh, mapped_vertices, boundary_verts, boundary_verts_vals);

    // now we use the (planar) IC with distorted boundary as an input to the built-in lscm routine
    const bool lscm_status = doMapping(mesh, boundary_verts, boundary_verts_vals, mapped_vertices);

    return lscm_status;
}


template<typename tMesh, MeshLayer layer>
void ConformalMappingBoundary<tMesh, layer>::computeFree(const tMesh & mesh, Eigen::MatrixXd & mapped_vertices) const
{
    const int nVertices = mesh.getNumberOfVertices();
    mapped_vertices.resize(nVertices, 3);
    mapped_vertices.setZero();

    applyFreeConformalMap(mesh, mapped_vertices);
}


template<typename tMesh, MeshLayer layer>
bool ConformalMappingBoundary<tMesh, layer>::compute(const tMesh & mesh, const Eigen::Ref<const Eigen::VectorXi> boundary_verts, const Eigen::Ref<const Eigen::MatrixXd> boundary_verts_vals, Eigen::MatrixXd & mapped_vertices, const bool constrainOnDisk) const
{
    // here we already know what vertices to constrain : do the free map and then call constrained map directly
    const int nVertices = mesh.getNumberOfVertices();
    mapped_vertices.resize(nVertices, 3);
    applyFreeConformalMap(mesh, mapped_vertices);

    bool lscm_status;

    if(constrainOnDisk)
    {
        // in addition to the input vertices, also make sure the perimeter lies on a circle

        // prepare fixed vertices for perimeter
        Eigen::VectorXi boundary_verts_perimeter;
        Eigen::MatrixXd boundary_verts_vals_perimeter;
        prepareBoundaryVertices_disk(mesh, mapped_vertices, boundary_verts_perimeter, boundary_verts_vals_perimeter);

        // append the given boundary vertices with the perimeter ones
        const int nBoundaryVerts_input = boundary_verts.rows();
        const int nBoundaryVerts_perimeter = boundary_verts_perimeter.rows();
        Eigen::VectorXi boundary_verts_combined(nBoundaryVerts_input + nBoundaryVerts_perimeter);
        Eigen::MatrixXd boundary_verts_vals_combined(nBoundaryVerts_input + nBoundaryVerts_perimeter, 2);
        boundary_verts_combined << boundary_verts, boundary_verts_perimeter;
        boundary_verts_vals_combined << boundary_verts_vals, boundary_verts_vals_perimeter;

        // do the mapping with the combined fixed vertices
        lscm_status = doMapping(mesh, boundary_verts_combined, boundary_verts_vals_combined, mapped_vertices);
    }
    else
    {
        lscm_status = doMapping(mesh, boundary_verts, boundary_verts_vals, mapped_vertices);
    }

    return lscm_status;
}

#include "Mesh.hpp"
template class ConformalMappingBoundary<Mesh, single>;
template class ConformalMappingBoundary<BilayerMesh, bottom>;
template class ConformalMappingBoundary<BilayerMesh, top>;
