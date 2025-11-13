//
//  ConformalMappingBoundary.hpp
//  Elasticity
//
//  Created by Wim van Rees on 11/30/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef ConformalMappingBoundary_hpp
#define ConformalMappingBoundary_hpp

#include "common.hpp"
#include "Profiler.hpp"

template<typename tMesh, MeshLayer layer = single>
class ConformalMappingBoundary
{
protected:
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;

    mutable Profiler profiler;

    void applyFreeConformalMap(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> confmap_result) const;
    void prepareBoundaryVertices_disk(const tMesh & mesh, const Eigen::Ref<const Eigen::MatrixXd> vertices, Eigen::VectorXi & boundary_verts, Eigen::MatrixXd & boundary_verts_vals) const;
    void prepareBoundaryVertices_rect(const tMesh & mesh, const Eigen::Ref<const Eigen::MatrixXd> vertices, Eigen::VectorXi & boundary_verts, Eigen::MatrixXd & boundary_verts_vals, const Real halfEdgeX, const Real halfEdgeY) const;
    bool doMapping(const tMesh & mesh, const Eigen::Ref<const Eigen::VectorXi> boundary_verts, const Eigen::Ref<const Eigen::MatrixXd> boundary_verts_vals, Eigen::Ref<Eigen::MatrixXd> vertices) const;

public:

    ConformalMappingBoundary()
    {}

    bool compute(const tMesh & mesh, const Eigen::Ref<const Eigen::VectorXi> boundary_verts, const Eigen::Ref<const Eigen::MatrixXd> boundary_verts_vals, Eigen::MatrixXd & mapped_vertices, const bool constrainOnDisk = false) const;
    bool compute_disk(const tMesh & mesh, Eigen::MatrixXd & mapped_vertices) const;
    bool compute_rect(const tMesh & mesh, Eigen::MatrixXd & mapped_vertices, const Real lx, const Real ly) const;
    void computeFree(const tMesh & mesh, Eigen::MatrixXd & mapped_vertices) const;
};


#endif /* ConformalMappingBoundary_hpp */
