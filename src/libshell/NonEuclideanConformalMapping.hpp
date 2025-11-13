//
//  NonEuclideanConformalMapping.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/15/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef NonEuclideanConformalMapping_hpp
#define NonEuclideanConformalMapping_hpp


#include "common.hpp"
#include "Profiler.hpp"

template<typename tMesh, MeshLayer layer = single>
class NonEuclideanConformalMapping
{
protected:
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;

    const Real eps_posdef_correction;
    mutable Profiler profiler;
    
    void constructLaplacianTriplets(const tMesh & mesh, std::vector<Eigen::Triplet<Real>> & laplacian_triplets) const;
    void addEpsilonToLaplacianTripletsDiagonal(const tMesh & mesh, const Real eps, std::vector<Eigen::Triplet<Real>> & laplacian_triplets) const;
    
    void constructAreaTriplets(const tMesh & mesh, const std::vector<int> & boundary_loop, std::vector<Eigen::Triplet<Real>> & area_triplets) const;
    void checkAreaTriplets(const tMesh & mesh, const std::vector<Eigen::Triplet<Real>> & area_triplets) const;
    
    void constructLHSTriplets(const tMesh & mesh, const std::vector<int> & boundary_loop, std::vector<Eigen::Triplet<Real>> & LHS_triplets) const;
    
    void getBoundaryLoop(const tMesh & mesh, std::vector<int> & boundary_loop) const;
    void checkBoundaryLoop(const tMesh & mesh, const std::vector<int> & boundary_loop) const;
    
    void rescaleEigenvectorsToSurfaceArea(const tMesh & mesh, Eigen::MatrixXd & eigenvecs) const;
    
    template<MeshLayer L = layer> typename std::enable_if<L==single, const tVecMat2d & >::type
    getFirstFundamentalForms(const tMesh & mesh) const
    {
        return mesh.getRestConfiguration().getFirstFundamentalForms();
    }
    
    template<MeshLayer L = layer> typename std::enable_if<L!=single, const tVecMat2d & >::type
    getFirstFundamentalForms(const tMesh & mesh) const
    {
        return mesh.getRestConfiguration().template getFirstFundamentalForms<L>();
    }
public:
    
    NonEuclideanConformalMapping(const Real eps_posdef_correction_in = 1e-8):
    eps_posdef_correction(eps_posdef_correction_in)
    {}
    
    void compute(const tMesh & mesh, Eigen::VectorXd & eigenvals, Eigen::MatrixXd & eigenvecs, const bool rescaleEigenvectors = false) const;
};

#endif /* NonEuclideanConformalMapping_hpp */
