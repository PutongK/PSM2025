//
//  NonEuclideanConformalMapping.cpp
//  Elasticity
//
//  Created by Wim van Rees on 9/15/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "NonEuclideanConformalMapping.hpp"
#include <Eigen/Eigenvalues>

#include <igl/boundary_loop.h>

#ifdef USEARPACK
#include <unsupported/Eigen/ArpackSupport>
#endif

template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::compute(const tMesh & mesh, Eigen::VectorXd & eigenvals, Eigen::MatrixXd & eigenvecs, const bool rescaleEigenvectors) const
{
    const int nVertices = mesh.getNumberOfVertices();

    // create the laplacian matrix
    profiler.push_start("laplacian triplets");
    std::vector<Eigen::Triplet<Real>> laplacian_triplets;
    constructLaplacianTriplets(mesh, laplacian_triplets);

    // add eps to diagonal to make it positive definite for sure
    if(eps_posdef_correction > std::numeric_limits<Real>::epsilon())
        addEpsilonToLaplacianTripletsDiagonal(mesh, eps_posdef_correction, laplacian_triplets);
    profiler.pop_stop();


    // find all boundary points
    profiler.push_start("boundary loop");
    std::vector<int> boundary_loop;
    getBoundaryLoop(mesh, boundary_loop);
    profiler.pop_stop();
    //    checkBoundaryLoop(mesh, boundary_loop);

    // compute the area triplets
    profiler.push_start("area triplets");
    std::vector<Eigen::Triplet<Real>> area_triplets;
    constructAreaTriplets(mesh, boundary_loop, area_triplets);
    profiler.pop_stop();
    //    checkAreaTriplets(mesh, area_triplets);

    // combine the triplets to create the RHS matrix
    profiler.push_start("prepare RHS matrix");
    laplacian_triplets.insert( laplacian_triplets.end(), area_triplets.begin(), area_triplets.end() );

    // create RHS matrix (laplacian)
    Eigen::SparseMatrix<Real> RHS_sparse(2*nVertices, 2*nVertices);
    RHS_sparse.setFromTriplets(laplacian_triplets.begin(), laplacian_triplets.end());
    profiler.pop_stop();

    // create LHS matrix (boundary stuff)
    profiler.push_start("construct LHS matrix");
    std::vector<Eigen::Triplet<Real>> LHS_triplets;
    constructLHSTriplets(mesh, boundary_loop, LHS_triplets);

    Eigen::SparseMatrix<Real> LHS_sparse(2*nVertices, 2*nVertices);
    LHS_sparse.setFromTriplets(LHS_triplets.begin(), LHS_triplets.end());
    profiler.pop_stop();


    // do the eigenvalue problem
    profiler.push_start("eigensolve");
#ifndef USEARPACK
    // make the matrices dense
    Eigen::MatrixXd LHS = Eigen::MatrixXd(LHS_sparse);
    Eigen::MatrixXd RHS = Eigen::MatrixXd(RHS_sparse);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(LHS, RHS);
#else
    // use the eigen wrapper for arpack
    LHS_sparse.makeCompressed();
    RHS_sparse.makeCompressed();
    Eigen::ArpackGeneralizedSelfAdjointEigenSolver<Eigen::SparseMatrix<Real>> es(LHS_sparse, RHS_sparse, 1, "LA", Eigen::ComputeEigenvectors); // compute 1 (1) largest algebraic (LA) eigenvalue including eigenvectors
    Eigen::ComputationInfo status = es.info();
    if(status != Eigen::Success)
    {
        std::cout << "PROBLEM : ArpackGeneralizedSelfAdjointEigenSolver status is not success \t " << status << std::endl;
        // Success = 0,
        // NumericalIssue = 1,
        // NoConvergence = 2,
        // InvalidInput = 3
    }
#endif
    profiler.pop_stop();

    // set the output
    eigenvals = es.eigenvalues();
    eigenvecs = es.eigenvectors();

    if(rescaleEigenvectors)
    {
        profiler.push_start("rescale eigenvectors");
        rescaleEigenvectorsToSurfaceArea(mesh, eigenvecs);
        profiler.pop_stop();
    }
    profiler.printSummary();
}


template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::constructLaplacianTriplets(const tMesh & mesh, std::vector<Eigen::Triplet<Real>> & laplacian_triplets) const
{
    const int nVertices = mesh.getNumberOfVertices();
    const int nFaces = mesh.getNumberOfFaces();

    const auto & topo = mesh.getTopology();
    const auto face2vertices = topo.getFace2Vertices();

    const tVecMat2d & aforms = getFirstFundamentalForms(mesh);


    laplacian_triplets.reserve(3*nFaces); //very rough estimation of number of connected vertices

    for(int i=0;i<nFaces;++i)
    {
        const Eigen::Matrix2d & aform_bar = aforms[i];

        const Real doublearea = std::sqrt(aform_bar.determinant());
        const Real prefac = 2.0 * 1.0/(4.0*doublearea); // not sure why 2x should be there

        // e0_term corresponds to c-edge
        // e1_term corresponds to a-edge (<a,a>)
        // e2_term corresponds to b-edge (<b,b>)
        const Real e0_term = -prefac * aform_bar(0,1); // (e0 = c since aform = aform(e1,e2)
        const Real e1_term =  prefac * (aform_bar(1,1) + aform_bar(0,1));
        const Real e2_term =  prefac * (aform_bar(0,0) + aform_bar(0,1));

        // get the three vertex indices
        const int idx_v0 = face2vertices(i,0);
        const int idx_v1 = face2vertices(i,1);
        const int idx_v2 = face2vertices(i,2);

        // edge e0 : v1 - v0
        // edge e1 : v2 - v1
        // edge e2 : v0 - v2

        // |e0|^2 = |v1 - v0|^2 = |v1|^2 + |v0|^2 - 2 <v1.v0> = v1_x^2 + v1_y^2 + v0_x^2 + v0_y^2 - 2 (v1_x v0_x + v1_y v0_y)
        // |e1|^2 = |v2 - v1|^2 = |v2|^2 + |v1|^2 - 2 <v2.v1>
        // |e2|^2 = |v0 - v2|^2 = |v0|^2 + |v2|^2 - 2 <v0.v2>

        // |v0|^2 --> e0_term + e2_term
        // |v1|^2 --> e1_term + e0_term
        // |v2|^2 --> e2_term + e1_term

        // two components
        for(int d=0;d<2;++d)
        {
            // diagonals
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v0, d*nVertices + idx_v0, e0_term + e2_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v1, d*nVertices + idx_v1, e1_term + e0_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v2, d*nVertices + idx_v2, e2_term + e1_term));

            // off-diagonals + symmetry
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v1, d*nVertices + idx_v0, -e0_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v2, d*nVertices + idx_v1, -e1_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v0, d*nVertices + idx_v2, -e2_term));

            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v0, d*nVertices + idx_v1, -e0_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v1, d*nVertices + idx_v2, -e1_term));
            laplacian_triplets.push_back(Eigen::Triplet<Real>(d*nVertices + idx_v2, d*nVertices + idx_v0, -e2_term));
        }
    }

}

template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::constructAreaTriplets(const tMesh & mesh, const std::vector<int> & boundary_loop, std::vector<Eigen::Triplet<Real>> & area_triplets) const
{
    const int nVertices = mesh.getNumberOfVertices();

    const size_t nBoundaryVertices = boundary_loop.size();
    area_triplets.reserve(4*nBoundaryVertices);

    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        // boundary edge is made up of these vertices:
        const int vb0_idx = boundary_loop[i];
        const int vb1_idx = boundary_loop[(i+1)%nBoundaryVertices];

        // area contribution is
        // dA = 0.5*( u(vb0_idx) * v(vb1_idx) - u(vb1_idx) * v(vb0_idx) )
        // so assign  +1/sqrt(2) to vb0_idx, vb1_idx + nVertices
        // and assign -1/sqrt(2) to vb1_idx, vb0_idx + nVertices
        const Real contrib = 0.25;// sum will be -1/2 (minus cause this contribution needs to be subtracted - but change sign so area is positive - is the boundary_loop ordering consistent with negative area?)

        area_triplets.push_back(Eigen::Triplet<Real>(vb0_idx, vb1_idx + nVertices, contrib));
        area_triplets.push_back(Eigen::Triplet<Real>(vb1_idx + nVertices, vb0_idx, contrib));

        area_triplets.push_back(Eigen::Triplet<Real>(vb1_idx, vb0_idx + nVertices, -contrib));
        area_triplets.push_back(Eigen::Triplet<Real>(vb0_idx + nVertices, vb1_idx, -contrib));
    }
}




template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::getBoundaryLoop(const tMesh & mesh, std::vector<int> & boundary_loop) const
{
    const auto & topo = mesh.getTopology();
    const auto face2vertices = topo.getFace2Vertices();

    // do all boundary loops
    //    std::vector<std::vector<int>> boundary_loops;
    //    igl::boundary_loop<Eigen::MatrixXi, int>(face2vertices, boundary_loops);
    //    const std::vector<int> & boundary_loop = boundary_loops[0]; // pick the first loop

    // do only longest boundary loop
    const Eigen::MatrixXi & ref_face2vertices = face2vertices;
    igl::boundary_loop(ref_face2vertices, boundary_loop); // pick the longest loop
    assert(boundary_loop.size() > 0);
}


template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::addEpsilonToLaplacianTripletsDiagonal(const tMesh & mesh, const Real eps, std::vector<Eigen::Triplet<Real>> & laplacian_triplets) const
{
    const int nVertices = mesh.getNumberOfVertices();

    for(int i=0;i<2*nVertices;++i)
        laplacian_triplets.push_back(Eigen::Triplet<Real>(i, i, eps));
}

template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::constructLHSTriplets(const tMesh & mesh, const std::vector<int> & boundary_loop, std::vector<Eigen::Triplet<Real>> & LHS_triplets) const
{
    const int nVertices = mesh.getNumberOfVertices();
    const size_t nBoundaryVertices = boundary_loop.size();

    LHS_triplets.reserve(2*nBoundaryVertices + 2*nBoundaryVertices*nBoundaryVertices);

    // diagonal = 1 if vertex is on boundary
    // off-diagonal = eB eB^T --> (eB eB^T)_ij = a_ik a_jk (symmetric)
    // if (i,j) is on the boundary --> one of k=1 or k=2 is non-zero --> entry is 1

    // fill in diagonal
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vb_idx = boundary_loop[i];
        LHS_triplets.push_back(Eigen::Triplet<Real>(vb_idx, vb_idx, 1));
        LHS_triplets.push_back(Eigen::Triplet<Real>(vb_idx + nVertices, vb_idx + nVertices, 1));
    }

    // fill in off-diagonal
    const Real offDiagonalFac = -1.0/nBoundaryVertices;

    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vb_idx_i = boundary_loop[i];
        for(size_t j=0;j<nBoundaryVertices;++j)
        {
            const int vb_idx_j = boundary_loop[j];
            LHS_triplets.push_back(Eigen::Triplet<Real>(vb_idx_i, vb_idx_j, offDiagonalFac));
            LHS_triplets.push_back(Eigen::Triplet<Real>(vb_idx_i + nVertices, vb_idx_j + nVertices, offDiagonalFac));
        }
    }
/*
// old dense formulation
    LHS.resize(2*nVertices, 2*nVertices);
    LHS.setZero();

    // fill the diagonal
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vb_idx = boundary_loop[i];
        // x and y components
        LHS(vb_idx,vb_idx) = 1;
        LHS(vb_idx + nVertices, vb_idx + nVertices) = 1;
    }

    // fill the off-diagonal
    Eigen::MatrixXd e_b(2*nVertices,2);
    e_b.setZero();
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int vb_idx = boundary_loop[i];
        e_b(vb_idx,0) = 1;
        e_b(vb_idx + nVertices,1) = 1;
    }
    LHS -= 1.0/nBoundaryVertices * (e_b  * e_b.transpose());
 */
}


template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::rescaleEigenvectorsToSurfaceArea(const tMesh & mesh, Eigen::MatrixXd & eigenvecs) const
{
    // compute the area of the original shape
    const int nFaces = mesh.getNumberOfFaces();
    const tVecMat2d & aforms = getFirstFundamentalForms(mesh);

    Real totalArea_before = 0;
    for(int i=0;i<nFaces;++i)
    {
        const Eigen::Matrix2d & aform_bar = aforms[i];
        totalArea_before += 0.5*std::sqrt(aform_bar.determinant());
    }

    // compute the area of the new shape according to the largest eigenvalue
    const int nVertices = mesh.getNumberOfVertices();
    const int lastIdx = eigenvecs.cols()-1;
    const auto face2vertices = mesh.getTopology().getFace2Vertices();

    Real totalArea_after = 0;
    for(int i=0;i<nFaces;++i)
    {
        const int idx_v0 = face2vertices(i,0);
        const int idx_v1 = face2vertices(i,1);
        const int idx_v2 = face2vertices(i,2);

        const Eigen::Vector3d v0 = (Eigen::Vector3d() << eigenvecs(idx_v0, lastIdx), eigenvecs(idx_v0 + nVertices, lastIdx), 0).finished();
        const Eigen::Vector3d v1 = (Eigen::Vector3d() << eigenvecs(idx_v1, lastIdx), eigenvecs(idx_v1 + nVertices, lastIdx), 0).finished();
        const Eigen::Vector3d v2 = (Eigen::Vector3d() << eigenvecs(idx_v2, lastIdx), eigenvecs(idx_v2 + nVertices, lastIdx), 0).finished();

        Real tmpsum = 0.0;
        for(int d=0; d<3; d++)
        {
            const int x = d;
            const int y = (d+1)%3;
            const auto rx = v0(x) - v2(x);
            const auto sx = v1(x) - v2(x);
            const auto ry = v0(y) - v2(y);
            const auto sy = v1(y) - v2(y);

            double dblAd = rx*sy - ry*sx;
            tmpsum += dblAd*dblAd;
        }

        totalArea_after += 0.5*std::sqrt(tmpsum);
    }


    // compute the ratio factor that we need to scale up the vertices by
    const Real areaRatioFac = std::sqrt(totalArea_before / totalArea_after);


    // rescale the eigenvectors (they are already centered around the origin so this is fine)
    for(int i=0;i<2*nVertices;++i)
        for(int j=0;j<eigenvecs.cols();++j)
            eigenvecs(i,j) *= areaRatioFac;

    // done
    std::cout << "rescaled the area of the planar shape from \t " << totalArea_after << "\t to \t" << totalArea_before << std::endl;
}

/*



============================ TESTING METHODS BELOW ==============================




 */


template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::checkAreaTriplets(const tMesh & mesh, const std::vector<Eigen::Triplet<Real>> & area_triplets) const
{
    // check area of current image shape
    const int nVertices = mesh.getNumberOfVertices();

    Eigen::VectorXd current_vertices_flat(2*nVertices);
    const auto current_vertices = mesh.getCurrentConfiguration().getVertices();

    for(int i=0;i<nVertices;++i)
    {
        current_vertices_flat(i) = current_vertices(i,0);
        current_vertices_flat(i + nVertices) = current_vertices(i,1);
    }
    Eigen::SparseMatrix<Real> areaMatrixSparse(2*nVertices, 2*nVertices);
    areaMatrixSparse.setFromTriplets(area_triplets.begin(), area_triplets.end());

    Eigen::MatrixXd areaMatrix = Eigen::MatrixXd(areaMatrixSparse);
    const Real area = current_vertices_flat.transpose() * (areaMatrix * current_vertices_flat );

    std::cout << "PROJECTED AREA OF 3D SHAPE = \t " << area << std::endl;
}


template<typename tMesh, MeshLayer layer>
void NonEuclideanConformalMapping<tMesh, layer>::checkBoundaryLoop(const tMesh & mesh, const std::vector<int> & boundary_loop) const
{
    const auto & topo = mesh.getTopology();
    const int nEdges = mesh.getNumberOfEdges();
    const auto edge2vertices = topo.getEdge2Vertices();
    const auto edge2faces = topo.getEdge2Faces();

    const size_t nBoundaryVertices = boundary_loop.size();

    // check boundary loops : are all vertices actually on the boundary?
    for(size_t i=0;i<nBoundaryVertices;++i)
    {
        const int idx_v = boundary_loop[i];
        bool checked = false;

        for(int j=0;j<nEdges;++j)
        {
            const int idx_v0 = edge2vertices(j,0);
            const int idx_v1 = edge2vertices(j,1);

            if(idx_v == idx_v0 or idx_v == idx_v1)
            {
                const int idx_f0 = edge2faces(j,0);
                const int idx_f1 = edge2faces(j,1);

                if(idx_f0 < 0 or idx_f1 < 0)
                {
                    checked = true;
                    break;
                }

            }
        }

        if(not checked) std::cout << "PROBLEM WITH BOUNDARY VERTEX \t " << idx_v << std::endl;
    }

    std::cout << "Done with boundary vertex checking \n" ;
}


#include "Mesh.hpp"
template class NonEuclideanConformalMapping<Mesh, single>;
template class NonEuclideanConformalMapping<BilayerMesh, bottom>;
template class NonEuclideanConformalMapping<BilayerMesh, top>;
