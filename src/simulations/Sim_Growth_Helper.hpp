//
//  Sim_Growth_Helper.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/1/17.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_Growth_Helper_hpp
#define Sim_Growth_Helper_hpp

#include "common.hpp"
#include "Mesh.hpp"

#include "ConformalMappingBoundary.hpp"

#include "EdgeDistortionOperator.hpp"
#include "AreaDistortionOperator.hpp"
#include "MeshQualityOperator.hpp"
#include "EnergyOperatorList.hpp"

#include <igl/boundary_loop.h>

#include "HLBFGS_Wrapper.hpp"

namespace SurfaceMappingSpace
{
    struct BottomLayer
    {
        static const MeshLayer tLayer = bottom;
    };
    struct SingleLayer
    {
        static const MeshLayer tLayer = single;
    };
}

template<typename tMesh>
class SurfaceMapping
{
protected:
    tMesh & mesh;

    // declare tLayer here instead of additional template parameter: if mesh = bilayer, we take the bottom layer, else we take the single layer
    // this way we can use SurfaceMapping both for bilayer meshes and monolayer meshes without changing how we call it
    static constexpr MeshLayer tLayer = std::conditional<std::is_same<tMesh, BilayerMesh>::value, SurfaceMappingSpace::BottomLayer, SurfaceMappingSpace::SingleLayer>::type::tLayer;

    int countNegativeAreas()
    {
        const int nFaces = mesh.getNumberOfFaces();
        const auto face2vertices = mesh.getTopology().getFace2Vertices();
        const auto rvertices = mesh.getRestConfiguration().getVertices();

        int negativeCount = 0;
        for(int i=0;i<nFaces;++i)
        {
            // vertex indices
            const int idx_v0 = face2vertices(i,0);
            const int idx_v1 = face2vertices(i,1);
            const int idx_v2 = face2vertices(i,2);

            // vertex locations
            const Eigen::Vector3d rv0 = rvertices.row(idx_v0);
            const Eigen::Vector3d rv1 = rvertices.row(idx_v1);
            const Eigen::Vector3d rv2 = rvertices.row(idx_v2);

            // computed signed area (assume the rest configuration is planar)
            const Real area = 0.5*(rv0(0)*rv1(1) - rv1(0)*rv0(1) + rv1(0)*rv2(1) - rv2(0)*rv1(1) + rv2(0)*rv0(1) - rv0(0)*rv2(1));

            if(area < 0) negativeCount++;
        }

        return negativeCount;
    }

    void flipRestVerticesIfNegativeAreas()
    {
        const int nFaces = mesh.getNumberOfFaces();
        const int negative_areas = countNegativeAreas();
        if(negative_areas > nFaces/2)
        {
            auto rvertices = mesh.getRestConfiguration().getVertices();
            const int nVertices = mesh.getNumberOfVertices();
            for(int i=0;i<nVertices;++i)
                rvertices(i,0) *= -1; //flip X
            printf("flipping rest vertices : before we had %d negative areas, afterwards %d (out of %d faces)\n", negative_areas, countNegativeAreas(), nFaces);
        }
    }

    void rotate(const Real deg)
    {
        auto rvertices = mesh.getRestConfiguration().getVertices();
        const int nVertices = mesh.getNumberOfVertices();
        for(int i=0;i<nVertices;++i){
            const Real rvertices_temp_0 = rvertices(i,0);
            const Real rvertices_temp_1 = rvertices(i,1);
            rvertices(i,0) = rvertices_temp_0*std::cos(deg) - rvertices_temp_1*std::sin(deg);
            rvertices(i,1) = rvertices_temp_0*std::sin(deg) + rvertices_temp_1*std::cos(deg);
        }
    }

    void setRestVertices(const Eigen::Ref<const Eigen::MatrixXd> vertices_in)
    {
        const int nVertices = mesh.getNumberOfVertices();
        assert(nVertices == vertices_in.rows());
        _unused(nVertices);
        mesh.getRestConfiguration().getVertices() = vertices_in;
    }

    void computeEdgeLengths(Eigen::VectorXd & edgelengths)
    {
        const int nEdges = mesh.getNumberOfEdges();
        edgelengths.resize(nEdges);

        const auto vertices = mesh.getCurrentConfiguration().getVertices();
        const auto edge2vertices = mesh.getTopology().getEdge2Vertices();

        for(int i=0;i<nEdges;++i)
        {
            const int idx_v0 = edge2vertices(i,0);
            const int idx_v1 = edge2vertices(i,1);

            const Eigen::Vector3d v0 = vertices.row(idx_v0);
            const Eigen::Vector3d v1 = vertices.row(idx_v1);
            edgelengths(i) = (v1-v0).norm();
        }
    }

    void computeFaceAreas(Eigen::VectorXd & faceAreas)
    {
        const int nFaces = mesh.getNumberOfFaces();
        faceAreas.resize(nFaces);
        for(int i=0;i<nFaces;++i)
        {
            const ExtendedTriangleInfo & info = mesh.getCurrentConfiguration().getTriangleInfo(i);
            faceAreas(i) = 0.5*info.double_face_area;
        }
    }

    std::pair<Real, Real> computeEdgeGrowth(const Eigen::Ref<const Eigen::VectorXd> edgeLengths_3D) const
    {
        Real max_edge_growth = 0, min_edge_growth = 1e9;
        const int nEdges = mesh.getNumberOfEdges();
        const auto vertices = mesh.getCurrentConfiguration().getVertices();
        const auto edge2vertices = mesh.getTopology().getEdge2Vertices();


        Eigen::VectorXd growthFacs_Edges_2D(nEdges);
        for(int i=0;i<nEdges;++i)
        {
            const int idx_v0 = edge2vertices(i,0);
            const int idx_v1 = edge2vertices(i,1);

            const Eigen::Vector3d v0 = vertices.row(idx_v0);
            const Eigen::Vector3d v1 = vertices.row(idx_v1);
            const Real length_3D = edgeLengths_3D(i);
            const Real length_2D = (v0-v1).norm();
            const Real growth = length_2D/length_3D;

            growthFacs_Edges_2D(i) = growth;
        }

        max_edge_growth = growthFacs_Edges_2D.maxCoeff();
        min_edge_growth = growthFacs_Edges_2D.minCoeff();

        return std::make_pair(min_edge_growth, max_edge_growth);
    }

public:

    SurfaceMapping(tMesh & mesh_in):
    mesh(mesh_in)
    {}


    void laplacian_smoothing_current(const int nIter, const Real lambda)
    {
        // fix boundaries
        std::vector<int> boundary_vertices;
        const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
        igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices); // pick the longest loop
        const int nBoundaryVertices = (int)boundary_vertices.size();
        {
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nBoundaryVertices;++i)
            {
                vertices_bc(boundary_vertices[i],0) = true;
                vertices_bc(boundary_vertices[i],1) = true;
            }
        }

        {
            const int nVertices = mesh.getNumberOfVertices();
            const auto & vertex2faces = mesh.getTopology().getVertex2Faces();
            const auto face2vertices = mesh.getTopology().getFace2Vertices();
            const auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();

            auto vertices = mesh.getCurrentConfiguration().getVertices();

            int iter = 0;

            Eigen::MatrixXd tmpVertices(nVertices, 3);
            tmpVertices = vertices;
            while(iter < nIter)
            {
                for(int vidx=0;vidx<nVertices;++vidx)
                {
                    if(vertices_bc(vidx,0) and vertices_bc(vidx,1)) continue;

                    const Eigen::Vector3d v_old = vertices.row(vidx);

                    Real neighborcnt = 0;
                    Eigen::Vector3d neighborsum;
                    neighborsum.setZero();

                    // loop over faces
                    const int nFaces_vidx = (int)vertex2faces[vidx].size();
                    for(int fidx_rel=0; fidx_rel < nFaces_vidx; ++fidx_rel)
                    {
                        const int fidx = vertex2faces[vidx][fidx_rel];

                        // compute area
                        const TriangleInfo rinfo = mesh.getCurrentConfiguration().getTriangleInfoLite(mesh.getTopology(), fidx);
                        Eigen::MatrixXd rxy_base(3,2);
                        rxy_base << rinfo.e1, rinfo.e2;

                        const Real face_area = 0.5*std::sqrt((rxy_base.transpose() * rxy_base).determinant());

                        // loop over vertices of this face
                        for(int d=0;d<3;++d)
                        {
                            const int vidx_other = face2vertices(fidx, d);
                            if(vidx_other == vidx) continue;

                            neighborsum += vertices.row(vidx_other) * face_area;
                            neighborcnt += face_area;
                        }
                    }

                    neighborsum /= neighborcnt;

                    const Eigen::Vector3d v_new = v_old + lambda * (neighborsum - v_old);
                    for(int d=0;d<3;++d)
                    if(not vertices_bc(vidx,d)) tmpVertices(vidx,d) = v_new(d);
                }

                vertices = tmpVertices;

                iter++;
            }
        }


        // reset boundary conditions
        {
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nBoundaryVertices;++i)
            {
                vertices_bc(boundary_vertices[i],0) = false;
                vertices_bc(boundary_vertices[i],1) = false;
            }
        }
    }



    void laplacian_smoothing_rest(const int nIter, const Real lambda)
    {
        // fix boundaries
        std::vector<int> boundary_vertices;
        const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
        igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices); // pick the longest loop
        const int nBoundaryVertices = (int)boundary_vertices.size();
        {
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nBoundaryVertices;++i)
            {
                vertices_bc(boundary_vertices[i],0) = true;
                vertices_bc(boundary_vertices[i],1) = true;
            }
        }

        {
            const int nVertices = mesh.getNumberOfVertices();
            const auto & vertex2faces = mesh.getTopology().getVertex2Faces();
            const auto face2vertices = mesh.getTopology().getFace2Vertices();
            const auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();

            auto vertices = mesh.getRestConfiguration().getVertices();

            int iter = 0;

            Eigen::MatrixXd tmpVertices(nVertices, 3);
            tmpVertices = vertices;
            while(iter < nIter)
            {
                for(int vidx=0;vidx<nVertices;++vidx)
                {
                    if(vertices_bc(vidx,0) and vertices_bc(vidx,1)) continue;

                    const Eigen::Vector3d v_old = vertices.row(vidx);

                    Real neighborcnt = 0;
                    Eigen::Vector3d neighborsum;
                    neighborsum.setZero();

                    // loop over faces
                    const int nFaces_vidx = (int)vertex2faces[vidx].size();
                    for(int fidx_rel=0; fidx_rel < nFaces_vidx; ++fidx_rel)
                    {
                        const int fidx = vertex2faces[vidx][fidx_rel];

                        // compute area
                        const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), fidx);
                        Eigen::MatrixXd rxy_base(3,2);
                        rxy_base << rinfo.e1, rinfo.e2;

                        const Real face_area = 0.5*std::sqrt((rxy_base.transpose() * rxy_base).determinant());

                        // loop over vertices of this face
                        for(int d=0;d<3;++d)
                        {
                            const int vidx_other = face2vertices(fidx, d);
                            if(vidx_other == vidx) continue;

                            neighborsum += vertices.row(vidx_other) * face_area;
                            neighborcnt += face_area;
                        }
                    }

                    neighborsum /= neighborcnt;

                    const Eigen::Vector3d v_new = v_old + lambda * (neighborsum - v_old);
                    for(int d=0;d<3;++d)
                    if(not vertices_bc(vidx,d)) tmpVertices(vidx,d) = v_new(d);
                }

                vertices = tmpVertices;

                iter++;
            }
        }


        // reset boundary conditions
        {
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nBoundaryVertices;++i)
            {
                vertices_bc(boundary_vertices[i],0) = false;
                vertices_bc(boundary_vertices[i],1) = false;
            }
        }
    }



    void optimalGrowthMapping(const Real edge_growth_fac, const Real face_growth_fac, const Real equilateral_fac)
    {
        // compute the edgelengths and areas of the current configuration
        Eigen::VectorXd edgeLengths_3D;
        computeEdgeLengths(edgeLengths_3D);

        Eigen::VectorXd faceAreas_3D;
        computeFaceAreas(faceAreas_3D);

        // store the current vertices
        Eigen::MatrixXd current_vertices = mesh.getCurrentConfiguration().getVertices();

        // conformal map as initial guess
        conformalMapFree();

        // set as current
        mesh.getCurrentConfiguration().getVertices() = mesh.getRestConfiguration().getVertices();

        // report the min/max growth before
        {
            const auto minmax = computeEdgeGrowth(edgeLengths_3D);
            printf("max growth : %10.10e \t min growth : %10.10e \t ratio : %10.10e \n", minmax.second, minmax.first, minmax.second/minmax.first);
        }

        // optimize for the edge distortion
        {
            const int nVertices = mesh.getNumberOfVertices();
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nVertices;++i) vertices_bc(i,2) = true; // only allow planar translation of vertices

            EdgeDistortionOperator<tMesh> distortionOp_edge(edge_growth_fac, edgeLengths_3D);
            AreaDistortionOperator<tMesh> distortionOp_area(faceAreas_3D, face_growth_fac);
            MeshQualityOperator<tMesh> equilateralOp(equilateral_fac, 2);
            EnergyOperatorList<tMesh> engOp({&distortionOp_edge, &distortionOp_area, &equilateralOp});


            const Real epsMin = std::numeric_limits<Real>::epsilon();
            HLBFGS_Methods::HLBFGS_Energy<tMesh, EnergyOperator<tMesh>, true> hlbfgs_wrapper(mesh, engOp);
            int retval = 0;
            Real eps = 1e-2;
            while(retval == 0 && eps > epsMin)
            {
                eps *= 0.1;
                retval = hlbfgs_wrapper.minimize("surfacemapping_diagnostics.dat", eps);
            }
        }

        // report the min/max growth after
        {
            const auto minmax = computeEdgeGrowth(edgeLengths_3D);
            printf("max growth : %10.10e \t min growth : %10.10e \t ratio : %10.10e \n", minmax.second, minmax.first, minmax.second/minmax.first);
        }

        // copy current to rest, and reset current
        mesh.getRestConfiguration().getVertices() = mesh.getCurrentConfiguration().getVertices();
        mesh.getCurrentConfiguration().getVertices() = current_vertices;
    }

    void conformalMapToDisk()
    {
        Eigen::MatrixXd mapped_vertices;
        ConformalMappingBoundary<tMesh, tLayer> mappingOp;// use conformal mapping: will automatically constrain boundary to a disk
        const bool status = mappingOp.compute_disk(mesh, mapped_vertices);
        std::cout << "done with LSCM mapping, status = \t " << status << std::endl;

        setRestVertices(mapped_vertices);
        flipRestVerticesIfNegativeAreas();
    }

    void conformalMapFree()
    {
        Eigen::MatrixXd mapped_vertices;
        ConformalMappingBoundary<tMesh, tLayer> mappingOp;// use conformal mapping: will automatically constrain boundary to a disk
        mappingOp.computeFree(mesh, mapped_vertices);

        setRestVertices(mapped_vertices);
        flipRestVerticesIfNegativeAreas();
    }


    void conformalMapToRectangle(Real halfEdgeX, Real halfEdgeY, const int corner_pospos, const int corner_posneg, const int corner_negneg, const int corner_negpos, const Real rotate_z_deg)
    {
        const Real edge_growth_fac = 1.0; //min/max edge growth
        const Real face_growth_fac = 10.0; // face growth deviation
        const Real equilateral_fac = 0.0; // equilateral triangle deviation

        // compute the edgelengths and areas of the current configuration
        Eigen::VectorXd edgeLengths_3D;
        computeEdgeLengths(edgeLengths_3D);

        Eigen::VectorXd faceAreas_3D;
        computeFaceAreas(faceAreas_3D);

        if (halfEdgeX < 1e-6 && halfEdgeY < 1e-6){
          halfEdgeX = 0.5*std::sqrt(faceAreas_3D.sum());
          halfEdgeY = halfEdgeX;
        }

        const auto face2vertices = mesh.getTopology().getFace2Vertices();
        const auto vertices = mesh.getCurrentConfiguration().getVertices();

        Eigen::VectorXi boundary_verts;
        Eigen::MatrixXd boundary_verts_vals;

        std::vector<int> boundary_vertices_vec;
        const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
        igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices_vec); // pick the longest loop
        // assign the values
        const int nBoundaryVertices = (int)boundary_vertices_vec.size();
        boundary_verts.resize(nBoundaryVertices);
        boundary_verts_vals.resize(nBoundaryVertices, 2);

        Eigen::Vector2d center;
        center.setZero();

        //divide in quarters
        Real dist = 0.0;
        Real maxdist_pospos = 0.0;
        Real maxdist_posneg = 0.0;
        Real maxdist_negpos = 0.0;
        Real maxdist_negneg = 0.0;

        Eigen::VectorXi corners_vnum(5);
        Eigen::VectorXi corners(5);

        for(size_t i=0;i<nBoundaryVertices;++i)
        {
          const int vidx = boundary_vertices_vec[i];
          boundary_verts(i) = vidx;

          //Clockwise tour

          dist = std::sqrt(std::pow((vertices(vidx,0)-center(0)),2) + std::pow((vertices(vidx,1)-center(1)),2));// + std::pow((vertices(vidx,2)-center(2)),2));

          if (corner_pospos > -1){
            if (corner_pospos == vidx){
              corners(0) = i;
              corners(4) = i;
            }
          } else {
            if (vertices(vidx,0) >= center(0) && vertices(vidx,1) >= center(1)){
              if (dist > maxdist_pospos){
                maxdist_pospos = dist;
                corners_vnum(0) = vidx;
                corners_vnum(4) = vidx;
                corners(0) = i;
                corners(4) = i;
              }
            }
          }

          if (corner_posneg > -1){
            if (corner_posneg == vidx){
              corners(1) = i;
            }
          } else {
            if (vertices(vidx,0) >= center(0) && vertices(vidx,1) < center(1)){
              if (dist > maxdist_posneg){
                maxdist_posneg = dist;
                corners_vnum(1) = vidx;
                corners(1) = i;
              }
            }
          }

          if (corner_negneg > -1){
            if (corner_negneg == vidx){
              corners(2) = i;
            }
          } else {
            if (vertices(vidx,0) < center(0) && vertices(vidx,1) < center(1)){
              if (dist > maxdist_negneg){
                maxdist_negneg = dist;
                corners_vnum(2) = vidx;
                corners(2) = i;
              }
            }
          }

          if (corner_negpos > -1){
            if (corner_negpos == vidx){
              corners(3) = i;
            }
          } else {
            if (vertices(vidx,0) < center(0) && vertices(vidx,1) >= center(1)){
              if (dist > maxdist_negpos){
                maxdist_negpos = dist;
                corners_vnum(3) = vidx;
                corners(3) = i;
              }
            }
          }
        }

        std::cout << "\n\ncorners\n" << corners << std::endl;
        std::cout << "\n\ncorners_vnum\n" << corners_vnum << std::endl;

        boundary_verts_vals(corners(0),0) = center(0)+halfEdgeX;
        boundary_verts_vals(corners(0),1) = center(1)+halfEdgeY;
        std::cout << "\n\ncoord_corner_pospos\n" << boundary_verts_vals(corners(0),0) << "  " << boundary_verts_vals(corners(0),1) << std::endl;

        boundary_verts_vals(corners(1),0) = center(0)+halfEdgeX;
        boundary_verts_vals(corners(1),1) = center(1)-halfEdgeY;

        std::cout << "\n\ncoord_corner_posneg\n" << boundary_verts_vals(corners(1),0) << "  " << boundary_verts_vals(corners(1),1) << std::endl;

        boundary_verts_vals(corners(2),0) = center(0)-halfEdgeX;
        boundary_verts_vals(corners(2),1) = center(1)-halfEdgeY;

        std::cout << "\n\ncoord_corner_negneg\n" << boundary_verts_vals(corners(2),0) << "  " << boundary_verts_vals(corners(2),1) << std::endl;

        boundary_verts_vals(corners(3),0) = center(0)-halfEdgeX;
        boundary_verts_vals(corners(3),1) = center(1)+halfEdgeY;

        std::cout << "\n\ncoord_corner_negpos\n" << boundary_verts_vals(corners(3),0) << "  " << boundary_verts_vals(corners(3),1) << std::endl;

        Eigen::VectorXd disttoprev_orig(nBoundaryVertices);
        Eigen::VectorXd disttoprev_new(nBoundaryVertices);
        Real l_orig = 0.0;
        Real ratio = 0.0;

        int incr = 0;
        int decr = 0;
        for(int i=1;i<5;++i){
          if (corners(i) > corners(i-1))
            incr++;
          else
            decr++;
        }
        std::cout << "\n\nincr\n" << incr << std::endl;
        std::cout << "\n\ndecr\n" << decr << std::endl;

        // Dtection of the corners of a rectangular plate
        if (incr > decr){
          if (corners(0) < corners(1)){
            for(int i=corners(0)+1;i<corners(1)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(0)+1;i<corners(1);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
            }
          } else {
            for(int i=corners(0)+1;i<nBoundaryVertices;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
            l_orig += disttoprev_orig(0);
            for(int i=1;i<corners(1)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(0)+1;i<nBoundaryVertices;++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
            }
            disttoprev_new(0) = disttoprev_orig(0)*ratio;
            boundary_verts_vals(0,0) = center(0)+halfEdgeX;
            boundary_verts_vals(0,1) = boundary_verts_vals(nBoundaryVertices-1,1) - disttoprev_new(0);
            for(int i=1;i<corners(1);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
            }
          }



          l_orig = 0.0;
          if (corners(1) < corners(2)){
            for(int i=corners(1)+1;i<corners(2)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(1)+1;i<corners(2);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)-halfEdgeY;
            }
          } else {
              for(int i=corners(1)+1;i<nBoundaryVertices;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
              l_orig += disttoprev_orig(0);
              for(int i=1;i<corners(2)+1;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              ratio = 2.0*halfEdgeX/l_orig;
              for(int i=corners(1)+1;i<nBoundaryVertices;++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
                boundary_verts_vals(i,1) = center(1)-halfEdgeY;
              }
              disttoprev_new(0) = disttoprev_orig(0)*ratio;
              boundary_verts_vals(0,0) = boundary_verts_vals(nBoundaryVertices-1,0) - disttoprev_new(0);
              boundary_verts_vals(0,1) = center(1)-halfEdgeY;
              for(int i=1;i<corners(2);++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
                boundary_verts_vals(i,1) = center(1)-halfEdgeY;
              }
          }




          l_orig = 0.0;
          if (corners(2) < corners(3)){
            for(int i=corners(2)+1;i<corners(3)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(2)+1;i<corners(3);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)-halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
            }
          } else {
              for(int i=corners(2)+1;i<nBoundaryVertices;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
              l_orig += disttoprev_orig(0);
              for(int i=1;i<corners(3)+1;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              ratio = 2.0*halfEdgeY/l_orig;
              for(int i=corners(2)+1;i<nBoundaryVertices;++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = center(0)-halfEdgeX;
                boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
              }
              disttoprev_new(0) = disttoprev_orig(0)*ratio;
              boundary_verts_vals(0,0) = center(0)-halfEdgeX;
              boundary_verts_vals(0,1) = boundary_verts_vals(nBoundaryVertices-1,1) + disttoprev_new(0);
              for(int i=1;i<corners(3);++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = center(0)-halfEdgeX;
                boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
              }
          }




          l_orig = 0.0;
          if (corners(3) < corners(0)){
            for(int i=corners(3)+1;i<corners(0)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(3)+1;i<corners(0);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
          } else {
            for(int i=corners(3)+1;i<nBoundaryVertices;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
            l_orig += disttoprev_orig(0);
            for(int i=1;i<corners(0)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(3)+1;i<nBoundaryVertices;++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
            disttoprev_new(0) = disttoprev_orig(0)*ratio;
            boundary_verts_vals(0,0) = boundary_verts_vals(nBoundaryVertices-1,0) + disttoprev_new(0);
            boundary_verts_vals(0,1) = center(1)+halfEdgeY;
            for(int i=1;i<corners(0);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
          }
        }






        else {
          if (corners(0) > corners(1)){
            for(int i=corners(1)+1;i<corners(0)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(1)+1;i<corners(0);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
            }
          } else {
            for(int i=corners(1)+1;i<nBoundaryVertices;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
            l_orig += disttoprev_orig(0);
            for(int i=1;i<corners(0)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(1)+1;i<nBoundaryVertices;++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
            }
            disttoprev_new(0) = disttoprev_orig(0)*ratio;
            boundary_verts_vals(0,0) = center(0)+halfEdgeX;
            boundary_verts_vals(0,1) = boundary_verts_vals(nBoundaryVertices-1,1) + disttoprev_new(0);
            for(int i=1;i<corners(0);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)+halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) + disttoprev_new(i);
            }
          }



          l_orig = 0.0;
          if (corners(1) > corners(2)){
            for(int i=corners(2)+1;i<corners(1)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(2)+1;i<corners(1);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)-halfEdgeY;
            }
          } else {
              for(int i=corners(2)+1;i<nBoundaryVertices;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
              l_orig += disttoprev_orig(0);
              for(int i=1;i<corners(1)+1;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              ratio = 2.0*halfEdgeX/l_orig;
              for(int i=corners(2)+1;i<nBoundaryVertices;++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
                boundary_verts_vals(i,1) = center(1)-halfEdgeY;
              }
              disttoprev_new(0) = disttoprev_orig(0)*ratio;
              boundary_verts_vals(0,0) = boundary_verts_vals(nBoundaryVertices-1,0) + disttoprev_new(0);
              boundary_verts_vals(0,1) = center(1)-halfEdgeY;
              for(int i=1;i<corners(1);++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) + disttoprev_new(i);
                boundary_verts_vals(i,1) = center(1)-halfEdgeY;
              }
          }




          l_orig = 0.0;
          if (corners(3) < corners(2)){
            for(int i=corners(3)+1;i<corners(2)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeY/l_orig;
            for(int i=corners(3)+1;i<corners(2);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = center(0)-halfEdgeX;
              boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
            }
          } else {
              for(int i=corners(3)+1;i<nBoundaryVertices;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
              l_orig += disttoprev_orig(0);
              for(int i=1;i<corners(2)+1;++i){
                disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
                l_orig += disttoprev_orig(i);
              }
              ratio = 2.0*halfEdgeY/l_orig;
              for(int i=corners(3)+1;i<nBoundaryVertices;++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = center(0)-halfEdgeX;
                boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
              }
              disttoprev_new(0) = disttoprev_orig(0)*ratio;
              boundary_verts_vals(0,0) = center(0)-halfEdgeX;
              boundary_verts_vals(0,1) = boundary_verts_vals(nBoundaryVertices-1,1) - disttoprev_new(0);
              for(int i=1;i<corners(2);++i){
                disttoprev_new(i) = disttoprev_orig(i)*ratio;
                boundary_verts_vals(i,0) = center(0)-halfEdgeX;
                boundary_verts_vals(i,1) = boundary_verts_vals(i-1,1) - disttoprev_new(i);
              }
          }




          l_orig = 0.0;
          if (corners(0) < corners(3)){
            for(int i=corners(0)+1;i<corners(3)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(0)+1;i<corners(3);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
          } else {
            for(int i=corners(0)+1;i<nBoundaryVertices;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            disttoprev_orig(0) = std::sqrt(std::pow((vertices(boundary_verts(0),0)-vertices(boundary_verts(nBoundaryVertices-1),0)),2) + std::pow((vertices(boundary_verts(0),1)-vertices(boundary_verts(nBoundaryVertices-1),1)),2) + std::pow((vertices(boundary_verts(0),2)-vertices(boundary_verts(nBoundaryVertices-1),2)),2));
            l_orig += disttoprev_orig(0);
            for(int i=1;i<corners(3)+1;++i){
              disttoprev_orig(i) = std::sqrt(std::pow((vertices(boundary_verts(i),0)-vertices(boundary_verts(i-1),0)),2) + std::pow((vertices(boundary_verts(i),1)-vertices(boundary_verts(i-1),1)),2) + std::pow((vertices(boundary_verts(i),2)-vertices(boundary_verts(i-1),2)),2));
              l_orig += disttoprev_orig(i);
            }
            ratio = 2.0*halfEdgeX/l_orig;
            for(int i=corners(0)+1;i<nBoundaryVertices;++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
            disttoprev_new(0) = disttoprev_orig(0)*ratio;
            boundary_verts_vals(0,0) = boundary_verts_vals(nBoundaryVertices-1,0) - disttoprev_new(0);
            boundary_verts_vals(0,1) = center(1)+halfEdgeY;
            for(int i=1;i<corners(3);++i){
              disttoprev_new(i) = disttoprev_orig(i)*ratio;
              boundary_verts_vals(i,0) = boundary_verts_vals(i-1,0) - disttoprev_new(i);
              boundary_verts_vals(i,1) = center(1)+halfEdgeY;
            }
          }
        }


        Eigen::MatrixXd mapped_vertices;
        ConformalMappingBoundary<tMesh, tLayer> mappingOp;// use conformal mapping: will automatically constrain boundary
        const bool status = mappingOp.compute(mesh, boundary_verts, boundary_verts_vals, mapped_vertices, false);

        std::cout << "done with LSCM mapping, status = \t " << status << std::endl;

        setRestVertices(mapped_vertices);
        auto rvertices = mesh.getRestConfiguration().getVertices(); 
        const int nVertices = mesh.getNumberOfVertices();
        for(int i=0;i<nVertices;++i){
            rvertices(i,0) *= -1; //flip X
        }
        rotate(rotate_z_deg);

        // set as current
        mesh.getCurrentConfiguration().getVertices() = mesh.getRestConfiguration().getVertices();

        // report the min/max growth before
        {
            const auto minmax = computeEdgeGrowth(edgeLengths_3D);
            printf("max growth : %10.10e \t min growth : %10.10e \t ratio : %10.10e \n", minmax.second, minmax.first, minmax.second/minmax.first);
        }

        // optimize for the edge distortion
        {
            const int nVertices = mesh.getNumberOfVertices();
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nVertices;++i) vertices_bc(i,2) = true; // only allow planar translation of vertices

            for(int i=0;i<nBoundaryVertices;++i)
            {
              vertices_bc(boundary_verts(i),0) = true;
              vertices_bc(boundary_verts(i),1) = true;
            }

            EdgeDistortionOperator<tMesh> distortionOp_edge(edge_growth_fac, edgeLengths_3D);
            AreaDistortionOperator<tMesh> distortionOp_area(faceAreas_3D, face_growth_fac);
            MeshQualityOperator<tMesh> equilateralOp(equilateral_fac, 2);
            EnergyOperatorList<tMesh> engOp({&distortionOp_edge, &distortionOp_area, &equilateralOp});


            const Real epsMin = std::numeric_limits<Real>::epsilon();
            HLBFGS_Methods::HLBFGS_Energy<tMesh, EnergyOperator<tMesh>, true> hlbfgs_wrapper(mesh, engOp);
            int retval = 0;
            Real eps = 1e-2;
            while(retval == 0 && eps > epsMin)
            {
                eps *= 0.1;
                retval = hlbfgs_wrapper.minimize("surfacemapping_diagnostics.dat", eps);
            }
        }

        // report the min/max growth after
        {
            const auto minmax = computeEdgeGrowth(edgeLengths_3D);
            printf("max growth : %10.10e \t min growth : %10.10e \t ratio : %10.10e \n", minmax.second, minmax.first, minmax.second/minmax.first);
        }

        // copy current to rest, and reset current
        mesh.getRestConfiguration().getVertices() = mesh.getCurrentConfiguration().getVertices();

    }


    void conformalMapKeepBoundary(const Real area_threshold)
    {
        // detect negative areas
        {
            const auto face2vertices = mesh.getTopology().getFace2Vertices();
            const auto vertices = mesh.getCurrentConfiguration().getVertices();
            const int nFaces = mesh.getNumberOfFaces();
            std::vector<int> faces_to_remove;
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

                const Eigen::Vector3d e0 = v1 - v0;
                const Eigen::Vector3d e1 = v2 - v0;
                const Real area = 0.5*(e0.cross(e1)).norm();
                //                        const Real area = 0.5*(v0(0)*v1(1) - v1(0)*v0(1) + v1(0)*v2(1) - v2(0)*v1(1) + v2(0)*v0(1) - v0(0)*v2(1));

                const ExtendedTriangleInfo & info = mesh.getCurrentConfiguration().getTriangleInfo(i);

                if(std::abs(area) < std::numeric_limits<Real>::epsilon())
                    printf("Found small area = %d, %10.10e \t %10.10e, %10.10e \t %10.10e, %10.10e \t %10.10e, %10.10e \t\t %10.10e\n", i, area, v0(0), v0(1), v1(0), v1(1), v2(0), v2(0), 0.5*info.double_face_area);

                if(std::abs(area) < std::numeric_limits<Real>::epsilon())
                    faces_to_remove.push_back(i);
            }

            const int nFaces_to_remove = (int)faces_to_remove.size();
            if(nFaces_to_remove > 0)
            {
                Eigen::MatrixXi faces2vertices_new(nFaces - nFaces_to_remove, 3);

                int idx = 0;
                for(int i=0;i<nFaces;++i)
                {
                    auto result1 = std::find(std::begin(faces_to_remove), std::end(faces_to_remove), i);
                    if (result1 != std::end(faces_to_remove)) continue;

                    faces2vertices_new.row(idx) = face2vertices.row(i);
                    idx++;
                }

                std::cout << "Removing " << nFaces_to_remove << " faces " << std::endl;
                // reinitialize mesh: first store the vertices
                const Eigen::MatrixXd vertices_rest = mesh.getRestConfiguration().getVertices();
                const Eigen::MatrixXd vertices_curr = mesh.getCurrentConfiguration().getVertices();
                // then reinitialize
                Geometry_Dummy geometry_dummy(vertices_rest, faces2vertices_new);
                mesh.init(geometry_dummy);
                // reset current vertices
                mesh.getCurrentConfiguration().getVertices() = vertices_curr;
                mesh.getCurrentConfiguration().update(mesh.getTopology(), mesh.getBoundaryConditions());
            }

        }

        Eigen::VectorXi boundary_vertices;
        Eigen::MatrixXd boundary_vertices_vals;
        {
            // find the boundary_loop

            std::vector<int> boundary_vertices_vec;
            const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
            igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices_vec); // pick the longest loop


            // assign the values
            const int nBoundaryVertices = (int)boundary_vertices_vec.size();
            boundary_vertices.resize(nBoundaryVertices);
            boundary_vertices_vals.resize(nBoundaryVertices, 2);

            const auto vertices = mesh.getCurrentConfiguration().getVertices();

            for(int i=0;i<nBoundaryVertices;++i)
            {
                boundary_vertices(i) = boundary_vertices_vec[i];
                boundary_vertices_vals(i,0) = vertices(boundary_vertices_vec[i],0);
                boundary_vertices_vals(i,1) = vertices(boundary_vertices_vec[i],1);
            }

            // check for the dominant sign of theta
            Eigen::VectorXi sign_dots(nBoundaryVertices);

            std::pair<int, int> sign_count = std::make_pair(0,0);
            for(int i=0;i<nBoundaryVertices;++i)
            {
                const int ip = (i+1)%nBoundaryVertices;

                // get actual tangent vector
                const Real tang_x = boundary_vertices_vals(ip,0) - boundary_vertices_vals(i,0);
                const Real tang_y = boundary_vertices_vals(ip,1) - boundary_vertices_vals(i,1);

                // get tangent vector if it was a circle (CCW)
                const Real theta = std::atan2(boundary_vertices_vals(i,1), boundary_vertices_vals(i,0));
                const Real ref_tang_x = -std::sin(theta);
                const Real ref_tang_y = +std::cos(theta);

                sign_dots(i) = (tang_x * ref_tang_x + tang_y * ref_tang_y) > 0 ? +1 : -1;

                //printf("%d \t %10.10e , %10.10e \t %10.10e, %10.10e \t %10.10e\n",i, tang_x, tang_y, ref_tang_x, ref_tang_y, (tang_x * ref_tang_x + tang_y * ref_tang_y));

                if(sign_dots(i) > 0)
                    sign_count.first += 1;
                else
                    sign_count.second += 1;
            }

            if(sign_count.first != 0 and sign_count.second != 0)
                std::cout << "PROBLEM : the boundary loop is not monotonously increasing in the angle ! " << "\t" << sign_count.first << "\t" << sign_count.second << std::endl;

#if 1==0
            const int dominantThetaSign = (sign_count.first > sign_count.second ? +1 : -1);
            std::cout << "dominant theta sign = " << dominantThetaSign << "\t" << sign_count.first << "\t" << sign_count.second << std::endl;

            // fix the ones that go the wrong way
            for(int i=0;i<nBoundaryVertices;++i)
            {
                if(sign_dots(i) != dominantThetaSign)
                {
                    const Real theta = std::atan2(boundary_vertices_vals(i,1), boundary_vertices_vals(i,0));
                    const Real ref_tang_x = -std::sin(theta);
                    const Real ref_tang_y = +std::cos(theta);


                    // we should space them out a little : find the next one that is correct again
                    int next_idx = i+1;
                    int next_sign = ((boundary_vertices_vals((next_idx)%nBoundaryVertices,0) - boundary_vertices_vals(i,0))*ref_tang_x + (boundary_vertices_vals((next_idx)%nBoundaryVertices,1) - boundary_vertices_vals(i,1))*ref_tang_y) > 0 ? +1 : -1;

                    std::cout << "found one that goes the wrong way " << i << "\t" << sign_dots(i) << "\t" << next_sign << std::endl;
                    while(next_sign != dominantThetaSign)
                    {
                        next_idx++;
                        next_sign = ((boundary_vertices_vals((next_idx)%nBoundaryVertices,0) - boundary_vertices_vals(i,0))*ref_tang_x + (boundary_vertices_vals((next_idx)%nBoundaryVertices,1) - boundary_vertices_vals(i,1))*ref_tang_y) > 0 ? +1 : -1;
                    }
                    std::cout << "found one that is correct again " << next_idx << "\t" << next_sign << std::endl;

                    // now we loop and space them out nicely (linear interpolation)
                    for(int j=i; j<next_idx+1;++j)
                    {
                        const int idx = j%nBoundaryVertices;
                        const Real fraction = (j-i)/((Real)(next_idx - i));
                        std::cout << "\t " << j << "\t" << fraction << std::endl; // not finished because it does not happen
                    }

                }
            }
#endif
        }

        Eigen::MatrixXd mapped_vertices;
        ConformalMappingBoundary<tMesh, tLayer> mappingOp;// use conformal mapping: will automatically constrain boundary to a disk
        const bool status = mappingOp.compute(mesh, boundary_vertices, boundary_vertices_vals, mapped_vertices, false);
        std::cout << "done with LSCM mapping, status = \t " << status << std::endl;

        // set the rest configuration to this
        setRestVertices(mapped_vertices);

        // detect small edges
        {
            const int nBoundaryVertices = boundary_vertices.rows();
            const auto rvertices = mesh.getRestConfiguration().getVertices();
            for(int i=0;i<nBoundaryVertices;++i)
            {
                const Eigen::Vector3d v0 = rvertices.row(boundary_vertices(i));
                const Eigen::Vector3d v1 = rvertices.row(boundary_vertices((i+1)%nBoundaryVertices));
                const Real length = (v1-v0).norm();
                if(length < 1e-6)
                {
                    printf("Found small edge : %10.10e \t %10.10e, %10.10e \t %10.10e, %10.10e\n", length, v0(0), v0(1), v1(0), v1(1));
                }
            }
        }

        // detect small areas
        {
            std::vector<int> faces_to_remove;
            while(true)
            {
                faces_to_remove.clear();

                const auto face2vertices = mesh.getTopology().getFace2Vertices();
                const auto rvertices = mesh.getRestConfiguration().getVertices();
                const int nFaces = mesh.getNumberOfFaces();

                std::vector<int> boundary_vertices_vec;
                const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
                igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices_vec); // pick the longest loop

                int nNegative = 0;
                int nPositive = 0;
                int nZero = 0;
                for(int i=0;i<nFaces;++i)
                {
                    // vertex indices
                    const int idx_v0 = face2vertices(i,0);
                    const int idx_v1 = face2vertices(i,1);
                    const int idx_v2 = face2vertices(i,2);

                    // vertex locations
                    const Eigen::Vector3d v0 = rvertices.row(idx_v0);
                    const Eigen::Vector3d v1 = rvertices.row(idx_v1);
                    const Eigen::Vector3d v2 = rvertices.row(idx_v2);

                    const Real area = 0.5*(v0(0)*v1(1) - v1(0)*v0(1) + v1(0)*v2(1) - v2(0)*v1(1) + v2(0)*v0(1) - v0(0)*v2(1));

                    const bool v0_on_b = (std::find(std::begin(boundary_vertices_vec), std::end(boundary_vertices_vec), idx_v0) != std::end(boundary_vertices_vec));
                    const bool v1_on_b = (std::find(std::begin(boundary_vertices_vec), std::end(boundary_vertices_vec), idx_v1) != std::end(boundary_vertices_vec));
                    const bool v2_on_b = (std::find(std::begin(boundary_vertices_vec), std::end(boundary_vertices_vec), idx_v2) != std::end(boundary_vertices_vec));

                    if(std::abs(area) < area_threshold)
                    {
                        printf("Found small area = %d, %10.10e \t %10.10e, %10.10e \t %10.10e, %10.10e \t %10.10e, %10.10e \t\t %d , %d , %d\n", i, area, v0(0), v0(1), v1(0), v1(1), v2(0), v2(1), v0_on_b, v1_on_b, v2_on_b);
                        nZero++;
                    }
                    else if(area < -area_threshold)
                    {
                        nNegative++;
                    }
                    else
                    {
                        nPositive++;
                    }


                    if(std::abs(area) < area_threshold)
                        faces_to_remove.push_back(i);
                }

                printf("Found %d zero areas, %d negative areas, and %d positive areas, out of %d total areas\n", nZero, nNegative, nPositive, nFaces);

                const int nFaces_to_remove = (int)faces_to_remove.size();

                if(nFaces_to_remove == 0) break;

                //if(nFaces_to_remove > 0)
                {
                    Eigen::MatrixXi faces2vertices_new(nFaces - nFaces_to_remove, 3);

                    int idx = 0;
                    for(int i=0;i<nFaces;++i)
                    {
                        auto result1 = std::find(std::begin(faces_to_remove), std::end(faces_to_remove), i);
                        if (result1 != std::end(faces_to_remove)) continue;

                        faces2vertices_new.row(idx) = face2vertices.row(i);
                        idx++;
                    }

                    std::cout << "Removing " << nFaces_to_remove << " faces " << std::endl;
                    // reinitialize mesh: first store the vertices
                    const Eigen::MatrixXd vertices_rest = mesh.getRestConfiguration().getVertices();
                    const Eigen::MatrixXd vertices_curr = mesh.getCurrentConfiguration().getVertices();
                    // then reinitialize
                    Geometry_Dummy geometry_dummy(vertices_rest, faces2vertices_new);
                    mesh.init(geometry_dummy);
                    // reset current vertices
                    mesh.getCurrentConfiguration().getVertices() = vertices_curr;
                    mesh.getCurrentConfiguration().update(mesh.getTopology(), mesh.getBoundaryConditions());
                }
            }
        }
    }


    void normalizeSurface(const Real target_scale_fac)
    {
        // we center it and make sure it has area 1
        const int nFaces = mesh.getNumberOfFaces();
        Real areaSum = 0.0;

        for(int i=0;i<nFaces;++i)
        {
            const ExtendedTriangleInfo & info = mesh.getCurrentConfiguration().getTriangleInfo(i);
            areaSum += 0.5*info.double_face_area;
        }

        const Real rescale_fac = target_scale_fac/std::sqrt(areaSum);

        auto vertices = mesh.getCurrentConfiguration().getVertices();
        auto rvertices = mesh.getRestConfiguration().getVertices();
        const int nVertices = mesh.getNumberOfVertices();
        Eigen::Vector3d centroid;
        centroid.setZero();
        for(int i=0;i<nVertices;++i)
            centroid += vertices.row(i);
        centroid /= nVertices;

        for(int i=0;i<nVertices;++i)
            for(int d=0;d<3;++d)
            {
                vertices(i,d) =  (vertices(i,d) - centroid(d)) * rescale_fac;
                rvertices(i,d) = vertices(i,d);
            }

        mesh.getCurrentConfiguration().update(mesh.getTopology(), mesh.getBoundaryConditions());
    }
};


#endif /* Sim_Growth_Helper_h */
