//
//  Sim_Inverse.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_Inverse_hpp
#define Sim_Inverse_hpp

#include "Sim.hpp"
#include "GrowthFacs.hpp"
#include "Geometry.hpp"
#include "ComputeCurvatures.hpp"
#include "ParametricSurfaceLibrary.hpp"
#include "MaterialProperties.hpp"

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>

#include "Sim_Growth_Helper.hpp"

template<typename tMesh>
class Sim_Inverse : public Sim<tMesh>
{
public:
    typedef std::function<std::pair<Real,Real>(const Real, const Real)> PlanarMapFunc;
protected:

    // some shortcuts
    using Sim<tMesh>::mesh;
    using Sim<tMesh>::parser;
    using Sim<tMesh>::tag;

    void initDiskToHemisphere(const Real diskRadius, const Real sphereRadius, const int nEdgePoints)
    {
        // init mesh
        CircularPlate geometry(diskRadius, nEdgePoints, false);
        mesh.init(geometry);

        // init surface
        ParametricSphericalShell target_surface(sphereRadius, M_PI, true);
        const auto mapFunc = [diskRadius, sphereRadius](const Real x, const Real y)
        {
            const Real uu = x;
            const Real vv = y;
            return std::make_pair(uu,vv);
        };

        setCurrentConfigurationToSurface(target_surface, mapFunc); // use mapFunc so we dont use getExtent_U/V from spherical shell (so that we can do partial hemispheres)
    }



    void initDiskToHemisphere_NoScaling(const Real diskRadius, const Real sphereRadius, const int nEdgePoints)
    {
        // init mesh
        CircularPlate geometry(diskRadius, nEdgePoints, false);
        mesh.init(geometry);

        // init surface
        ParametricSphericalShell target_surface(sphereRadius, M_PI);
        const auto mapFunc = [diskRadius, sphereRadius](const Real x, const Real y)
        {
            const Real uu = x/diskRadius;
            const Real vv = y/diskRadius;
            return std::make_pair(uu,vv);
        };

        setCurrentConfigurationToSurface(target_surface, mapFunc); // use mapFunc so we dont use getExtent_U/V from spherical shell (so that we can do partial hemispheres)
    }



    void initRectangleToPartsphere(const Real Lx, const Real Ly, const Real relArea, const Real sphereRadius, const bool clamped)
    {

        if (clamped){
          RectangularPlate_3clampvert geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        } else{
          RectangularPlate_RightAngle geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        }

        // init surface
        ParametricPartSphere target_surface(sphereRadius);

        // use mapFunc so we dont use getExtent_U/V from spherical shell (so that we can do partial hemispheres)
        setCurrentConfigurationToSurface(target_surface, Lx, Ly);
    }


    void initDiskToEnneper(const Real diskRadius, const int enneperFac, const int nEdgePoints)
    {
        // init mesh
        CircularPlate geometry(diskRadius, nEdgePoints, false);
        mesh.init(geometry);

        // init surface
        ParametricEnneper target_surface(enneperFac);

        // create mapping from xy to r,theta
        const auto mapFunc = [diskRadius](const Real x, const Real y)
        {
            const Real rad = std::sqrt(x*x + y*y);
            const Real r = rad/diskRadius;
            const Real theta = std::atan2(y,x);
            return std::make_pair(r,theta);
        };
        setCurrentConfigurationToSurface(target_surface, mapFunc);
    }


    void initRectangleToCylinder(const Real Lx, const Real Ly, const Real cylinderRadius, const Real cylinderLength, const Real cylinderAngle, const Real relArea, const bool clamped)
    {
        // init mesh
        if (clamped){
          RectangularPlate_3clampvert geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        } else{
          RectangularPlate_RightAngle geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        }

        // init surface
        ParametricCylinder target_surface(cylinderRadius, cylinderLength, cylinderAngle); // u : zero to 2pi, v: 0 to L
        setCurrentConfigurationToSurface(target_surface, Lx, Ly);
    }

    void initRectangleToHalfCylinder(const Real Lx, const Real Ly, const Real cylinderRadius, const Real cylinderLength, const Real relArea)
    {
        // init mesh
        RectangularPlate geometry(Lx, Ly, relArea, {false,false}, {false,false});
        mesh.init(geometry);

        // init surface
        ParametricCylinder target_surface(cylinderRadius, cylinderLength); // u : 0 to L, v : zero to pi

        const auto mapFunc = [cylinderRadius, cylinderLength, Lx, Ly](const Real xx, const Real yy)
        {
            const Real xrel = 0.5*(xx + Lx)/Lx; // between 0 and 1
            const Real vv = xrel*cylinderLength; // between 0 and length (v is the x-dimension in the cylinder paramterization)

            const Real yrel = 0.5*(yy + Ly)/Ly; // between 0 and 1
            const Real uu =  yrel*M_PI; // between 0 and pi (u is the theta-direction)

            return std::make_pair(uu,vv);
        };
        setCurrentConfigurationToSurface(target_surface, mapFunc);
    }

    void initHalfCylinderToHalfCylinder(const Real Lx, const Real Ly, const Real cylinderRadius, const Real cylinderLength, const Real relArea, const bool fixbc)
    {
        // init mesh and set target configuration
        initRectangleToHalfCylinder(Lx, Ly, cylinderRadius, cylinderLength, relArea);

        // set boundary conditions
        if(fixbc)
        {
#if 1==0
            const auto rvertices = mesh.getRestConfiguration().getVertices();
            const int nVertices = mesh.getNumberOfVertices();
            auto vertices_bc = mesh.getBoundaryConditions().getVertexBoundaryConditions();
            for(int i=0;i<nVertices;++i)
            {
                const bool curvedEdge = std::abs(std::abs(vertices(i,0)) - 0.5*Lx) < 1e-9;
                const bool straightEdge = std::abs(std::abs(vertices(i,1)) - 0.5*Ly) < 1e-9;

                if(curvedEdge)
                {
                    vertices_bc(i,2) = true;
                }
                if(straightEdge)
                {
                    // TODO!!
                }

            }
#endif
        }
    }


    void initRectangleToCylinderEquilateral(const Real Lx, const Real Ly, const Real cylinderRadius, const Real cylinderLength, const Real edgeLength)
    {
        // init mesh
        RectangularPlate_RightAngle geometry(Lx, Ly, edgeLength);
        mesh.init(geometry);

        // init surface
        ParametricCylinder target_surface(cylinderRadius, cylinderLength); // u : zero to 2pi, v: 0 to L
        setCurrentConfigurationToSurface(target_surface, Lx, Ly);
    }

    void initDiskToSaddle(const Real diskRadius, const Real afac, const Real bfac, const int nEdgePoints)
    {
        // init mesh
        CircularPlate geometry(diskRadius, nEdgePoints, false);
        mesh.init(geometry);

        // init surface
        ParametricHyperbolicParaboloid target_surface(afac, bfac);
        setCurrentConfigurationToSurface(target_surface, diskRadius, diskRadius);
    }

    void initRectangleToSaddle(const Real Lx, const Real Ly, const Real relArea, const Real afac, const Real bfac, const bool clamped)
    {
        // init mesh
        if (clamped){
          RectangularPlate_3clampvert geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        } else{
          RectangularPlate_RightAngle geometry(Lx, Ly, relArea, false, false);
          mesh.init(geometry);
        }

        // init surface
        ParametricHyperbolicParaboloid target_surface(afac, bfac);
        setCurrentConfigurationToSurface(target_surface, Lx, Ly);
    }





    void initInverseProblem()
    {
        const std::string geometryCase = parser.template parse<std::string>("-geometry", "");

        if(geometryCase == "disk2hemisphere")
        {
            const Real diskRadius = 1.0;
            const Real sphereRadius = parser.template parse<Real>("-sphereradrel", 1.0) * diskRadius;
            const int nPointsAlongBoundary = parser.template parse<int>("-res", 128);
            initDiskToHemisphere(diskRadius, sphereRadius, nPointsAlongBoundary);
        }
        else if(geometryCase == "rectangle2partsphere")
        {
            const Real res = parser.template parse<Real>("-res", 0.01); //quantity of nodes per boundary
            const Real Lx = parser.template parse<Real>("-lx", 0.5);
            const Real Ly = parser.template parse<Real>("-ly", 0.5);
            const Real relArea = 2.0*Lx*res;

            const Real sphereRadius = parser.template parse<Real>("-sphereradrel", 3.0) * Lx;

            const bool clamped = parser.template parse<bool>("-clamped", true);
            initRectangleToPartsphere(Lx, Ly, relArea, sphereRadius, clamped);
        }
        else if(geometryCase == "disk2enneper")
        {
            const int enneperFac = parser.template parse<int>("-n", 2);
            const Real diskRadius = 1.0;
            const int nPointsAlongBoundary = parser.template parse<int>("-res", 128);
            initDiskToEnneper(diskRadius, enneperFac, nPointsAlongBoundary);
        }
        else if(geometryCase == "rectangle2cylinder")
        {
            const Real Ly = parser.template parse<Real>("-ly", 0.5);
            const Real cylinderR = 1.0;
            const Real cylinderH = 2.0*Ly;
            const Real cylinderTheta = parser.template parse<Real>("-thetarel", 1.0)*2.0*M_PI;
            const Real Lx = 0.5 * cylinderTheta * cylinderR;// this will be the circumferential guy
            const Real res = parser.template parse<Real>("-res", 0.01);
            const Real relArea = 2.0*Lx*res;
            const bool clamped = parser.template parse<bool>("-clamped", true);
            initRectangleToCylinder(Lx, Ly, cylinderR, cylinderH, cylinderTheta, relArea, clamped);
        }
        else if(geometryCase == "rectangle2halfcylinder")
        {
            const Real Lx = 1.0;// this will be the circumferential guy
            const Real Ly = 0.5*M_PI; // this will be the length
            const Real cylinderR = 1.0;
            const Real cylinderH = 2.0*Lx;
            const Real res = parser.template parse<Real>("-res", 0.001);
            const Real relArea = 2.0*Lx*res;
            initRectangleToHalfCylinder(Lx, Ly, cylinderR, cylinderH, relArea);
        }
        else if(geometryCase == "halfcylinder2halfcylinder")
        {
            const Real Lx = 1.0;// this will be the circumferential guy
            const Real Ly = 0.5*M_PI; // this will be the length
            const Real cylinderR = 1.0;
            const Real cylinderH = 2.0*Lx;
            const Real res = parser.template parse<Real>("-res", 0.01);
            const Real relArea = 2.0*Lx*res;
            const bool fixbc = parser.template parse<bool>("-fixbc", false);
            initHalfCylinderToHalfCylinder(Lx, Ly, cylinderR, cylinderH, relArea, fixbc);
        }
        else if(geometryCase == "rectangle2cylinder_equilateral")
        {
            const Real Lx = M_PI;// this will be the circumferential guy
            const Real Ly = 1.0; // this will be the length
            const Real cylinderR = 1.0;
            const Real cylinderH = 2.0*Ly;
            const Real res = parser.template parse<Real>("-edgelength", 0.01);
            const Real relArea = 2.0*Lx*res;
            initRectangleToCylinderEquilateral(Lx, Ly, cylinderR, cylinderH, relArea);
        }
        else if(geometryCase == "disk2saddle")
        {
            const Real diskRadius = 1.0;
            const Real saddle_afac = parser.template parse<Real>("-afac", 2.0);
            const Real saddle_bfac = parser.template parse<Real>("-bfac", 2.0);
            const int nPointsAlongBoundary = parser.template parse<int>("-res", 128);
            initDiskToSaddle(diskRadius, saddle_afac, saddle_bfac, nPointsAlongBoundary);
        }
        else if(geometryCase == "rectangle2saddle")
        {
            const Real res = parser.template parse<Real>("-res", 0.01); //quantity of nodes per boundary
            const Real Lx = parser.template parse<Real>("-lx", 0.5);
            const Real Ly = parser.template parse<Real>("-ly", 0.5);
            const Real relArea = 2.0*Lx*res;

            const Real saddle_afac = parser.template parse<Real>("-afac", 2.0);
            const Real saddle_bfac = parser.template parse<Real>("-bfac", 2.0);

            const bool clamped = parser.template parse<bool>("-clamped", true);

            initRectangleToSaddle(Lx, Ly, relArea, saddle_afac, saddle_bfac, clamped);
        }
        else if(geometryCase == "external")
        {
           // fname and bname are full file names
           const std::string fname = parser.template parse<std::string>("-filename", ""); // current state
           const std::string bname = parser.template parse<std::string>("-basename", ""); // rest state

           const bool clamped_vert = parser.template parse<bool>("-clamped_vert", false);
           const int laplace_iter_current = parser.template parse<int>("-laplace_iter_current", 0);
           const Real laplace_lambda_current = parser.template parse<Real>("-laplace_lambda_current", 0.75);

           Eigen::MatrixXi faces_r;
           Eigen::MatrixXd vertices_r;

           IOGeometry geometry(fname);
           mesh.init(geometry);

           if (laplace_iter_current > 0){
             laplacian_smoothing_current(laplace_iter_current, laplace_lambda_current);
           }

           finishSetupFromExternalMesh();

           vertices_r = mesh.getRestConfiguration().getVertices();
           auto vertices_bc_r = mesh.getBoundaryConditions().getVertexBoundaryConditions();
           faces_r = mesh.getTopology().getFace2Vertices();

           if (clamped_vert == true){
             std::vector<int> boundary_vertices_vec;
             const Eigen::MatrixXi & ref_face2vertices = mesh.getTopology().getFace2Vertices();
             igl::boundary_loop<Eigen::MatrixXi, int>(ref_face2vertices, boundary_vertices_vec); // pick the longest loop
             const int nBoundaryVertices = (int)boundary_vertices_vec.size();

             for(size_t i=0;i<nBoundaryVertices;++i)
             {
               const int vidx = boundary_vertices_vec[i];
               vertices_bc_r(vidx,0) = true;
               vertices_bc_r(vidx,1) = true;
               vertices_bc_r(vidx,2) = true;
             }


             Geometry_Dummy_With_BC geometry_r(vertices_r, faces_r, vertices_bc_r);
             const bool clamped_edge = parser.template parse<bool>("-clamped_edge", true);
             mesh.init_rest(geometry_r, clamped_edge);
           } else {
             Geometry_Dummy_With_BC geometry_r(vertices_r, faces_r, vertices_bc_r);
             mesh.init_rest(geometry_r);
           }
        }


        else if(geometryCase == "external_dat")
        {
            // filetag and basetag are file names WITHOUT format, i.e., without ".vtp" or ".obj"
            const std::string filetag = parser.template parse<std::string>("-filetag", ""); // current state
            const std::string basetag = parser.template parse<std::string>("-basetag", ""); // rest state

            // restart the mesh from the filetag case
            mesh.readFromFile(filetag);

            // load the vertices and edge directors of the base state
            Eigen::MatrixXd vertexdata_rest;
            Eigen::VectorXd edgedata_rest;
            helpers::read_matrix_binary(basetag+"_vertices.dat", vertexdata_rest);
            helpers::read_matrix_binary(basetag+"_edgedirs.dat", edgedata_rest);

            mesh.getRestConfiguration().getVertices() = vertexdata_rest;
            mesh.getRestConfiguration().getEdgeDirectors() = edgedata_rest;
        }
        else
        {
            std::cout << "No valid geometry defined. Options are \n";
            std::cout << "\t -geometry disk2hemisphere\n";
            std::cout << "\t -geometry square2hemisphere\n";
            std::cout << "\t -geometry disk2enneper\n";
            std::cout << "\t -geometry rectangle2catenoid\n";
            std::cout << "\t -geometry rectangle2cylinder\n";
            std::cout << "\t -geometry rectangle2halfcylinder\n";
            std::cout << "\t -geometry halfcylinder2halfcylinder\n";
            std::cout << "\t -geometry disk2saddle\n";
            std::cout << "\t -geometry annulus2pseudosphere\n";
            std::cout << "\t -geometry external\n";
            std::cout << "\t -geometry external_dat\n";
            helpers::catastrophe("no valid geometry", __FILE__, __LINE__);
        }

        this->dumpWithNormals(tag+"_target");

        //dumpCurvatures(tag+"_target_curvs");

    }

    virtual void addCurvaturesToWriter(WriteVTK & writer)
    {
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd gauss(nFaces);
        Eigen::VectorXd mean(nFaces);
        Eigen::VectorXd PrincCurv1(nFaces);
        Eigen::VectorXd PrincCurv2(nFaces);
        Eigen::VectorXd CurvX(nFaces);
        Eigen::VectorXd CurvY(nFaces);

        Eigen::Vector3d Dir1 = (Eigen::Vector3d() <<  1, 0, 0).finished();
        Eigen::Vector3d Dir2 = (Eigen::Vector3d() <<  0, 1, 0).finished();

        ComputeCurvatures<tMesh> computeCurvatures;
        //computeCurvatures.compute(mesh, gauss, mean);
        computeCurvatures.computeDir(mesh, gauss, mean, PrincCurv1, PrincCurv2, CurvX, CurvY, Dir1, Dir2);
        writer.addScalarFieldToFaces(gauss, "gauss");
        writer.addScalarFieldToFaces(mean, "mean");
        writer.addScalarFieldToFaces(PrincCurv1, "PrincCurv1");
        writer.addScalarFieldToFaces(PrincCurv2, "PrincCurv2");
        writer.addScalarFieldToFaces(CurvX, "CurvX");
        writer.addScalarFieldToFaces(CurvY, "CurvY");
    }

    virtual void dumpCurvatures(const std::string filename)
    {
        const auto cvertices = mesh.getCurrentConfiguration().getVertices();
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();

        WriteVTK writer(cvertices, cface2vertices);
        addCurvaturesToWriter(writer);
        writer.write(filename);
    }


    virtual void laplacian_smoothing_current(const int laplace_iter, const Real laplace_lambda)
    {
      SurfaceMapping<tMesh> mapping(mesh);

      mapping.laplacian_smoothing_current(laplace_iter, laplace_lambda);
      auto cvertices = mesh.getCurrentConfiguration().getVertices();
      const auto face2vertices = mesh.getTopology().getFace2Vertices();
      WriteVTK writer(cvertices, face2vertices);
      writer.write(tag+"_target_laplace");

      const std::string fname = tag+"_target_laplace.vtp";
      IOGeometry geometry(fname);
      mesh.init(geometry);
    }


    virtual void finishSetupFromExternalMesh()
    {

        SurfaceMapping<tMesh> mapping(mesh);

        const bool normalizeTarget = parser.template parse<bool>("-normalize", false);
        if(normalizeTarget)
        {
            const Real target_scale_fac = parser.template parse<Real>("-rescale_fac", 1.0);
            mapping.normalizeSurface(target_scale_fac);
        }

        const std::string basename = parser.template parse<std::string>("-basename", "");
        if(basename == "")
        {
            // we dont have a base file specified : need to do a custom flattening

            // first set up the first fundamental form (euclidean now - set both bottom and top)
            // so that we can do conformal mapping of any form
            mesh.getRestConfiguration().setFormsFromVertices(mesh.getTopology(), mesh.getBoundaryConditions());

            // then parse the options : we will check them in order (ie if more than one true : first one is executed)
            // if all are false : we do conformal mapping with free boundary
            const bool initDisk = parser.template parse<bool>("-initdisk", false);
            const bool initRectangle = parser.template parse<bool>("-initrectangle", false);
            const bool keepBoundary = parser.template parse<bool>("-keepboundary", false);
            const bool optimalGrowth = parser.template parse<bool>("-optimalgrowth", false);

            if(initDisk)
            {
                // conformal mapping with boundary in a disk (radius is maximum extent of original shape from centroid)
                mapping.conformalMapToDisk();
            } else if(initRectangle)
            {
                // conformal mapping with boundary in a rectangle with manually set lx and ly
                Real lx = parser.template parse<Real>("-lx", 0.0);
                Real ly = parser.template parse<Real>("-ly", 0.0);

                const int corner_pospos = parser.template parse<int>("-corner_pospos", -1);
                const int corner_posneg = parser.template parse<int>("-corner_posneg", -1);
                const int corner_negneg = parser.template parse<int>("-corner_negneg", -1);
                const int corner_negpos = parser.template parse<int>("-corner_negpos", -1);

                const Real rotate_z_deg = parser.template parse<Real>("-rotate_z_deg", -0.5)*M_PI;

                const int laplace_iter_current = parser.template parse<int>("-laplace_iter_current", 0);

                mapping.conformalMapToRectangle(lx, ly, corner_pospos, corner_posneg, corner_negneg, corner_negpos, rotate_z_deg);

                if (laplace_iter_current > 0){
                  const std::string fname = tag+"_target_laplace.vtp";
                  IOGeometry geometry_c(fname);
                  mesh.init_current(geometry_c);
                } else {
                  const std::string fname = parser.template parse<std::string>("-filename", "");
                  IOGeometry geometry_c(fname);
                  mesh.init_current(geometry_c);
                }
            }
            else if(keepBoundary)
            {
                // conformal mapping within the current boundary
                const Real area_threshold = parser.template parse<Real>("-area_threshold", std::numeric_limits<Real>::epsilon());

                mapping.conformalMapKeepBoundary(area_threshold);
            }
            else if(optimalGrowth)
            {
                // custom mapping that attempts to minimize the ratio max_growth/min_growth over all edges (ie over all linear growth factors of the edges) - together with some regularizers including facefac
                const Real edge_growth_fac = parser.template parse<Real>("-edgefac", 1.0); //min/max edge growth
                const Real face_growth_fac = parser.template parse<Real>("-facefac", 10.0); // face growth deviation
                const Real equilateral_fac = parser.template parse<Real>("-equifac", 0.0); // equilateral triangle deviation

                mapping.optimalGrowthMapping(edge_growth_fac, face_growth_fac, equilateral_fac);
            }
            else
            {
                // conformal mapping with free boundary
                mapping.conformalMapFree();
            }

            // optional laplacian smoothing
            const int laplace_iter_rest = parser.template parse<int>("-laplace_iter_rest", 0);
            const Real laplace_lambda_rest = parser.template parse<Real>("-laplace_lambda_rest", 0.75);

            if(laplace_iter_rest > 0) mapping.laplacian_smoothing_rest(laplace_iter_rest, laplace_lambda_rest);
        }
        else
        {

            // basename is true
            IOGeometry geometry_r(basename);
            // mesh.init_rest(geometry_r);
            auto restvertices = mesh.getRestConfiguration().getVertices();
            const auto face2vertices = mesh.getTopology().getFace2Vertices();

            Eigen::MatrixXd restvertices_in;
            Eigen::MatrixXi face2vertices_in;
            Eigen::MatrixXb vertices_bc_in;
            geometry_r.get(restvertices_in, face2vertices_in, vertices_bc_in);

            restvertices = restvertices_in;
            const Real face_norm = (face2vertices - face2vertices_in).norm();
            printf("face_norm = %10.10e\n", face_norm);
        }

        this->dumpWithNormals(tag+"_org_normals_base", true);
        this->dumpWithNormals(tag+"_org_normals_final", false);


        const bool flipRestX = parser.template parse<bool>("-fliprestX", false);
        const bool flipRestY = parser.template parse<bool>("-fliprestY", false);
        const bool flipRestZ = parser.template parse<bool>("-fliprestZ", false);

        if(flipRestX or flipRestY or flipRestZ)
        {
            {
                const int nVertices = mesh.getNumberOfVertices();
                auto verts = mesh.getRestConfiguration().getVertices();
                for(int i=0;i<nVertices;++i)
                {
                    if(flipRestX) verts(i,0) *= -1;
                    if(flipRestY) verts(i,1) *= -1;
                    if(flipRestZ) verts(i,2) *= -1;
                }
            }
            mesh.updateDeformedConfiguration();
            this->dumpWithNormals(tag+"_org_normals_base_flip", true);
        }

        const bool flipTargetX = parser.template parse<bool>("-fliptargetX", false);
        const bool flipTargetY = parser.template parse<bool>("-fliptargetY", false);
        const bool flipTargetZ = parser.template parse<bool>("-fliptargetZ", false);

        if(flipTargetX or flipTargetY or flipTargetZ)
        {
            {
                const int nVertices = mesh.getNumberOfVertices();
                auto verts = mesh.getCurrentConfiguration().getVertices();
                for(int i=0;i<nVertices;++i)
                {
                    if(flipTargetX) verts(i,0) *= -1;
                    if(flipTargetY) verts(i,1) *= -1;
                    if(flipTargetZ) verts(i,2) *= -1;
                }
            }
            mesh.updateDeformedConfiguration();
            this->dumpWithNormals(tag+"_org_normals_final_flip", false);
        }

        this->dumpObjWithTextureCoordinates(tag+"_uv");
    }

    void deformIntoSphere(const Real radius)
    {
        auto vertices = mesh.getCurrentConfiguration().getVertices();
        const auto & restvertices = mesh.getRestConfiguration().getVertices();
        const int nVertices = mesh.getNumberOfVertices();
        for(int i=0;i<nVertices;++i)
        {

            // treat it as a parabolic surface
            const Real planarDistSq = std::pow(restvertices(i,0),2) + std::pow(restvertices(i,1),2);
            vertices(i,2) = planarDistSq/(radius*radius);
        }
    }

    void setCurrentConfigurationToSurface(const ParametricSurface & surface, const Real Lx, const Real Ly)
    {
        const auto u_extent = surface.getExtent_U();
        const auto v_extent = surface.getExtent_V();

        const auto mapFunc = [u_extent, v_extent, Lx, Ly](const Real xx, const Real yy)
        {
            const Real xrel = 0.5*(xx + Lx)/Lx; // between 0 and 1
            const Real uu = u_extent.first + xrel*(u_extent.second - u_extent.first);

            const Real yrel = 0.5*(yy + Ly)/Ly; // between 0 and 1
            const Real vv = v_extent.first + yrel*(v_extent.second - v_extent.first);

            return std::make_pair(uu,vv);
        };

        setCurrentConfigurationToSurface(surface, mapFunc);
    }

    void setCurrentConfigurationToSurface(const ParametricSurface & surface, const PlanarMapFunc & mapFunc)
    {
        const auto restvertices = mesh.getRestConfiguration().getVertices();
        auto vertices = mesh.getCurrentConfiguration().getVertices();
        const int nVertices = mesh.getNumberOfVertices();
        for(int i=0;i<nVertices;++i)
        {
            const auto uv = mapFunc(restvertices(i,0), restvertices(i,1));

            const Real uu = uv.first;
            const Real vv = uv.second;

            const Eigen::Vector3d pos = surface(uu,vv);
            for(int d=0;d<3;++d)
                vertices(i,d) = pos(d);
        }

        const auto edge2vertices = mesh.getTopology().getEdge2Vertices();
        const int nEdges = mesh.getTopology().getNumberOfEdges();
        const Real fd_eps = 1e-9;
        Eigen::MatrixXd edgeNormals(nEdges,3);
        for(int i=0;i<nEdges;++i)
        {
            const int idx_v0 = edge2vertices(i,0);
            const int idx_v1 = edge2vertices(i,1);

            const Real xx = 0.5*(restvertices(idx_v0,0) + restvertices(idx_v1,0));
            const Real yy = 0.5*(restvertices(idx_v0,1) + restvertices(idx_v1,1));

            const auto uv = mapFunc(xx, yy);
            const Real uu = uv.first;
            const Real vv = uv.second;

//            const Eigen::Vector3d norm = surface.getNormal(uu,vv);
            const Eigen::Vector3d xu_uv = surface.get_xu(uu, vv);
            const Eigen::Vector3d xv_uv = surface.get_xv(uu, vv);

            const Real du_dx = (mapFunc(xx + fd_eps, yy).first - mapFunc(xx - fd_eps, yy).first)/(2.0*fd_eps);
            const Real dv_dx = (mapFunc(xx + fd_eps, yy).second - mapFunc(xx - fd_eps, yy).second)/(2.0*fd_eps);
            const Real du_dy = (mapFunc(xx, yy + fd_eps).first - mapFunc(xx, yy - fd_eps).first)/(2.0*fd_eps);
            const Real dv_dy = (mapFunc(xx, yy + fd_eps).second - mapFunc(xx, yy - fd_eps).second)/(2.0*fd_eps);

            const Eigen::Vector3d xu = xu_uv * du_dx + xv_uv * dv_dx;
            const Eigen::Vector3d xv = xu_uv * du_dy + xv_uv * dv_dy;

            const Eigen::Vector3d norm = (xu.cross(xv)).normalized();

            for(int d=0;d<3;++d)
                edgeNormals(i,d) = norm(d);
        }

        mesh.updateDeformedConfiguration();
        mesh.getCurrentConfiguration().setEdgeDirectors_fromarray(mesh.getTopology(), edgeNormals);
    }

    void computeQuadraticForms(tVecMat2d & firstFF, tVecMat2d & secondFF) const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const auto currentState = mesh.getCurrentConfiguration();

        firstFF.resize(nFaces);
        secondFF.resize(nFaces);

        for(int i=0;i<nFaces;++i)
        {
            firstFF[i] = currentState.getTriangleInfo(i).computeFirstFundamentalForm();
            secondFF[i] = currentState.getTriangleInfo(i).computeSecondFundamentalForm();
        }
    }

    Eigen::VectorXd computeDistancesToBoundary(const Eigen::Ref<const Eigen::MatrixXd> points) const
    {
        // get the boundary edges (each row contains the two vertex indices of the boundary)
        const auto face2vertices = mesh.getTopology().getFace2Vertices();
        Eigen::MatrixXi boundary_edges;
        igl::boundary_facets<Eigen::MatrixXi, Eigen::MatrixXi>(face2vertices, boundary_edges);
        const int nBoundaryEdges = boundary_edges.rows();

        // get the vertex locations
        const auto vertices = mesh.getRestConfiguration().getVertices();

        // loop over all points and compute the distances
        const int nPoints = points.rows();
        Eigen::VectorXd retval(nPoints);

#pragma omp parallel for
        for(int i=0;i<nPoints;++i)
        {
            const Eigen::Vector3d P = points.row(i);
            retval(i) = 1e9;
            for(int j=0;j<nBoundaryEdges;++j)
            {
                const int idx_v0 = boundary_edges(j,0);
                const int idx_v1 = boundary_edges(j,1);
                const Eigen::Vector3d v0 = vertices.row(idx_v0);
                const Eigen::Vector3d v1 = vertices.row(idx_v1);

                // source : http://geomalgorithms.com/a02-_lines.html
                // method : dist_Point_to_Segment
                const Eigen::Vector3d v = v1 - v0;
                const Eigen::Vector3d w = P - v0;
                const Real c1 = v.dot(w);
                const Real c2 = v.dot(v);
                Real distance = -1;
                if(c1 <= 0)
                {
                    distance = (P - v0).norm();
                }
                else if(c2 <= c1)
                {
                    distance = (P - v1).norm();
                }
                else
                {
                    const Real b = c1/c2;
                    const Eigen::Vector3d Pb = v0 + b * v;
                    distance = (P - Pb).norm();
                }

                retval(i) = std::min(distance, retval(i));
            }
        }

        return retval;
    }

    MaterialProperties_Iso_Array getMaterialProperties() const
    {
        const Real E = parser.template parse<Real>("-E", 0.3);
        const Real nu = parser.template parse<Real>("-nu", 0.3);
        assert(nu > -1.0 and nu < 0.5);
        const Real thickness = parser.template parse<Real>("-h", 0.01);
        assert(thickness > 0);

        const int nFaces = mesh.getNumberOfFaces();

        Eigen::VectorXd thicknessVector = Eigen::VectorXd::Constant(nFaces, thickness);
        Eigen::VectorXd YoungVector = Eigen::VectorXd::Constant(nFaces, E);

        // check if we want a boundary layer
        const Real boundarylayer = parser.template parse<Real>("-boundarylayer", -1);
        if(boundarylayer > 0)
        {
            // compute distance vector from facecenters to boundary
            Eigen::MatrixXd restFaceCenters(nFaces, 3);
            for(int i=0;i<nFaces;++i)
            {
                restFaceCenters.row(i) = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i).computeFaceCenter();
            }
            const Eigen::VectorXd distanceVector = computeDistancesToBoundary(restFaceCenters);

            // compute the prefactor vector
            Eigen::VectorXd prefacVector(nFaces);
            for(int i=0;i<nFaces;++i)
            {
                const Real dist_to_boundary = distanceVector(i);
                prefacVector(i) = std::min( 1.0, std::pow(dist_to_boundary / boundarylayer, 2) ); // between 0 and 1
            }

            // check if we want a boundary layer on thickness or on E : default is thickness
            const Real minE = parser.template parse<Real>("-minE", -1);
            if(minE > 0)
            {
                YoungVector = Eigen::VectorXd::Constant(nFaces, minE) + prefacVector*(E - minE);
            }
            else
            {
                const Real minThickness = parser.template parse<Real>("-minh", thickness/100);
                thicknessVector = Eigen::VectorXd::Constant(nFaces, minThickness) + prefacVector*(thickness - minThickness);
            }
        }

        MaterialProperties_Iso_Array retval(YoungVector, nu, thicknessVector);

        {
            // dump
            const auto cvertices = mesh.getRestConfiguration().getVertices();
            const auto cface2vertices = mesh.getTopology().getFace2Vertices();
            WriteVTK writer(cvertices, cface2vertices);
            writer.addScalarFieldToFaces(YoungVector, "E");
            writer.addScalarFieldToFaces(thicknessVector, "h");
            writer.write(tag+"_matprop");
        }

        return retval;
    }


public:

    Sim_Inverse(ArgumentParser & parser):
    Sim<tMesh>(parser)
    {}
};

#endif /* Sim_Inverse_hpp */
