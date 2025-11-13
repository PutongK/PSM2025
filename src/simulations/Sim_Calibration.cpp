//
//  Sim_Calibration.cpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#include "Sim_Calibration.hpp"
#include "Geometry.hpp"
#include "MaterialProperties.hpp"
#include "GrowthHelper.hpp"

#include "NonEuclideanConformalMapping.hpp"
#include "ConformalMappingBoundary.hpp"
#include "EnergyOperatorList.hpp"

#include "QuadraticFormOperator.hpp"

#include "MeshQualityOperator.hpp"
#include "AreaDistortionOperator.hpp"

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#include <igl/boundary_loop.h>

void Sim_Calibration::init()
{

}

void Sim_Calibration::run()
{
    const Real initialPerturbationFac = parser.parse<Real>("-perturbation", 0.0);

    const std::string runCase = parser.parse<std::string>("-case", "");
    if(runCase == "almen_optim")
      Almen_Optim(initialPerturbationFac);
    else if(runCase == "wave_optim")
      Wave_Optim(initialPerturbationFac);
}



// adjust growthfactors in an Almen strip sized coupon to fit its deflection to the one measured with the Almen gage
void Sim_Calibration::Almen_Optim(const Real deformRad)
{
    tag = "Almen_optim_param";

    const Real res = parser.parse<Real>("-res", 0.02); // res = 1/(quantity of nodes per boundary)
    const Real Lx = parser.parse<Real>("-lx", 0.03769125); // almen strip dimensions
    const Real Ly = parser.parse<Real>("-ly", 0.00991875); // almen strip dimensions
    const Real relArea = 2.0*Lx*res;

    const Real true_deflec = parser.parse<Real>("-true_deflec", 0.0001545); //the experimentally measured deflection with the Almen gage
    const Real toler = parser.parse<Real>("-toler", 2e-6); // stop criterion 1: difference between the simulated and the real deflections
    const int maxiter = parser.parse<int>("-maxiter", 10); // stop criterion 2: maximal number of iterations

    RectangularPlate_4clampvert geometry(Lx, Ly, relArea, false, false); // fix vertical displacement at four corners to measure accurately the deflection
    mesh.init(geometry);

    const Real h_total = parser.parse<Real>("-h_total", 0.00206); // total thickness
    // formulate growth as a step eigenstrain profile
    Real heq = parser.parse<Real>("-heq", 0.0011);
    Real epseq = parser.parse<Real>("-epseq", 0.00182);

    Real growthRate_b = 0.0;
    Real growthRate_t = 0.0;
    Trilayer_2_Bilayer(epseq, heq, h_total, &growthRate_t, &growthRate_b);

    Eigen::VectorXd deflec(maxiter);
    deflec.setZero();

    const auto Vertices = mesh.getCurrentConfiguration().getVertices();
    const int nVert = mesh.getNumberOfVertices();
    const auto Connect = mesh.getTopology().getFace2Vertices();
    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXd growthRates_b(nFaces);
    Eigen::VectorXd growthRates_t(nFaces);
    growthRates_b.setZero();
    growthRates_t.setZero();

    // coordinates of one of the supports of the Almen gage and of it probe, which is at the origin in the flat state and lifted along z in the deformed state
    Eigen::VectorXd support_coord(3);
    support_coord(0) = 0.01587;
    support_coord(1) = 0.00794;
    Eigen::VectorXd origin_coord(3);
    origin_coord.setZero();

    // assign growthfactors
    growthRates_b = Eigen::VectorXd::Constant(nFaces, growthRate_b);
    growthRates_t = Eigen::VectorXd::Constant(nFaces, growthRate_t);

    // write initial condition
    mesh.writeToFile(tag+"_init");

    // dump
    dumpIso(growthRates_b, growthRates_t, tag+"_init");

    // define the material and operator
    const Real E = 73500;
    const Real nu = 0.33;
    MaterialProperties_Iso_Constant matprop_bot(E, nu, h_total);
    MaterialProperties_Iso_Constant matprop_top(E, nu, h_total);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // dump 0 swelling rate (initial condition) for nicer movies afterwards
    dumpIso(growthRates_b, growthRates_t, tag+"_final_"+helpers::ToString(0,2));

    int s=0;
    Real deflec_ratio;
    Real deflec_diff = 10.0;

    // iterative adjustment of the growthfactors based on the deflection measured with the Almen gage
    while (deflec_diff > toler && s < maxiter){

      mesh.resetToRestState();

      // adjust growthfactors
      if (s > 0){
        deflec_ratio = true_deflec/deflec(s-1);
        growthRate_t *= deflec_ratio;
        growthRate_b *= deflec_ratio;
        growthRates_b = Eigen::VectorXd::Constant(nFaces, growthRate_b);
        growthRates_t = Eigen::VectorXd::Constant(nFaces, growthRate_t);
      }

      const std::string curTag = tag+"_final_"+helpers::ToString(s+1,2); // s+1 since we already dumped s=0 as the initial condition

      // apply swelling
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());
      // minimize energy
      Real eps = 1e-2;
      minimizeEnergy(engOps, eps);

      // dump
      dumpIso(growthRates_b, growthRates_t, curTag);
      mesh.writeToFile(curTag);
      
      
      ////////////////////
      //Measure deflection
      // find the triangular element that contains the support
      Real mindist = 1000.0;
      Eigen::VectorXd dist(3);
      int support_triangle;
      for (int i=0; i<nFaces; ++i){
        dist.setZero();
        for (int j=0; j<3; ++j){
          dist(j) = std::pow((Vertices(Connect(i,j),0) - support_coord(0)),2) + std::pow((Vertices(Connect(i,j),1) - support_coord(1)),2);
        }
        if (dist.sum() < mindist){
          mindist  = dist.sum();
          support_triangle = i;
        }
      }
      // find the triangular element that contains the origin
      mindist = 1000.0;
      int origin_triangle;
      for (int i=0; i<nFaces; ++i){
        dist.setZero();
        for (int j=0; j<3; ++j){
          dist(j) = std::pow(Vertices(Connect(i,j),0),2) + std::pow(Vertices(Connect(i,j),1),2);
        }
        if (dist.sum() < mindist){
          mindist  = dist.sum();
          origin_triangle = i;
        }
      }

      //we draw a plane through the 3 vertices of the triangle that contains the support
      //next we calculate its z coordinate as a function of its x and y
      Real x1 = Vertices(Connect(support_triangle,0),0);
      Real y1 = Vertices(Connect(support_triangle,0),1);
      Real z1 = Vertices(Connect(support_triangle,0),2);

      Real x2 = Vertices(Connect(support_triangle,1),0);
      Real y2 = Vertices(Connect(support_triangle,1),1);
      Real z2 = Vertices(Connect(support_triangle,1),2);

      Real x3 = Vertices(Connect(support_triangle,2),0);
      Real y3 = Vertices(Connect(support_triangle,2),1);
      Real z3 = Vertices(Connect(support_triangle,2),2);

      Real x = support_coord(0);
      Real y = support_coord(1);

      Real z = (((x-x1)*(y2-y1)*(z3-z1) + (y-y1)*(z2-z1)*(x3-x1) - (y-y1)*(x2-x1)*(z3-z1) - (x-x1)*(z2-z1)*(y3-y1)) / ((y2-y1)*(x3-x1) - (x2-x1)*(y3-y1))) + z1;
      support_coord(2) = z;

      std::cout << "\n" << "SUPPORT_COORD " << z << std::endl;



      //we draw a plane through the 3 vertices of the triangle that contains the origin
      //next we calculate its z coordinate as a function of its x and y
      x1 = Vertices(Connect(origin_triangle,0),0);
      y1 = Vertices(Connect(origin_triangle,0),1);
      z1 = Vertices(Connect(origin_triangle,0),2);

      x2 = Vertices(Connect(origin_triangle,1),0);
      y2 = Vertices(Connect(origin_triangle,1),1);
      z2 = Vertices(Connect(origin_triangle,1),2);

      x3 = Vertices(Connect(origin_triangle,2),0);
      y3 = Vertices(Connect(origin_triangle,2),1);
      z3 = Vertices(Connect(origin_triangle,2),2);

      x = 0.0;
      y = 0.0;

      z = (((x-x1)*(y2-y1)*(z3-z1) + (y-y1)*(z2-z1)*(x3-x1) - (y-y1)*(x2-x1)*(z3-z1) - (x-x1)*(z2-z1)*(y3-y1)) / ((y2-y1)*(x3-x1) - (x2-x1)*(y3-y1))) + z1;
      origin_coord(2) = z;

      std::cout << "\n" << "ORIGIN_COORD " << z << std::endl;



      deflec(s) = origin_coord(2) - support_coord(2);
      std::cout << "\n" << "THE_DEFLECTION " << "\n" << deflec << std::endl;
      deflec_diff = std::abs(deflec(s) - true_deflec);

      s++;
    }


    //Measure curvature 
    ////////////////////////////////////////
        Eigen::VectorXd PrincCurv1(nFaces);
        Eigen::VectorXd PrincCurv2(nFaces);
        Eigen::VectorXd mean_t(nFaces);
        Eigen::VectorXd gauss_t(nFaces);
        ComputeCurvatures<tMesh> computeCurvatures;
        computeCurvatures.computeNew(mesh, gauss_t, mean_t, PrincCurv1, PrincCurv2);
    ///////////////////////////////////////

    // dump
    dumpIso(growthRates_b, growthRates_t, tag+"_final");
    // write
    mesh.writeToFile(tag+"_final");

    Bilayer_2_Trilayer(&epseq, &heq, h_total, growthRate_t, growthRate_b);

    std::cout << "\n" << "GROWTH_BOT" << "\n" << growthRate_b << std::endl;
    std::cout << "\n" << "GROWTH_TOP" << "\n" << growthRate_t << std::endl;
    std::cout << "\n" << "H_EQ" << "\n" << heq << std::endl;
    std::cout << "\n" << "EPS_EQ" << "\n" << epseq << std::endl;
}




// Optimize the anisotropy coefficient for during the "wave" experiment
void Sim_Calibration::Wave_Optim(const Real deformRad)
{
    tag = "Wave_Optim";
    
    // the 3D scanned mesh is given here under the -filename
    // "-geometry external" option must be chosen
    // the principal curvature of the wave has to be aligned with x-axis 
    // choose -initrectangle = true if the initial shape is not provided and we want to generate it
    initInverseProblem();

    const Real E = 1;
    const Real nu = 0.33;
    const Real h_total = parser.parse<Real>("-h_total", 0.00206);

    const Real growthRate_t = parser.parse<Real>("-growth_top", 0.00119); //the growthfactors measured with the Almen_Optim method
    const Real growthRate_b = parser.parse<Real>("-growth_bot", -(1.0)*growthRate_t);
    const Real centersize = parser.parse<Real>("-centersize", 0.8); // we measure curvatures in a square of the size centersize*l, where l is the length of the plate 
    const Real toler = parser.parse<Real>("-toler", 0.01); // stop criterion 1: |ratio_curv-1|<toler, so that ratio_curv=1 means perfect consistency of the simulated and of the experimental shapes 
    const int maxiter = parser.parse<int>("-maxiter", 10); // stop criterion 2: maximal number of iterations

    auto Vertices = mesh.getCurrentConfiguration().getVertices();
    auto RestVertices = mesh.getRestConfiguration().getVertices();
    const int nVert = mesh.getNumberOfVertices();
    const auto Connect = mesh.getTopology().getFace2Vertices();
    const int nFaces = mesh.getNumberOfFaces();

    //measure the curvature and stretching of the experimental shape
    tVecMat2d firstFF,secondFF;
    computeQuadraticForms(firstFF, secondFF);
    Eigen::VectorXd area_t(nFaces);
    for(int i=0;i<nFaces;++i)
    {
        const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
        area_t(i) = 0.5*std::sqrt(firstFF[i].determinant());
    }
    ////////////////////////////////////////
    Eigen::VectorXd gauss_t(nFaces);
    Eigen::VectorXd mean_t(nFaces);
    Eigen::VectorXd princcurv1_t(nFaces);
    Eigen::VectorXd princcurv2_t(nFaces);
    Eigen::VectorXd CurvX_t(nFaces);
    Eigen::VectorXd CurvY_t(nFaces);
    Eigen::Vector3d DirX = (Eigen::Vector3d() <<  1, 0, 0).finished();
    Eigen::Vector3d DirY = (Eigen::Vector3d() <<  0, 1, 0).finished();
    ComputeCurvatures<tMesh> computeCurvatures;
    computeCurvatures.computeDir(mesh, gauss_t, mean_t, princcurv1_t, princcurv2_t, CurvX_t, CurvY_t, DirX, DirY);
    ///////////////////////////////////////

    const Real lx = (RestVertices.col(0).maxCoeff() - RestVertices.col(0).minCoeff())/2.0;
    const Real ly = (RestVertices.col(1).maxCoeff() - RestVertices.col(1).minCoeff())/2.0;


    // Define the central square and measure the mean curvature in the x direction inside this square
    Real CenterX=0.0;
    Real CenterY=0.0;
    for (int i=0; i<nVert; ++i){
        CenterX += Vertices(i,0);
        CenterY += Vertices(i,1);
    }
    CenterX /= nVert;
    CenterY /= nVert;

    Eigen::VectorXi IndicV_curv(nVert);
    IndicV_curv.setZero();
    Eigen::VectorXd IndicV_curv1(nVert); //for dumping
    IndicV_curv.setZero();

    for (int i=0; i<nVert; ++i){
      if (std::abs(Vertices(i,0) - CenterX) <= centersize*lx && std::abs(Vertices(i,1)-CenterY) < centersize*ly){
        IndicV_curv(i) = 1;
        IndicV_curv1(i) = 1;
      }
    }

    Real CurvXt = 0.0;
    int count = 0;
    for (int i=0; i<nFaces; ++i){
      if (IndicV_curv(Connect(i,0))==1 && IndicV_curv(Connect(i,1))==1 && IndicV_curv(Connect(i,2))==1) {
        CurvXt += std::abs(CurvX_t(i));
        //CurvXt += std::abs(CurvX_t(i)/area_t(i));
        count ++;
      } 
    }
    CurvXt /= count;

    WriteVTK writer(RestVertices, Connect);
    writer.addScalarFieldToVertices(IndicV_curv1, "IndicV");
    writer.write(tag+"_MeasureCurv");


    // Draw the wave pattern
    Eigen::VectorXi IndicV(nVert);
    for (int i=0; i<nVert; ++i){
      if (RestVertices(i,0)<= 1e-4){
        IndicV(i) = 1;
      } else{
        IndicV(i) = 0;
      }
    }
    Eigen::VectorXd growthRates_t(nFaces);
    Eigen::VectorXd growthRates_b(nFaces);
    for (int i=0; i<nFaces; ++i){
      if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
        growthRates_t(i) = growthRate_t;
        growthRates_b(i) = growthRate_b;
      }
      else{
        growthRates_t(i) = growthRate_b;
        growthRates_b(i) = growthRate_t;
      }
    }
    const Eigen::VectorXd growthAngles = Eigen::VectorXd::Constant(nFaces, 0.0);
    Eigen::VectorXd growth_1_t(nFaces);
    Eigen::VectorXd growth_1_b(nFaces);
    Eigen::VectorXd growth_2_t(nFaces);
    Eigen::VectorXd growth_2_b(nFaces);
    for (int i=0; i<nFaces; i++){
      growth_1_t(i) = growthRates_t(i);
      growth_1_b(i) = growthRates_b(i);
      growth_2_t(i) = growthRates_t(i);
      growth_2_b(i) = growthRates_b(i);
    }
    MaterialProperties_Iso_Constant matprop_bot(E, nu, h_total);
    MaterialProperties_Iso_Constant matprop_top(E, nu, h_total);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});




    int s=0;
    Real ortho_coeff = 1.0;
    Real curvX_ratio = 0.0;

    // iteratively adjust the orthotropi coefficient
    while (std::abs(curvX_ratio-1.0) > toler && s < maxiter){

      mesh.resetToRestState();

      if (s > 0){
        ortho_coeff *= curvX_ratio;

        for (int i=0; i<nFaces; i++){
          growth_1_t(i) = growthRates_t(i)*ortho_coeff;
          growth_1_b(i) = growthRates_b(i)*ortho_coeff;
          growth_2_t(i) = growthRates_t(i)*(2.0-ortho_coeff);
          growth_2_b(i) = growthRates_b(i)*(2.0-ortho_coeff);
        }
      } else {
          dumpOrtho(growth_1_b, growth_2_b, growth_1_t, growth_2_t, growthAngles, tag+"_init");
      }

      // apply swelling
      GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growth_1_b, growth_2_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growth_1_t, growth_2_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

      // minimize energy
      Real eps = 1e-2;
      minimizeEnergy(engOps, eps);

      // dump
      const std::string curTag = tag+"_final_"+helpers::ToString(s+1,2); // s+1 since we already dumped s=0 as the initial condition
      dumpOrtho(growth_1_b, growth_2_b, growth_1_t, growth_2_t, growthAngles, curTag);
      mesh.writeToFile(curTag);

      // measure curvature and stretching at the center of the simulated shape
      computeQuadraticForms(firstFF, secondFF);
      Eigen::VectorXd area_c(nFaces);
      for(int i=0;i<nFaces;++i)
      {
          const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
          area_c(i) = 0.5*std::sqrt(firstFF[i].determinant());
      }
      ////////////////////////////////////////
      Eigen::VectorXd gauss_c(nFaces);
      Eigen::VectorXd mean_c(nFaces);
      Eigen::VectorXd princcurv1_c(nFaces);
      Eigen::VectorXd princcurv2_c(nFaces);
      Eigen::VectorXd CurvX_c(nFaces);
      Eigen::VectorXd CurvY_c(nFaces);
      Eigen::Vector3d DirX = (Eigen::Vector3d() <<  1, 0, 0).finished();
      Eigen::Vector3d DirY = (Eigen::Vector3d() <<  0, 1, 0).finished();
      ComputeCurvatures<tMesh> computeCurvatures;
      computeCurvatures.computeDir(mesh, gauss_c, mean_c, princcurv1_c, princcurv2_c, CurvX_c, CurvY_c, DirX, DirY);
      ///////////////////////////////////////
      count = 0;
      Real CurvXc = 0.0;
      for (int i=0; i<nFaces; ++i){
        if (IndicV_curv(Connect(i,0))==1 && IndicV_curv(Connect(i,1))==1 && IndicV_curv(Connect(i,2))==1) {
          CurvXc += std::abs(CurvX_c(i));
          //CurvXc += std::abs(CurvX_c(i)/area_c(i));
          count++;
        } 
      }
      CurvXc /= count;

      curvX_ratio = CurvXt / CurvXc;

      std::cout << "\n\n" << "ortho_coeff " << "\n" << (ortho_coeff-1.0) << std::endl;
      std::cout << "\n\n" << "curvX_ratio " << "\n" << curvX_ratio << std::endl;

      s++;
    }
}




void Sim_Calibration::dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig, bool curvinvert)
{
    const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();
    const int nFaces = mesh.getNumberOfFaces();
    WriteVTK writer(cvertices, cface2vertices);

    const TopologyData & topology = mesh.getTopology();
    const tReferenceConfigData & restState = mesh.getRestConfiguration();
    const tCurrentConfigData & currentState = mesh.getCurrentConfiguration();
    const BoundaryConditionsData & boundaryConditions = mesh.getBoundaryConditions();

    Eigen::MatrixXd normal_vectors(nFaces,3);
    if(restConfig)
        restState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);
    else
        currentState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);

    writer.addVectorFieldToFaces(normal_vectors, "normals");

    if (restConfig == false)
    {
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
        if (curvinvert){
          mean *=(-1);
          PrincCurv1 *=(-1);
          PrincCurv2 *=(-1);
          CurvX *=(-1);
          CurvY *=(-1);
        }

        writer.addScalarFieldToFaces(gauss, "gauss");
        writer.addScalarFieldToFaces(mean, "mean");
        writer.addScalarFieldToFaces(PrincCurv1, "PrincCurv1");
        writer.addScalarFieldToFaces(PrincCurv2, "PrincCurv2");
        writer.addScalarFieldToFaces(CurvX, "CurvX");
        writer.addScalarFieldToFaces(CurvY, "CurvY");
    }

    writer.addScalarFieldToFaces(growthRates_1_bot, "rate1_bot");
    writer.addScalarFieldToFaces(growthRates_2_bot, "rate2_bot");
    writer.addScalarFieldToFaces(growthAngles, "dir_growth");
    writer.addScalarFieldToFaces(growthRates_1_top, "rate1_top");
    writer.addScalarFieldToFaces(growthRates_2_top, "rate2_top");

    writer.write(filename);
}




void Sim_Calibration::dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig)
{
    const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();;
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();

    WriteVTK writer(cvertices, cface2vertices);
    if(growthRates_bot.rows() == mesh.getNumberOfFaces())
    {
        writer.addScalarFieldToFaces(growthRates_bot, "growthrates_bot");
        writer.addScalarFieldToFaces(growthRates_top, "growthrates_top");
    }
    else if(growthRates_bot.rows() == mesh.getNumberOfVertices())
    {
        writer.addScalarFieldToVertices(growthRates_bot, "growthrates_bot");
        writer.addScalarFieldToVertices(growthRates_top, "growthrates_top");
    }
    else
    {
        const std::string errmsg = ("Problem  : number of rows in growthRates = "+std::to_string(growthRates_bot.rows())+" while nVertices / nFaces = "+std::to_string(mesh.getNumberOfVertices())+" , "+std::to_string(mesh.getNumberOfFaces()));
        helpers::catastrophe(errmsg, __FILE__, __LINE__, false);
    }

    addCurvaturesToWriter(writer);

    writer.write(filename);
}





void Sim_Calibration::Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot)
{
  *theta_top = heq*epseq*(3*h-2*heq)/std::pow(h,2);
  *theta_bot = heq*epseq*(2*heq-h)/std::pow(h,2);
}

void Sim_Calibration::Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot)
{
  *heq=(theta_top+3.0*theta_bot)*h/((theta_top+theta_bot)*2.0);
  *epseq=std::pow((theta_top+theta_bot),2)/(theta_top+3.0*theta_bot);
}

