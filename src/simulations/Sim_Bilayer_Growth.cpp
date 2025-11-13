//
//  Sim_Bilayer_Growth.cpp
//  Elasticity
//
//  Created by Wim van Rees on 10/27/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#include "Sim_Bilayer_Growth.hpp"
#include "Geometry.hpp"
#include "GrowthHelper.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include "ComputeCurvatures.hpp"

void Sim_Bilayer_Growth::run()
{
  const std::string runCase = parser.parse<std::string>("-case", "");
  if(runCase == "custom")
    TestCustomGrowth(); // custom growth pattern
  else if(runCase == "testR")
    TestRandomPatterns(); // random patterns on a rectangular plate
  else
    {
      std::cout << "No valid runCase defined. Options are \n";
      std::cout << "\t -case custom\n";
      std::cout << "\t -case testR\n";
    }
}




void Sim_Bilayer_Growth::TestCustomGrowth()
{
    const std::string growth_type = parser.parse<std::string>("-growth_type","chess");
    // Options: 
    // - homo (uniform expansion)
    // - chess (the checkerboard pattern)
    // - center (rectangular zone in the center of the plate)
    // - wave (half of the plate is expanding on the bottom side, and half - on the top side)
    // - circle (circular zone in the center of the plate)
    // - external (projection of a pattern coming from another mesh)
    const std::string geometryCase = parser.parse<std::string>("-geometry", ""); //see initForwardProblem()
    const Real margin_x = parser.parse<Real>("-margin_x", 0.0); // margins to simulate the clamping frame: no eigensrain in this zone. 0.001 = 1mm
    const Real margin_y = parser.parse<Real>("-margin_y", 0.0);
    tag = "bilayer_" + growth_type;

    initForwardProblem();

    const Real E = 1;
    const Real nu = 0.33;
    const Real h_total = parser.parse<Real>("-h_total", 0.003);

    // init growth
    Real growthRate_t = parser.parse<Real>("-growth_top", 0.001);
    Real growthRate_b = parser.parse<Real>("-growth_bot", -0.001);

    const Real growthAngle = parser.parse<Real>("-growth_angle", 0.0)*M_PI; // principal growth direction (angle with respect to x-axis)
    const Real ortho_coeff = parser.parse<Real>("-ortho_coeff", 0.0); // orthotropy coefficient

    auto Vertices = mesh.getCurrentConfiguration().getVertices();
    const int nVert = mesh.getNumberOfVertices();
    const auto Connect = mesh.getTopology().getFace2Vertices();
    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXi IndicV(nVert);
    IndicV.setZero();
    Eigen::VectorXd growthRates_b(nFaces);
    Eigen::VectorXd growthRates_t(nFaces);
    growthRates_b.setZero();
    growthRates_t.setZero();

    if (growth_type == "homo"){
      growthRates_b = Eigen::VectorXd::Constant(nFaces, growthRate_b);
      growthRates_t = Eigen::VectorXd::Constant(nFaces, growthRate_t);
    }
    else if (growth_type == "chess"){
      Real CenterX;
      Real CenterY;

      for (int i=0; i<nVert; ++i){
        if ((std::abs(Vertices(i,0))<10e-9 && std::abs(Vertices(i,1))<10e-9)){
            CenterX = Vertices(i,0);
            CenterY = Vertices(i,1);
        }
      }

      //first quarter

      for (int i=0; i<nVert; ++i){
        if (Vertices(i,0) > CenterX && Vertices(i,1) > CenterY){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
  	     else{
           // growthRates_b(i) = 0.0;
           // growthRates_t(i) = 0.0;
           growthRates_b(i) = growthRate_t;
           growthRates_t(i) = growthRate_b;
  	     }
      }

      //second quarter

      IndicV.setZero();

      for (int i=0; i<nVert; ++i){
        //IndicV(i) = 0;
        if (Vertices(i,0) < CenterX && Vertices(i,1) < CenterY){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
      }
    }

    else if (growth_type == "center"){
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);

      const Real size = parser.parse<Real>("-size", 0.4);
      const Real size_hole = parser.parse<Real>("-size_hole", 0.0);

      for (int i=0; i<nVert; ++i){
        if (abs(Vertices(i,0))<=Lx*size && abs(Vertices(i,1))<=Ly*size){
          IndicV(i) = 1;
        } else{
          IndicV(i) = 0;
        }
        if (abs(Vertices(i,0))<=Lx*size_hole && abs(Vertices(i,1))<=Ly*size_hole){
          IndicV(i) = 0;
        }
      }

      for (int i=0; i<nFaces; ++i){
        if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
          //growthRates_b(i) = growthRate_b;
          //growthRates_t(i) = growthRate_t;
          growthRates_t(i) = 0.0;
          growthRates_b(i) = 0.0;
        }
        else{
          //growthRates_t(i) = 0.0;
          //growthRates_b(i) = 0.0;
          growthRates_b(i) = growthRate_b;
          growthRates_t(i) = growthRate_t;
        }
      }
    }
    else if (growth_type == "wave"){

      const bool sym_x = parser.parse<bool>("-sym_x", false);
      const bool inv_wave = parser.parse<bool>("-inv_wave", false);
      const Real overlap = parser.parse<Real>("-overlap", 0.0);

      Real CenterX=0.0;
      Real CenterY=0.0;
      for (int i=0; i<nVert; ++i){
          CenterX += Vertices(i,0);
          CenterY += Vertices(i,1);
      }
      CenterX /= nVert;
      CenterY /= nVert;

      if (sym_x){
        for (int i=0; i<nVert; ++i){
          if (Vertices(i,1)<= CenterY + 1e-9){
            IndicV(i) = 1;
          } else{
            IndicV(i) = 0;
          }

          if ((Vertices(i,1)<= CenterY + overlap) && (Vertices(i,1)>= CenterY - overlap)){
            IndicV(i) = 2;
          }

        }
      } else {
        for (int i=0; i<nVert; ++i){
          if (Vertices(i,0)<= CenterX + 1e-9){
            IndicV(i) = 1;
          } else{
            IndicV(i) = 0;
          }

          if ((Vertices(i,0)<= CenterX + overlap) && (Vertices(i,0)>= CenterX - overlap)){
            IndicV(i) = 2;
          }
        }
      }

      for (int i=0; i<nFaces; ++i){
        if (inv_wave){
          if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
          else{
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
        } else{
          if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
          else{
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
        }

        if (IndicV(Connect(i,0))==2 || IndicV(Connect(i,1))==2 || IndicV(Connect(i,2))==2) {
            growthRates_b(i) = growthRate_b + growthRate_t;
            growthRates_t(i) = growthRate_t + growthRate_b;
        }
      }
    }

    else if (growth_type == "circle"){
      const Real radius = parser.parse<Real>("-radius", 0.2);

      for (int i=0; i<nVert; ++i){
        if (std::pow(Vertices(i,0),2) + std::pow(Vertices(i,1),2) <= std::pow(radius,2)){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
  	     else{
           growthRates_b(i) = 0.0;
           growthRates_t(i) = 0.0;
  	     }
      }
    }

    else if (growth_type == "external"){

      const bool samemesh = parser.parse<bool>("-samemesh", false); 
      // whether the external pattern is generated on the same mesh that we use or not
      // if true, then we just activate the elements specified in the text files
      // if false, then we project the pattern specified on another mesh onto our initial mesh

      if (samemesh){
        const std::string growth_tag = parser.template parse<std::string>("-growth_tag", "");
        const Real correction_coeff = parser.parse<Real>("-correction_coeff", 1.0);
        const int size1 = parser.parse<int>("-size1", nFaces);
        const int size2 = parser.parse<int>("-size2", 1);

        // two .txt files that specify "active" elements on the top and bottom layers
        // each file contains a vector of 0 or 1, where 1 corresponds to the active elements
        Eigen::VectorXi active_b(nFaces);
        Eigen::VectorXi active_t(nFaces);
        helpers::read_matrix(growth_tag + "_peening_bot.txt", active_b, size1, size2);
        helpers::read_matrix(growth_tag + "_peening_top.txt", active_t, size1, size2);

        // std::cout<<active_b<<std::endl;

        for (int i=0; i<nFaces; i++){
          if (active_b(i) == 1){
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
          if (active_t(i) == 1){
            growthRates_t(i) += growthRate_t;
            growthRates_b(i) += growthRate_b;
          }
        }

        Real CenterX = 0.0;
        Real CenterY = 0.0;
        Real CenterZ = 0.0;

        for (int i=0; i<nVert; ++i){
          CenterX += Vertices(i,0);
          CenterY += Vertices(i,1);
          CenterZ += Vertices(i,2);
        } 

        CenterX /= nVert;
        CenterY /= nVert;
        CenterZ /= nVert;

        for (int i=0; i<nVert; i++){
          Vertices(i,0) -= CenterX;
          Vertices(i,1) -= CenterY;
          Vertices(i,2) -= CenterZ;

          Vertices(i,0) *= correction_coeff;
          Vertices(i,1) *= correction_coeff;
        }

        mesh.getCurrentConfiguration().getVertices() = Vertices;
        mesh.getRestConfiguration().getVertices() = Vertices;
      }


      else {
        const std::string growth_tag = parser.template parse<std::string>("-growth_tag", "");

        // specify the number of faces and vertices in the mesh containing the pattern
        const int nFaces_Reg = parser.template parse<int>("-nFaces_Reg", 2178);
        const int nVert_Reg = parser.template parse<int>("-nVert_Reg", 1156);

        Eigen::MatrixXi Reg_Vertices(nVert_Reg,3);
        Eigen::MatrixXi Reg_Faces(nFaces_Reg,3);
        Eigen::VectorXi Peening_bot(nFaces_Reg);
        Eigen::VectorXi Peening_top(nFaces_Reg);
        Eigen::VectorXi Reg_Clusters(nFaces_Reg);
        Reg_Clusters.setZero();

        // two .txt files that specify "active" elements on the top and bottom layers,
        // each file contains a vector of 0 or 1, where 1 corresponds to the active elements
        helpers::read_matrix(growth_tag + "_peening_bot.txt", Peening_bot, nFaces_Reg, 1);
        helpers::read_matrix(growth_tag + "_peening_top.txt", Peening_top, nFaces_Reg, 1);
        // two .txt files that define the mesh containing the pattern
        helpers::read_matrix(growth_tag + "_vertices.txt", Reg_Vertices, nVert_Reg, 3);
        helpers::read_matrix(growth_tag + "_faces.txt", Reg_Faces, nFaces_Reg, 3);

        //scale the pattern before projection
        const Real mesh_ratio_x = (Vertices.col(0).maxCoeff() - Vertices.col(0).minCoeff()) / (Reg_Vertices.col(0).maxCoeff() - Reg_Vertices.col(0).minCoeff());
        const Real mesh_ratio_y = (Vertices.col(1).maxCoeff() - Vertices.col(1).minCoeff()) / (Reg_Vertices.col(1).maxCoeff() - Reg_Vertices.col(1).minCoeff());
        for (int i=0; i<nVert_Reg; i++){
          Reg_Vertices(i,0) *= mesh_ratio_x;
          Reg_Vertices(i,1) *= mesh_ratio_y;
        }

        const Real CenterX = Reg_Vertices.col(0).mean();
        const Real CenterY = Reg_Vertices.col(1).mean();
        for (int i=0; i<nVert_Reg; i++){
          Reg_Vertices(i,0) -= CenterX;
          Reg_Vertices(i,1) -= CenterY;
          Reg_Vertices(i,2) = 0;
        }

        for (int i=0; i<nFaces_Reg; i++){
          if(Peening_bot(i)==1){
            Reg_Clusters(i) = 2;
          }
          if(Peening_top(i)==1){
            Reg_Clusters(i) = 1;
          }
        }

        // we draw a circle around each vertex and look if the vertices in the regular pattern
        // that lie inside this circle are all "activated". This is a simplified formulation that works bad if the pattern is complex

        // So the sizes and positions of the two plates must perfectly coincide!

        //First we assign the IndicV to the initial pattern
        Eigen::VectorXi IndicV_Reg(nVert_Reg);
        IndicV_Reg.setZero();
        for (int i=0; i<nFaces_Reg; i++){
          if(Reg_Clusters(i)==1 || Reg_Clusters(i)==2){
            for (int j=0; j<3; j++){
              IndicV_Reg(Reg_Faces(i,j)) = Reg_Clusters(i);
            }
          }
        }

        //we divide the -lx by number of elms along this side and multiply by sqrt(2)/2 (if it falls exactly in center of square)
        const Real search_rad = parser.template parse<Real>("-search_rad", ((Reg_Vertices.col(0).maxCoeff() - Reg_Vertices.col(0).minCoeff())/std::sqrt(nFaces_Reg*0.5)) * std::sqrt(2.0)*0.5 * 1.1);
        for (int i=0; i<nVert; i++){
          Eigen::VectorXi Inside_Circle(nVert_Reg);
          Inside_Circle.setZero();
          int olol=0;
          for (int j=0; j<nVert_Reg; j++){
            if (std::pow(Reg_Vertices(j,0) - Vertices(i,0),2) + std::pow(Reg_Vertices(j,1) - Vertices(i,1),2) < std::pow(search_rad*1.01,2)){
              Inside_Circle(olol) = IndicV_Reg(j);
              olol ++;
            }
          }

          int not_clust_1 = 0;
          int not_clust_2 = 0;
          for (int j=0; j<olol; j++){
            if(Inside_Circle(j) != 1){
              not_clust_1 = 1;
            }
            if(Inside_Circle(j) != 2){
              not_clust_2 = 1;
            }
          }

          if (not_clust_1 == 0){
            IndicV(i) = 1;
          } else if (not_clust_2 == 0){
            IndicV(i) = 2;
          }

        }

        for (int i=0; i<nFaces; ++i){
          if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
          else if (IndicV(Connect(i,0))==2 || IndicV(Connect(i,1))==2 || IndicV(Connect(i,2))==2) {
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
        }
      }
    }

    

    if (margin_x > 0.0 || margin_y > 0.0) MarginCut(margin_x, margin_y, growthRates_b, growthRates_t);

    const Eigen::VectorXd growthAngles = Eigen::VectorXd::Constant(nFaces, growthAngle);

    Eigen::VectorXd growthRates_1_t(nFaces);
    Eigen::VectorXd growthRates_1_b(nFaces);
    Eigen::VectorXd growthRates_2_t(nFaces);
    Eigen::VectorXd growthRates_2_b(nFaces);

    for (int i=0; i<nFaces; i++){
      growthRates_1_t(i) = growthRates_t(i)*(1.0+ortho_coeff);
      growthRates_1_b(i) = growthRates_b(i)*(1.0+ortho_coeff);
      growthRates_2_t(i) = growthRates_t(i)*(1.0-ortho_coeff);
      growthRates_2_b(i) = growthRates_b(i)*(1.0-ortho_coeff);
    }


    GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_b, growthRates_2_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
    GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_t, growthRates_2_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

    // write initial condition
    mesh.writeToFile(tag+"_init");

    // dump
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_init");

    // define the material and operator
    MaterialProperties_Iso_Constant matprop_bot(E, nu, h_total);
    MaterialProperties_Iso_Constant matprop_top(E, nu, h_total);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // dump 0 swelling rate (initial condition) for nicer movies afterwards
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_final_"+helpers::ToString(0,2));


    const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
    const Real swelling_step = 1.0/((Real)nSwellingRuns);
    const int startidx = 0;

    std::string curTag;
    for(int s=startidx;s<nSwellingRuns;++s)
    {
        curTag = tag+"_final_"+helpers::ToString(s+1,2); // s+1 since we already dumped s=0 as the initial condition

        const Real swelling_fac = (s+1)*swelling_step;

        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_b, growthRates_2_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_t, growthRates_2_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

        // minimize energy
        Real eps = 1e-2;
        minimizeEnergy(engOps, eps);

        // dump
        // dumpIso(growthRates_b, growthRates_t, curTag);
        dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, curTag);
    }

    const std::string fname1 = curTag + ".vtp";
    IOGeometry geometry_dummy(fname1);
    mesh.init_rest(geometry_dummy);
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_final");
    
    dumpWithNormals(tag+"_final_curv");

}





// Generate n random patterns on a rectangular plate and solve the forward problem for each of them
void Sim_Bilayer_Growth::TestRandomPatterns()
{

    initForwardProblem();

    const int nFaces = mesh.getNumberOfFaces();

    Eigen::MatrixXi adj_faces(nFaces,3);
    const auto face2edges = mesh.getTopology().getFace2Edges();
    const auto edge2faces = mesh.getTopology().getEdge2Faces();

    Real adj_faceidx = 0.0;

    for(int i=0;i<nFaces;++i)
    {
      for(int e0=0;e0<3;++e0){
        for(int f0=0;f0<2;++f0)
        {
          adj_faceidx = edge2faces(face2edges(i, e0), f0);
          if (adj_faceidx!=i){
            adj_faces(i,e0) = adj_faceidx;
          }
        }
      }
    }

    const int nVert = mesh.getNumberOfVertices();
    const int nEdges = mesh.getNumberOfEdges();
    const auto vertex2faces = mesh.getTopology().getVertex2Faces();
    const auto edge2vertices = mesh.getTopology().getEdge2Vertices();

    Eigen::VectorXi vertex_edge(nVert);
    vertex_edge.setZero();

    for (int i=0;i<nVert;++i){
      for (int j=0;j<nEdges;++j){
        for (int k=0;k<2;++k){
          if(edge2vertices(j,k) == i){
            ++vertex_edge(i);
          }
        }
      }
    }

    int ntests = parser.parse<int>("-ntests", 3);

    Eigen::VectorXd h_ac (4);
    Eigen::VectorXd growth (4);

    Eigen::VectorXd growthRate1 (4);
    Eigen::VectorXd growthRate2 (4);

    growthRate1(0) = parser.parse<Real>("-growth1t", 0.00047);
    growthRate2(0) = parser.parse<Real>("-growth1b", -growthRate1(1));

    growthRate1(1) = parser.parse<Real>("-growth2t", 0.00082);
    growthRate2(1) = parser.parse<Real>("-growth2b", -growthRate1(2));

    growthRate1(2) = parser.parse<Real>("-growth3t", 0.0012);
    growthRate2(2) = parser.parse<Real>("-growth3b", -growthRate1(3));

    growthRate1(3) = parser.parse<Real>("-growth4t", 0.001475);
    growthRate2(3) = parser.parse<Real>("-growth4b", -growthRate1(4));

    Eigen::VectorXd maxerror(ntests);
    Eigen::VectorXd relative_error(ntests);
    Eigen::MatrixXd h_nu(ntests,2);

    Eigen::VectorXi ActiveElms_top(nFaces);
    ActiveElms_top.setZero();

    Eigen::VectorXi ActiveElms_bot(nFaces);
    ActiveElms_bot.setZero();

    const Real E = 1;

    Eigen::VectorXd growthRates_bot(nFaces);
    Eigen::VectorXd growthRates_top(nFaces);

    growthRates_bot.setZero();
    growthRates_top.setZero();

    Eigen::VectorXd growthRates_bot_eqv(nFaces);
    Eigen::VectorXd growthRates_top_eqv(nFaces);

    growthRates_bot_eqv.setZero();
    growthRates_top_eqv.setZero();

    Eigen::VectorXd h_bot(nFaces);
    Eigen::VectorXd h_top(nFaces);



    for(int k=0;k<ntests;++k){

      growthRates_bot.setZero();
      growthRates_top.setZero();

      growthRates_bot_eqv.setZero();
      growthRates_top_eqv.setZero();

      tag = "TestRandom_"+ helpers::ToString(k,2);

      Real h_total = std::rand() % 12 + 4;
      h_total = h_total/1000.0;
      std::cout << "\n" << "h_total" <<  " = " << h_total << "\n" << std::endl;

//      Real h_total = parser.parse<Real>("-h_total", 0.005);

      Real nu = std::rand() % 5 + 32;
      nu = nu/100.0;
      std::cout << "\n" << "nu" <<  " = " << nu << "\n" << std::endl;

//      Real nu = parser.parse<Real>("-nu", 0.33);

      h_nu(k,0)=h_total;
      h_nu(k,1)=nu;

      for (int i=0 ; i<nFaces ; ++i){
        h_bot(i) = h_total;
        h_top(i) = h_total;
      }

      Real CenterX;
      Real CenterY;

      int regime;

      auto Vertices = mesh.getCurrentConfiguration().getVertices();
      const auto Connect = mesh.getTopology().getFace2Vertices();

      helpers::write_matrix("VERTICES.txt", Vertices);
      helpers::write_matrix("CONNEC.txt", Connect);

      Eigen::VectorXi IndicV(nVert);

      Eigen::VectorXi Nregime(nFaces);
      Nregime.setZero();

      Eigen::VectorXi Active_top(nFaces);
      Active_top.setZero();

      Eigen::VectorXi Active_bot(nFaces);
      Active_bot.setZero();

      Eigen::VectorXd GrowthFacs_bot_temp(nFaces);
      GrowthFacs_bot_temp.setZero();

      Eigen::VectorXd GrowthFacs_top_temp(nFaces);
      GrowthFacs_top_temp.setZero();

      const int nspots_bot = std::rand() % 6 + 1;
      const int nspots_top = std::rand() % 6 + 1;

      //////////// BOTTOM LAYER ////////////

      for(int n=0;n<nspots_bot;++n){

          Real spotsize = std::rand() % 40 + 5;
          spotsize = spotsize/100.0;

          IndicV.setZero();

          regime = std::rand() % 4 + 1;
          CenterX = std::rand() % 1000 - 500;
          CenterY = std::rand() % 1000 - 500;

          CenterX = CenterX/1000.0;
          CenterY = CenterY/1000.0;


          for (int i=0; i<nVert; ++i){

            if (Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
  	           IndicV(i) = 1;
            }

            if ((CenterX-spotsize)<-0.5){ // check if we overpassed the borders along X
              if (Vertices(i,0)>(0.5+(CenterX-spotsize+0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                IndicV(i) = 1;
              }
            }

            if ((CenterX+spotsize)>0.5){
                if (Vertices(i,0)<(-0.5+(CenterX+spotsize-0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY-spotsize)<-0.5){
                if (Vertices(i,1)>(0.5+(CenterY-spotsize+0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY+spotsize)>0.5){
                if (Vertices(i,1)<(-0.5+(CenterY+spotsize-0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }
          }

          for (int i=0; i<nFaces; ++i){
               if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {

                 if (growthRates_bot(i) == 0){
                   ActiveElms_bot(i) = ActiveElms_bot(i)+1;
                 }

                 h_bot(i) = h_ac(regime-1);
                 h_top(i) = h_ac(regime-1);
                 //h_top(i) = h_total - h_bot(i);
                 growthRates_bot(i) = growth(regime-1);

                 growthRates_bot_eqv(i) = growthRate1(regime-1);
                 growthRates_top_eqv(i) = growthRate2(regime-1);
                 Active_bot(i) = 1;
                 Nregime(i) = regime-1;
               }
             //}
          }

      }

      //////////// TOP LAYER ////////////

      for(int n=0;n<nspots_top;++n){

          Real spotsize = std::rand() % 40 + 5;
          spotsize = spotsize/100.0;

          IndicV.setZero();

          regime = std::rand() % 4 + 1;

          CenterX = std::rand() % 1000 - 500;
          CenterY = std::rand() % 1000 - 500;

          CenterX = CenterX/1000.0;
          CenterY = CenterY/1000.0;

          for (int i=0; i<nVert; ++i){

            if (Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
  	           IndicV(i) = 1;
            }

            if ((CenterX-spotsize)<-0.5){ // check if we overpassed the borders along X
              if (Vertices(i,0)>(0.5+(CenterX-spotsize+0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                IndicV(i) = 1;
              }
            }

            if ((CenterX+spotsize)>0.5){
                if (Vertices(i,0)<(-0.5+(CenterX+spotsize-0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY-spotsize)<-0.5){
                if (Vertices(i,1)>(0.5+(CenterY-spotsize+0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY+spotsize)>0.5){
                if (Vertices(i,1)<(-0.5+(CenterY+spotsize-0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }
          }

          for (int i=0; i<nFaces; ++i){
  	         if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {

               if (growthRates_top(i) == 0){
                 ActiveElms_top(i) = ActiveElms_top(i)+1;
               }

               growthRates_top(i) = growth(regime-1);
               h_top(i) = h_ac(regime-1);
               Active_top(i) = 1;

               if (Active_bot(i) != 1){
                 growthRates_top_eqv(i) = growthRate1(regime-1);
                 growthRates_bot_eqv(i) = growthRate2(regime-1);
                 h_bot(i) = h_ac(regime-1);
               }

               if (Active_bot(i) == 1){
                 growthRates_top_eqv(i) = growthRate2(Nregime(i)) + growthRate1(regime-1);
                 growthRates_bot_eqv(i) = growthRate1(Nregime(i)) + growthRate2(regime-1);
               }
              }
          }

      }

      helpers::write_matrix(tag+"_growthRates_bot.txt", growthRates_bot);
      helpers::write_matrix(tag+"_thickness_bot.txt", h_bot);
      
      helpers::write_matrix(tag+"_growthRates_top.txt", growthRates_top);
      helpers::write_matrix(tag+"_thickness_top.txt", h_top);
      
      helpers::write_matrix(tag+"_growthRates_bot_eqv.txt", growthRates_bot_eqv);
      helpers::write_matrix(tag+"_growthRates_top_eqv.txt", growthRates_top_eqv);
      
      helpers::write_matrix_binary(tag+"_growthRates_bot.dat", growthRates_bot);
      helpers::write_matrix_binary(tag+"_thickness_bot.dat", h_bot);
      
      helpers::write_matrix_binary(tag+"_growthRates_top.dat", growthRates_top);
      helpers::write_matrix_binary(tag+"_thickness_top.dat", h_top);
      
      helpers::write_matrix_binary(tag+"_growthRates_bot_eqv.dat", growthRates_bot_eqv);
      helpers::write_matrix_binary(tag+"_growthRates_top_eqv.dat", growthRates_top_eqv);

      /////////////////////////////////////////////

      tag = "TestRandom_"+ helpers::ToString(k,2);

      initForwardProblem();

      mesh.resetToRestState();

      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_bot_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_top_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

      MaterialProperties_Iso_Constant matprop_bot_eqv(E, nu, h_total, 0.0);
      MaterialProperties_Iso_Constant matprop_top_eqv(E, nu, h_total, 0.0);

      CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot_eqv(matprop_bot_eqv);
      CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top_eqv(matprop_top_eqv);
      EnergyOperatorList<tMesh> engOps_eqv({&engOp_bot_eqv, &engOp_top_eqv});

      // write initial condition
      mesh.writeToFile(tag+"_init");

      // dump
      dumpIso(growthRates_bot_eqv, growthRates_top_eqv, tag+"_init");

      const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
      const Real steppo = 1.0/nSwellingRuns;

      for(int s=0;s<nSwellingRuns;++s)
      {

          const Real swelling_fac = steppo*(s+1);

          // apply swelling
          GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, swelling_fac*growthRates_bot_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
          GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, swelling_fac*growthRates_top_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<top>());
          Real eps = 1e-2;
          minimizeEnergy(engOps_eqv, eps);

      }

      // dump
      dumpIso(growthRates_bot_eqv, growthRates_top_eqv, tag+"_final");

      // write
      mesh.writeToFile(tag+"_final");

      auto Vertices_Final_Eqv = mesh.getCurrentConfiguration().getVertices();
      helpers::write_matrix(tag+"_vertices_final.txt", Vertices_Final_Eqv);

    }

    tag = "TestRandom";
    helpers::write_matrix(tag+"_h_nu.txt", h_nu);
    helpers::write_matrix_binary(tag+"_h_nu.dat", h_nu);

}





void Sim_Bilayer_Growth::initForwardProblem()
{
    const std::string geometryCase = parser.parse<std::string>("-geometry", "");

    if (geometryCase == "external")
    // external mesh from the fname file
    {
        const std::string fname = parser.template parse<std::string>("-basename", "");
        IOGeometry geometry(fname);
        mesh.init(geometry);
    }
    else if (geometryCase == "external_rectangle_clamped")
    // external mesh from the fname file
    // all translations of vertices that lie in the margins are restricted
    {
        const std::string fname = parser.template parse<std::string>("-basename", "");
        const Real margin_width = parser.parse<Real>("-margin_width", 0.0); //margins to simulate the clamping frame: no eigensrain in this zone. 0.001 = 1mm
        const Real fixedX = parser.parse<Real>("-fixedX", true); // either we fix the borders aligned with x- or not
        const Real fixedY = parser.parse<Real>("-fixedY", true); // either we fix the borders aligned with y- or not
        IOGeometry_Rectangle_Clamped geometry(fname, margin_width, fixedX, fixedY);
        const Real clamped = parser.parse<Real>("-clamped", true);
        mesh.init(geometry, clamped);
    }
    else if (geometryCase == "rectangle")
    // regular mesh
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate_RightAngle geometry(Lx, Ly, relArea, false, false);
      mesh.init(geometry);
    }
    else if (geometryCase == "rectangle_allclamped")
    // regular mesh
    // all translations of vertices that lie in the margins are restricted
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      const Real margin_width = parser.parse<Real>("-margin_width", 0.0);
      const Real fixedX = parser.parse<Real>("-fixedX", true);   // either we fix the borders aligned with x- or not
      const Real fixedY = parser.parse<Real>("-fixedY", true);   // either we fix the borders aligned with y- or not
      RectangularPlate_RightAngle_Clamped geometry(Lx, Ly, relArea, margin_width, fixedX, fixedY);
      const Real clamped = parser.parse<Real>("-clamped", true);
      mesh.init(geometry, clamped);
    }
    else if (geometryCase == "rectangle_3clampvert")
    // regular mesh
    // one vertex is fixed along all three axes, one vertex is fixed along x- and z-, and one vertex is fixed only along z-
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate_3clampvert geometry(Lx, Ly, relArea, false, false);
      mesh.init(geometry);
    }
    else if (geometryCase == "rectangle_irreg") 
    //irregular mesh
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate geometry(Lx, Ly, relArea, {false,false}, {false,false});
      mesh.init(geometry);
    }
    else
    {
        std::cout << "No valid geometry defined. Options are \n";
        std::cout << "\t -geometry external\n";
        std::cout << "\t -geometry external_rectangle_clamped\n";
        std::cout << "\t -geometry rectangle\n";
        std::cout << "\t -geometry rectangle_allclamped\n";
        std::cout << "\t -geometry rectangle_3clampvert\n";
        std::cout << "\t -geometry rectangle_irreg\n";
        helpers::catastrophe("no valid geometry", __FILE__, __LINE__);
    }
}





void Sim_Bilayer_Growth::dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig)
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

    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXd gauss(nFaces);
    Eigen::VectorXd mean(nFaces);
    ComputeCurvatures<tMesh> computeCurvatures;
    computeCurvatures.compute(mesh, gauss, mean);
    writer.addScalarFieldToFaces(gauss, "gauss");
    writer.addScalarFieldToFaces(mean, "mean");

    writer.write(filename);
}




void Sim_Bilayer_Growth::dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig)
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
      computeCurvatures.computeDir(mesh, gauss, mean, PrincCurv1, PrincCurv2, CurvX, CurvY, Dir1, Dir2);

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



// eliminate any expansion from the margins
void Sim_Bilayer_Growth::MarginCut(const Real margin_x, const Real margin_y, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top)
{
  const auto Vertices = mesh.getRestConfiguration().getVertices();
  const auto Connect = mesh.getTopology().getFace2Vertices();

  const int nFaces = mesh.getNumberOfFaces();
  const int nVert = mesh.getNumberOfVertices();

  Eigen::VectorXi IndicV(nVert);
  IndicV.setZero();
  auto MinVert = Vertices.colwise().minCoeff();
  auto MaxVert = Vertices.colwise().maxCoeff();

  for (int i=0; i<nVert; ++i){
    if (Vertices(i,0)<=(MinVert(0)+margin_x) || Vertices(i,0)>=(MaxVert(0)-margin_x) || Vertices(i,1)<=(MinVert(1)+margin_y) || Vertices(i,1)>=(MaxVert(1)-margin_y)){
      IndicV(i) = 1;
    }
  }

  for (int i=0; i<nFaces; ++i){
     if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
       GrowthFacs_bot(i) = 0.0;
       GrowthFacs_top(i) = 0.0;
     }
  }
}




void Sim_Bilayer_Growth::init()
{
}




void  Sim_Bilayer_Growth::computeQuadraticForms(tVecMat2d & firstFF, tVecMat2d & secondFF)
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




// reformulate the growth from the step profile to the bilayer profile 
void Sim_Bilayer_Growth::Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot)
{
  *theta_top = heq*epseq*(3*h-2*heq)/std::pow(h,2);
  *theta_bot = heq*epseq*(2*heq-h)/std::pow(h,2);
}

// reformulate the growth from the bilayer profile to the step profile 
void Sim_Bilayer_Growth::Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot)
{
  *heq=(theta_top+3.0*theta_bot)*h/((theta_top+theta_bot)*2.0);
  *epseq=std::pow((theta_top+theta_bot),2)/(theta_top+3.0*theta_bot);
}
