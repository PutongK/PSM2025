//
//  Sim_InverseBilayer.cpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#include "Sim_InverseBilayer.hpp"
#include "Geometry.hpp"
#include "MaterialProperties.hpp"
#include "GrowthHelper.hpp"
#include "EnergyOperator_Inverse_Bilayer.hpp"
#include "CombinedOperator_Parametric_InverseGrowth_Bilayer.hpp"
#include "Parametrizer_InverseGrowth.hpp"
#include "HLBFGS_Wrapper_Parametrized.hpp"

#include "NonEuclideanConformalMapping.hpp"
#include "ConformalMappingBoundary.hpp"
#include "DirichletEnergyOperator_Inverse.hpp"
#include "EnergyOperatorList.hpp"

#include "QuadraticFormOperator.hpp"

#include "MeshQualityOperator.hpp"
#include "AreaDistortionOperator.hpp"

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

#include <igl/boundary_loop.h>

void Sim_InverseBilayer::init()
{

}

void Sim_InverseBilayer::run()
{
    const Real initialPerturbationFac = parser.parse<Real>("-perturbation", 0.0);

    const std::string runCase = parser.parse<std::string>("-case", "");
    if(runCase == "inversetest_one")
      inverseTestOne(initialPerturbationFac);
    else if(runCase == "inversetest_m")
      inverseTestMultiple(initialPerturbationFac);
    else if(runCase == "inversetest_optim")
        inverseTestOptim(initialPerturbationFac);
    else if(runCase == "inversetest_exact")
        inverseTestExact(initialPerturbationFac);
    else
    {
        std::cout << "No valid test case defined. Options are \n";
        std::cout << "\t -case inversetest_one\n";
        std::cout << "\t -case inversetest_m\n";
        std::cout << "\t -case inversetest_optim\n";
        std::cout << "\t -case inversetest_exact\n";
    }
}

struct Inverse_Machinery_Wrapper
{

    virtual int minimize(const std::string outname, const Real eps) = 0;
    virtual Real get_lastnorm() const = 0;
    virtual const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const = 0;
    virtual const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const = 0;
    virtual ~Inverse_Machinery_Wrapper()
    {}
};


struct Inverse_Bilayer_Iso_Wrapper : Inverse_Machinery_Wrapper
{
    typedef BilayerMesh tMesh;

    Parametrizer_Bilayer_IsoGrowth<tMesh> parametrizer;
    HLBFGS_Methods::HLBFGS_EnergyOp_Parametrized<tMesh, Parametrizer_Bilayer_IsoGrowth, true> hlbfgs_wrapper;

    Inverse_Bilayer_Iso_Wrapper(tMesh & mesh, const EnergyOperatorList<tMesh> & engOp_list, const Real initVal):
    parametrizer(mesh),
    hlbfgs_wrapper(mesh, engOp_list, parametrizer)
    {
        // initialize the data
        const int nVars = parametrizer.getNumberOfVariables();
        Eigen::VectorXd initGrowthFacs = Eigen::VectorXd::Constant(nVars, initVal);
        parametrizer.initSolution(initGrowthFacs);
    }

    Inverse_Bilayer_Iso_Wrapper(tMesh & mesh, const EnergyOperatorList<tMesh> & engOp_list, const Eigen::Ref<const Eigen::VectorXd> initVals):
    parametrizer(mesh),
    hlbfgs_wrapper(mesh, engOp_list, parametrizer)
    {
        // initialize the data
        const int nVars = parametrizer.getNumberOfVariables();
        assert(nVars == initVals.rows());
        _unused(nVars);
        parametrizer.initSolution(initVals);
    }

    int minimize(const std::string outname, const Real eps) override
    {
        return hlbfgs_wrapper.minimize(outname, eps);
    }

    Real get_lastnorm() const override
    {
        return hlbfgs_wrapper.get_lastnorm();
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const override
    {
        return parametrizer.getGrowthRates_bot();
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const override
    {
        return parametrizer.getGrowthRates_top();
    }
};




struct Inverse_Bilayer_Iso_Wrapper_Bound : Inverse_Machinery_Wrapper
{
    typedef BilayerMesh tMesh;

    Parametrizer_Bilayer_IsoGrowth_WithBounds<tMesh> parametrizer;
    HLBFGS_Methods::HLBFGS_EnergyOp_Parametrized<tMesh, Parametrizer_Bilayer_IsoGrowth_WithBounds, true> hlbfgs_wrapper;

    Inverse_Bilayer_Iso_Wrapper_Bound(tMesh & mesh, const EnergyOperatorList<tMesh> & engOp_list, const Real initVal, const Real minVal, const Real maxVal):
    parametrizer(mesh, minVal, maxVal),
    hlbfgs_wrapper(mesh, engOp_list, parametrizer)
    {
        // initialize the data
        const int nVars = parametrizer.getNumberOfVariables();
        Eigen::VectorXd initGrowthFacs = Eigen::VectorXd::Constant(nVars, initVal);
        parametrizer.initSolution(initGrowthFacs);
    }

    Inverse_Bilayer_Iso_Wrapper_Bound(tMesh & mesh, const EnergyOperatorList<tMesh> & engOp_list, const Eigen::Ref<const Eigen::VectorXd> initVals, const Real minVal, const Real maxVal):
    parametrizer(mesh, minVal, maxVal),
    hlbfgs_wrapper(mesh, engOp_list, parametrizer)
    {
        // initialize the data
        const int nVars = parametrizer.getNumberOfVariables();
        assert(nVars == initVals.rows());
        _unused(nVars);
        parametrizer.initSolution(initVals);
    }

    int minimize(const std::string outname, const Real eps) override
    {
        return hlbfgs_wrapper.minimize(outname, eps);
    }

    Real get_lastnorm() const override
    {
        return hlbfgs_wrapper.get_lastnorm();
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const override
    {
        return parametrizer.getGrowthRates_bot();
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const override
    {
        return parametrizer.getGrowthRates_top();
    }
};




// solve the inverse problem for any shape
// user can assign the growth anisotropy, where the principal growth direction and the anisotropy coefficient are constant over the plate
void Sim_InverseBilayer::inverseTestOne(const Real deformRad)
{

tag = "inverseTestOne";

const Real E = 1;
const Real nu = parser.parse<Real>("-nu", 0.33);
const Real h_total = parser.parse<Real>("-h_total", 0.00302); //total thickness

const Real toler = parser.parse<Real>("-toler", 0.01); //stop criterion 1: (Hausdorff distance between the current and the target shape)/deflection < toler
const int niter = parser.parse<int>("-maxiter", 3); //stop criterion 2: maximal number of iterations, must be superior than 2
const Real maxratio = parser.parse<Real>("-maxratio", 5.0); //maximal ratio for adjustment of a local growth on one iteration
const bool just_curvature = parser.parse<bool>("-just_curvature", false); //if we consider the local stretching of the target shape (just_curvature == false) or not (just_curvature == true)

const std::string segmentation = parser.parse<std::string>("-segmentation", ""); //options: "grouping", "kmeans"
const int nClusters = parser.parse<int>("-nClusters", 4); //number of clusters: maximum 4 in the case of grouping, illimited in the case of kmeans 

const int nTries_kmeans = parser.parse<int>("-nTries_kmeans", 10); //number of initializations of the kmeans algorithm 

const bool do_filtering = parser.parse<bool>("-do_filtering", false); //the filtering algorithm after segmentation

const Real margin_width = parser.parse<Real>("-margin_width", 0.0); //margins to simulate the clamping frame: no eigensrain in this zone. 0.001 = 1mm

const Real ortho_coeff = parser.parse<Real>("-ortho_coeff", 0.014); // orthotropy coefficient
const Real growthAngle = parser.parse<Real>("-growth_angle", 0.0)*M_PI; // principal growth direction (angle with respect to x-axis)


//////////////////////////////////////////////////
Eigen::VectorXd growthRate1 (nClusters+1);
growthRate1.setZero();
Eigen::VectorXd growthRate2 (nClusters+1);
growthRate2.setZero();

// specifying predefined growth (peening regimes)
growthRate1(1) = parser.parse<Real>("-growth1t", 0.001475);
growthRate2(1) = parser.parse<Real>("-growth1b", -growthRate1(1));

if (nClusters>1){
  growthRate1(2) = parser.parse<Real>("-growth2t", 0.0012);
  growthRate2(2) = parser.parse<Real>("-growth2b", -growthRate1(2));
}

if (nClusters>2){
  growthRate1(3) = parser.parse<Real>("-growth3t", 0.00082);
  growthRate2(3) = parser.parse<Real>("-growth3b", -growthRate1(3));
}

if (nClusters>3){
  growthRate1(4) = parser.parse<Real>("-growth4t", 0.00047);
  growthRate2(4) = parser.parse<Real>("-growth4b", -growthRate1(4));
}
////////////////////////////////////

// initialize the mesh and define target surface
initInverseProblem();


// define combinations of peening regimes
const int growthfactors_size = std::pow((nClusters+1),2);
Eigen::MatrixXd growthfactors (growthfactors_size,2); // 2*(4+6)+4+1 = (one side+different)*same_thing_but_inversed + same_from_both_sides + zero
growthfactors.setZero();

for (int i=0 ; i<nClusters+1 ; ++i){
  for (int j=0 ; j<nClusters+1 ; ++j){
    growthfactors(i*(nClusters+1)+j,0) = growthRate1(i) + growthRate2(j);
    growthfactors(i*(nClusters+1)+j,1) = growthRate2(i) + growthRate1(j);
  }
}
//std::cout << "\n" << "growthfactors\n" << growthfactors << "\n" << std::endl;



const int nFaces = mesh.getNumberOfFaces();
//dump the target shape curvature
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


// explore face connectivity (for filtering afterwards)
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


// explore local stretching of the current and the target shapes 
tVecMat2d firstFF,secondFF;
computeQuadraticForms(firstFF, secondFF);
Eigen::VectorXd area_t(nFaces);
Eigen::VectorXd base_area(nFaces);
Eigen::VectorXd ratio_area(nFaces);
for(int i=0;i<nFaces;++i)
{
    const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
    Eigen::MatrixXd rxy_base(3,2);
    rxy_base << rinfo.e1, rinfo.e2;

    base_area(i) = 0.5*std::sqrt((rxy_base.transpose() * rxy_base).determinant());
    area_t(i) = 0.5*std::sqrt(firstFF[i].determinant());
    ratio_area(i) = area_t(i)/base_area(i);
}



Eigen::VectorXd GrowthFacs_top(nFaces);
GrowthFacs_top.setZero();
Eigen::VectorXd GrowthFacs_bot(nFaces);
GrowthFacs_bot.setZero();
Eigen::MatrixXd GrowthFacs_unclustered(nFaces,2);
GrowthFacs_unclustered.setZero();
Eigen::VectorXi Clusters(nFaces);
Clusters.setZero();
const auto cvertices = mesh.getCurrentConfiguration().getVertices();
const auto cface2vertices = mesh.getTopology().getFace2Vertices();


// define operators
MaterialProperties_Iso_Constant matprop(E, nu, h_total);
CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop);
CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop);
EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});


// make the initial guess
std::vector<GrowthState> growth_bot_exact, growth_top_exact;
const bool flipLayers = parser.parse<bool>("-fliplayers", false);
computeGrowthForCurrentState(matprop, matprop, growth_bot_exact, growth_top_exact, true, flipLayers, just_curvature);
for(int i=0;i<nFaces;++i)
{
  const DecomposedGrowthState & decomp_bot = growth_bot_exact[i].getDecomposedFinalState();
  const DecomposedGrowthState & decomp_top = growth_top_exact[i].getDecomposedFinalState();

  // exract the growth factors
  const Real s1_bot = decomp_bot.get_s1();
  const Real s2_bot = decomp_bot.get_s2();

  const Real s1_top = decomp_top.get_s1();
  const Real s2_top = decomp_top.get_s2();

  GrowthFacs_bot(i) = std::sqrt(s1_bot*s2_bot) - 1;
  GrowthFacs_top(i) = std::sqrt(s1_top*s2_top) - 1;
}

//eliminate the growth within the margins
if (margin_width > 0.0) MarginCut(margin_width, GrowthFacs_bot, GrowthFacs_top, Clusters);


//apply the orthotropy coefficient
const Eigen::VectorXd growthAngles = Eigen::VectorXd::Constant(nFaces, growthAngle);
Eigen::VectorXd growthRates_1_t(nFaces);
Eigen::VectorXd growthRates_1_b(nFaces);
Eigen::VectorXd growthRates_2_t(nFaces);
Eigen::VectorXd growthRates_2_b(nFaces);
for (int i=0; i<nFaces; i++){
  growthRates_1_t(i) = GrowthFacs_top(i)*(1.0+ortho_coeff);
  growthRates_1_b(i) = GrowthFacs_bot(i)*(1.0+ortho_coeff);
  growthRates_2_t(i) = GrowthFacs_top(i)*(1.0-ortho_coeff);
  growthRates_2_b(i) = GrowthFacs_bot(i)*(1.0-ortho_coeff);
}
mesh.resetToRestState();
dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_inverse_00");
mesh.writeToFile(tag + "_inverse_00");



// run the forward problem
{
    GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_b, growthRates_2_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
    GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_t, growthRates_2_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

    printf("%10.10e \t %10.10e \t %10.10e\n", engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
    // minimize energy
    Real eps = 1e-2;
    minimizeEnergy(engOps, eps);
}

dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_final_00");
mesh.writeToFile(tag+"_final_00");




// compute and dump the errors
Eigen::VectorXd Maxerr(niter+1);
Maxerr.setZero();
Eigen::VectorXd Relerr(niter+1);
Relerr.setZero();
Eigen::VectorXd Sqsumerr(niter+1);
Sqsumerr.setZero();
Eigen::VectorXd Curverr(niter+1);
Curverr.setZero();

const std::string fname1 = tag + "_final_00.vtp";
std::string fname2;

const std::string geometryCase = parser.template parse<std::string>("-geometry", "");

if(geometryCase == "external")
{
    fname2 = parser.template parse<std::string>("-filename", "");
} else if(geometryCase == "external_dat")
{
    const std::string filetag = parser.template parse<std::string>("-filetag", "");
    fname2 = filetag + ".vtp";
} else
{
    fname2 = tag + "_target.vtp";
}

Real sqsumerror;
Real maxerror;
Real relerror;
ErrorMap(fname1, fname2, 0, &maxerror, &relerror, &sqsumerror);
Maxerr(0) = maxerror;
Relerr(0) = relerror;
Sqsumerr(0) = sqsumerror;

std::cout<< "\n\n" << "MAX ERROR:"<< "\n" << Maxerr << "\n" << std::endl;
std::cout<< "\n\n" << "RELATIVE ERROR:"<< "\n" << Relerr << "\n" << std::endl;
std::cout<< "\n\n" << "SQUARED SUM ERROR:"<< "\n" << Sqsumerr << "\n" << std::endl;

Eigen::VectorXd gauss_c(nFaces);
Eigen::VectorXd mean_c(nFaces);
Eigen::VectorXd princcurv1_c(nFaces);
Eigen::VectorXd princcurv2_c(nFaces);
computeCurvatures.computeNew(mesh, gauss_c, mean_c, princcurv1_c, princcurv2_c);
for(int i=0;i<nFaces;++i)
{
  Curverr(0) += std::abs(mean_t(i)-mean_c(i));
}
Curverr(0) /= nFaces;
std::cout<< "\n\n" << "CURVATURE ERROR:"<< "\n" << Curverr << "\n" << std::endl;



Eigen::VectorXd Relerr_prev(niter+2);
Relerr_prev.setZero();
Relerr_prev(0)=1.0;
Relerr_prev(1)=1.0;

int t = 1;
int indicator = 0;

// the iterative inverse problem resolution
while (indicator != 2){

    // indicator = 0: do iterative adjustment of the growthfactors
    // indicator = 1: do grouping/clustering and filtering
    // indicator = 2: exit the loop

    if (indicator == 0){
      int m = -1;
      GrowthAdjust(nFaces, maxratio, area_t, GrowthFacs_bot, GrowthFacs_top, mean_t, princcurv1_t, m, t, just_curvature);

      if (t==niter-1) {
        dumpIso(GrowthFacs_bot, GrowthFacs_top, tag+"_unclustered_"+helpers::ToString(t,2), true);
      }
    }
    if (margin_width > 0.0) MarginCut(margin_width, GrowthFacs_bot, GrowthFacs_top, Clusters);


    //Perform grouping/clustering and filtering before exiting the loop
    if (indicator == 1){
      if (segmentation == "grouping")
      {
        Grouping(nFaces, nClusters, GrowthFacs_bot, GrowthFacs_top, Clusters, growthfactors, adj_faces, t);
      } 
      else if (segmentation == "kmeans")
      {
        Kmeans(nTries_kmeans, nClusters, nFaces, h_total, GrowthFacs_bot, GrowthFacs_top, Clusters);
      }
      if (do_filtering)
      {
        Filtering(nFaces, GrowthFacs_bot, GrowthFacs_top, Clusters, adj_faces, t);
      }

      helpers::write_matrix(tag+"_growthRates_bot.txt", GrowthFacs_bot);
      helpers::write_matrix(tag+"_growthRates_top.txt", GrowthFacs_top);
      helpers::write_matrix_binary(tag+"_growthRates_bot.dat", GrowthFacs_bot);
      helpers::write_matrix_binary(tag+"_growthRates_top.dat", GrowthFacs_top);
    }



    mesh.resetToRestState();

    //apply the orthotropy coefficient
    for (int i=0; i<nFaces; i++){
      growthRates_1_t(i) = GrowthFacs_top(i)*(1.0+ortho_coeff);
      growthRates_1_b(i) = GrowthFacs_bot(i)*(1.0+ortho_coeff);
      growthRates_2_t(i) = GrowthFacs_top(i)*(1.0-ortho_coeff);
      growthRates_2_b(i) = GrowthFacs_bot(i)*(1.0-ortho_coeff);
    }
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_inverse_"+helpers::ToString(t,2));

    // solve the forward problem
    {
      GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_b, growthRates_2_b, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_t, growthRates_2_t, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

      printf("%10.10e \t %10.10e \t %10.10e\n", engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
      // minimize energy
      Real eps = 1e-2;
      minimizeEnergy(engOps, eps);
    }
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag + "_final_"+helpers::ToString(t,2));
    mesh.writeToFile(tag + "_final_"+helpers::ToString(t,2));
    std::cout<< "\n\n\n\n\n\n" << "End of iteration " << t << "\n\n\n\n\n\n\n" << std::endl;




    //compute and dump the errors
    const std::string fname1 = tag + "_final_" + helpers::ToString(t,2) + ".vtp";
    std::string fname2;

    if(geometryCase == "external")
    {
        fname2 = parser.template parse<std::string>("-filename", "");
    } else if(geometryCase == "external_dat")
    {
        const std::string filetag = parser.template parse<std::string>("-filetag", "");
        fname2 = filetag + ".vtp";
    } else
    {
        fname2 = tag + "_target.vtp";
    }

    ErrorMap(fname1, fname2, t, &maxerror, &relerror, &sqsumerror);
    Maxerr(t) = maxerror;
    Relerr(t) = relerror;
    Relerr_prev(t+1) = Relerr(t);
    Sqsumerr(t) = sqsumerror;

    computeCurvatures.computeNew(mesh, gauss_c, mean_c, princcurv1_c, princcurv2_c);
    Curverr(t) = 0;
    for(int i=0;i<nFaces;++i)
    {
      Curverr(t) += std::abs(mean_t(i)-mean_c(i));
    }
    Curverr(t) /= nFaces;

    std::cout<< "\n\n" << "MAX ERROR:"<< "\n" << Maxerr << "\n" << std::endl;
    std::cout<< "\n\n" << "RELATIVE ERROR:"<< "\n" << Relerr << "\n" << std::endl;
    std::cout<< "\n\n" << "SQUARED SUM ERROR:"<< "\n" << Sqsumerr << "\n" << std::endl;
    std::cout<< "\n\n" << "CURVATURE ERROR:"<< "\n" << Curverr << "\n" << std::endl;



    //stop conditions
    if (indicator == 1){
      indicator = 2;
    }
    

    if(indicator == 0){
      if (Relerr(t) <= toler || t==niter-1 || (Relerr(t) > toler && Relerr(t) > Relerr_prev(t))){
        if (segmentation == "grouping" || segmentation == "kmeans"){
          indicator = 1;
        } else {
          indicator = 2;
        }
      }
    }

    t++;
  }
}





// solve the inverse problem for the random shapes generated by the TestRandomPatterns() method in Sim_Bilayer_Growth.cpp
// uses only isotropic in-plane growth 
void Sim_InverseBilayer::inverseTestMultiple(const Real deformRad)
{

tag = "inverseTestMultiple";

Eigen::MatrixXd h_nu;
helpers::read_matrix_binary("TestRandom_h_nu.dat", h_nu); //file containing thicknesses and Poisson's ratios of the target shapes
const Real E = 1;

const int ntests = parser.parse<int>("-ntests", 100); //number of tests
const int start = parser.parse<int>("-start", 0); //the test that we start with

const Real toler = parser.parse<Real>("-toler", 0.01); //stop criterion 1: (Hausdorff distance between the current and the target shape)/deflection < toler
const int niter = parser.parse<int>("-maxiter", 3); //stop criterion 2: maximal number of iterations, must be superior than 2
const Real maxratio = parser.parse<Real>("-maxratio", 2.0); //maximal ratio for adjustment of a local growth on one iteration
const bool just_curvature = parser.parse<bool>("-just_curvature", false); //if we consider the local stretching of the target shape (just_curvature == false) or not (just_curvature == true)

const std::string segmentation = parser.parse<std::string>("-segmentation", ""); //options: "grouping", "kmeans"
const int nClusters = parser.parse<int>("-nClusters", 4); //number of clusters: maximum 4 in the case of grouping, illimited in the case of kmeans 

const int nTries_kmeans = parser.parse<int>("-nTries_kmeans", 10); //number of initializations of the kmeans algorithm 

const bool do_filtering = parser.parse<bool>("-do_filtering", false); //the filtering algorithm after segmentation

//////////////////////////////////////////////////
Eigen::VectorXd growthRate1 (nClusters+1);
growthRate1.setZero();
Eigen::VectorXd growthRate2 (nClusters+1);
growthRate2.setZero();

growthRate1(1) = parser.parse<Real>("-growth1t", 0.00047);
growthRate2(1) = parser.parse<Real>("-growth1b", -growthRate1(1));

if (nClusters>1){
  growthRate1(2) = parser.parse<Real>("-growth2t", 0.00082);
  growthRate2(2) = parser.parse<Real>("-growth2b", -growthRate1(2));
}

if (nClusters>2){
  growthRate1(3) = parser.parse<Real>("-growth3t", 0.0012);
  growthRate2(3) = parser.parse<Real>("-growth3b", -growthRate1(3));
}

if (nClusters>3){
  growthRate1(4) = parser.parse<Real>("-growth4t", 0.001475);
  growthRate2(4) = parser.parse<Real>("-growth4b", -growthRate1(4));
}
////////////////////////////////////

Eigen::VectorXd Maxerr(ntests);
Eigen::VectorXd Relerr(ntests);
Eigen::VectorXd Sqsumerr(ntests);

Eigen::VectorXd Maxerr_grouping(ntests);
Eigen::VectorXd Relerr_grouping(ntests);
Eigen::VectorXd Sqsumerr_grouping(ntests);

Eigen::VectorXd Maxerr_kmeans(ntests);
Eigen::VectorXd Relerr_kmeans(ntests);
Eigen::VectorXd Sqsumerr_kmeans(ntests);

Eigen::MatrixXd Rel_err_GLOBAL(niter+1,ntests);
Rel_err_GLOBAL.setZero();

Eigen::MatrixXd Max_err_GLOBAL(niter+1,ntests);
Max_err_GLOBAL.setZero();

Eigen::MatrixXd Sqsumerr_GLOBAL(niter+1,ntests);
Sqsumerr_GLOBAL.setZero();


//////////////////////////////////////////////////////////////////////////////////
// loop over the target shapes
for(int m=start;m<start+ntests;++m){

  const Real h_total = h_nu(m,0); 
  const Real nu = h_nu(m,1);

  // define combinations of peening regimes
  const int growthfactors_size = std::pow((nClusters+1),2);
  Eigen::MatrixXd growthfactors (growthfactors_size,2);
  growthfactors.setZero();

  for (int i=0 ; i<nClusters+1 ; ++i){
    for (int j=0 ; j<nClusters+1 ; ++j){
      growthfactors(i*(nClusters+1)+j,0) = growthRate1(i) + growthRate2(j);
      growthfactors(i*(nClusters+1)+j,1) = growthRate2(i) + growthRate1(j);
    }
  }

  //std::cout << "\n" << "growthfactors\n" << growthfactors << "\n" << std::endl;

  // initialize the mesh and define the target surface
  const std::string filetag = "TestRandom_" + helpers::ToString(m,2) + "_final";
  const std::string basetag = "TestRandom_" + helpers::ToString(m,2) + "_init";

  // restart the mesh from the filetag case
  mesh.readFromFile(filetag);

  // load the vertices and edge directors of the base state
  Eigen::MatrixXd vertexdata_rest;
  Eigen::VectorXd edgedata_rest;
  helpers::read_matrix_binary(basetag+"_vertices.dat", vertexdata_rest);
  helpers::read_matrix_binary(basetag+"_edgedirs.dat", edgedata_rest);

  mesh.getRestConfiguration().getVertices() = vertexdata_rest;
  mesh.getRestConfiguration().getEdgeDirectors() = edgedata_rest;

  const int nFaces = mesh.getNumberOfFaces();

  // dump the target shape curvature
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



  // explore face connectivity (for filtering afterwards)
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


  Eigen::VectorXd GrowthFacs_top(nFaces);
  GrowthFacs_top.setZero();

  Eigen::VectorXd GrowthFacs_bot(nFaces);
  GrowthFacs_bot.setZero();

  Eigen::MatrixXd GrowthFacs_unclustered(nFaces,2);
  GrowthFacs_unclustered.setZero();

  Eigen::VectorXi Clusters(nFaces);
  Clusters.setZero();


  // make the initial guess
  // define operators
  MaterialProperties_Iso_Constant matprop(E, nu, h_total);
  CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop);
  CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop);
  EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

  std::vector<GrowthState> growth_bot_exact, growth_top_exact;
  const bool flipLayers = parser.parse<bool>("-fliplayers", false);
  computeGrowthForCurrentState(matprop, matprop, growth_bot_exact, growth_top_exact, true, flipLayers, just_curvature);

  for(int i=0;i<nFaces;++i)
  {
    const DecomposedGrowthState & decomp_bot = growth_bot_exact[i].getDecomposedFinalState();
    const DecomposedGrowthState & decomp_top = growth_top_exact[i].getDecomposedFinalState();

    // exract the growth factors
    const Real s1_bot = decomp_bot.get_s1();
    const Real s2_bot = decomp_bot.get_s2();

    const Real s1_top = decomp_top.get_s1();
    const Real s2_top = decomp_top.get_s2();

    GrowthFacs_bot(i) = std::sqrt(s1_bot*s2_bot) - 1;
    GrowthFacs_top(i) = std::sqrt(s1_top*s2_top) - 1;
  }



  // explore local stretching of the current and the target shapes 
  tVecMat2d firstFF,secondFF;
  computeQuadraticForms(firstFF, secondFF);
  Eigen::VectorXd area_t(nFaces);
  Eigen::VectorXd base_area(nFaces);
  Eigen::VectorXd ratio_area(nFaces);
  for(int i=0;i<nFaces;++i)
  {
      const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
      Eigen::MatrixXd rxy_base(3,2);
      rxy_base << rinfo.e1, rinfo.e2;

      base_area(i) = 0.5*std::sqrt((rxy_base.transpose() * rxy_base).determinant());
      area_t(i) = 0.5*std::sqrt(firstFF[i].determinant());
      ratio_area(i) = area_t(i)/base_area(i);

  }

  const auto cvertices = mesh.getRestConfiguration().getVertices();
  const auto cface2vertices = mesh.getTopology().getFace2Vertices();
  WriteVTK writer(cvertices, cface2vertices);
  writer.addScalarFieldToFaces(base_area, "base_area");
  writer.addScalarFieldToFaces(area_t, "target_area");
  writer.addScalarFieldToFaces(ratio_area, "ratio_area");
  writer.write(tag+"_areas_00");



  // apply growth and solve the forward problem

  mesh.resetToRestState();
  dumpIso(GrowthFacs_bot, GrowthFacs_top, tag+"_inverse_"+helpers::ToString(m,2)+"_00", true);
  mesh.writeToFile(tag+"_inverse_"+helpers::ToString(m,2)+"_00");

  {
      // apply swelling
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_bot, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_top, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

      printf("%10.10e \t %10.10e \t %10.10e\n", engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
      // minimize energy
      Real eps = 1e-2;
      minimizeEnergy(engOps, eps);
  }

  dumpIso(GrowthFacs_bot, GrowthFacs_top, tag+"_final_"+helpers::ToString(m,2)+"_00", false);
  mesh.writeToFile(tag+"_final_"+helpers::ToString(m,2)+"_00");




  // compute the error and dump
  Eigen::VectorXd Maxerr1(niter+1);
  Maxerr1.setZero();
  Eigen::VectorXd Relerr1(niter+1);
  Relerr1.setZero();
  Eigen::VectorXd Sqsumerr1(niter+1);
  Sqsumerr1.setZero();

  const std::string fname1 = tag + "_final_" + helpers::ToString(m,2) + "_00.vtp";
  const std::string fname2 = "TestRandom_" + helpers::ToString(m,2) + "_final.vtp";

  Real sqsumerror;
  Real maxerror;
  Real relerror;
  ErrorMap(fname1, fname2, 0, &maxerror, &relerror, &sqsumerror);
  Maxerr1(0) = maxerror;
  Relerr1(0) = relerror;
  Sqsumerr1(0) = sqsumerror;

  std::cout<< "\n\n" << "MAX ERROR:"<< "\n" << Maxerr1 << "\n" << std::endl;
  std::cout<< "\n\n" << "RELATIVE ERROR:"<< "\n" << Relerr1 << "\n" << std::endl;
  std::cout<< "\n\n" << "SQUARED SUM ERROR:"<< "\n" << Sqsumerr1 << "\n" << std::endl;




  Eigen::VectorXd Relerr_prev(niter+2);
  Relerr_prev.setZero();
  Relerr_prev(0)=1.0;
  Relerr_prev(1)=1.0;

  int t = 1;
  int indicator = 0;


  //the iterative inverse problem resolution
  //////////////////////////////////////////////////////////////////////////////////
  while (indicator != 2){
  // indicator = 0: do iterative adjustment of the growthfactors
  // indicator = 1: do grouping/clustering and filtering
  // indicator = 2: exit the loop

      // adjust the growth
      if (indicator == 0){
        GrowthAdjust(nFaces, maxratio, area_t, GrowthFacs_bot, GrowthFacs_top, mean_t, princcurv1_t, m, t, just_curvature);
        if (t==niter-1) {
          dumpIso(GrowthFacs_bot, GrowthFacs_top, tag+"_unclustered_"+helpers::ToString(m,2)+"_"+helpers::ToString(t,2), true);
        }
      }


      // do grouping or kmeans and filtering
      if (indicator == 1){
        if (segmentation == "grouping")
        {
          Grouping(nFaces, nClusters, GrowthFacs_bot, GrowthFacs_top, Clusters, growthfactors, adj_faces, t);
        } 
        else if (segmentation == "kmeans")
        {
          Kmeans(nTries_kmeans, nClusters, nFaces, h_total, GrowthFacs_bot, GrowthFacs_top, Clusters);
        }   
        if ((segmentation == "grouping" || segmentation == "kmeans") && do_filtering)
        {
          Filtering(nFaces, GrowthFacs_bot, GrowthFacs_top, Clusters, adj_faces, t);
        }
      }



      // solve the forward problem
      mesh.resetToRestState();

      dumpIso(GrowthFacs_bot, GrowthFacs_top, tag+"_inverse_"+helpers::ToString(m,2)+"_"+helpers::ToString(t,2), true);

      {
        // apply swelling
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_bot, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_top, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

        printf("%10.10e \t %10.10e \t %10.10e\n", engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
        // minimize energy
        Real eps = 1e-2;
        minimizeEnergy(engOps, eps);
      }

      dumpIso(GrowthFacs_bot, GrowthFacs_top, tag + "_final_"+helpers::ToString(m,2)+"_"+helpers::ToString(t,2), false);
      mesh.writeToFile(tag + "_final_"+helpers::ToString(m,2)+"_"+helpers::ToString(t,2));
      std::cout<< "\n\n\n\n\n\n" << "Test " << m << "\n\nEnd of iteration " << t << "\n\n\n\n\n\n\n" << std::endl;





      //compute the error
      const std::string fname1 = tag + "_final_" + helpers::ToString(m,2) + "_" + helpers::ToString(t,2) + ".vtp";
      const std::string fname2 = "TestRandom_" + helpers::ToString(m,2) + "_final.vtp";

      ErrorMap(fname1, fname2, t, &maxerror, &relerror, &sqsumerror);
      Maxerr1(t) = maxerror;
      Relerr1(t) = relerror;
      Relerr_prev(t+1) = Relerr1(t);
      Sqsumerr1(t) = sqsumerror;

      std::cout<< "\n\n" << "MAX ERROR:"<< "\n" << Maxerr1 << "\n" << std::endl;
      std::cout<< "\n\n" << "RELATIVE ERROR:"<< "\n" << Relerr1 << "\n" << std::endl;
      std::cout<< "\n\n" << "SQUARED SUM ERROR:"<< "\n" << Sqsumerr1 << "\n" << std::endl;


      // STOP CONDITIONS
      if (indicator == 1){
        indicator = 2;
        t--;
      }

      if(indicator == 0){
        if (Relerr1(t) <= toler || t==niter-1 || (Relerr1(t) > toler && Relerr1(t) > Relerr_prev(t))){
          if (segmentation == "grouping" || segmentation == "kmeans"){
            indicator = 1;
          } else {
            indicator = 2;
            t--;
          }
        }
      }

      t++;
    }


    // dump errors for all tests
    Maxerr(m-start) = Maxerr1(t-1);
    Relerr(m-start) = Relerr1(t-1);
    Sqsumerr(m-start) = Sqsumerr1(t-1);

    if (segmentation == "grouping"){
      Maxerr_grouping(m-start) = Maxerr1(t);
      Relerr_grouping(m-start) = Relerr1(t);
      Sqsumerr_grouping(m-start) = Sqsumerr1(t);
    }
    else if (segmentation == "kmeans"){
      Maxerr_kmeans(m-start) = Maxerr1(t);
      Relerr_kmeans(m-start) = Relerr1(t);
      Sqsumerr_kmeans(m-start) = Sqsumerr1(t);
    }


    std::cout<< "\n\n\n\n\n\n" << "MAX ERROR:"<< "\n\n" << Maxerr << "\n\n" << std::endl;
    std::cout<< "\n\n\n\n\n\n" << "RELATIVE ERROR:"<< "\n\n" << Relerr << "\n\n" << std::endl;
    std::cout<< "\n\n\n\n\n\n" << "SQUARED SUM ERROR:"<< "\n\n" << Sqsumerr << "\n\n" << std::endl;

    helpers::write_matrix("MAX_ERROR.txt", Maxerr);
    helpers::write_matrix("RELATIVE_ERROR.txt", Relerr);
    helpers::write_matrix("sqsumerror.txt", Sqsumerr);

    if (segmentation == "grouping"){
      std::cout<< "\n\n\n\n\n\n" << "MAX ERROR_GROUPING:"<< "\n\n" << Maxerr_grouping << "\n\n" << std::endl;
      std::cout<< "\n\n\n\n\n\n" << "RELATIVE ERROR_GROUPING:"<< "\n\n" << Relerr_grouping << "\n\n" << std::endl;
      std::cout<< "\n\n\n\n\n\n" << "SQUARED SUM ERROR_GROUPING:"<< "\n\n" << Sqsumerr_grouping << "\n\n" << std::endl;

      helpers::write_matrix("MAX_ERROR_GROUPING.txt", Maxerr_grouping);
      helpers::write_matrix("RELATIVE_ERROR_GROUPING.txt", Relerr_grouping);
      helpers::write_matrix("sqsumerror_GROUPING.txt", Sqsumerr_grouping);
    }
    else if (segmentation == "kmeans"){
      std::cout<< "\n\n\n\n\n\n" << "MAX ERROR_KMEANS:"<< "\n\n" << Maxerr_kmeans << "\n\n" << std::endl;
      std::cout<< "\n\n\n\n\n\n" << "RELATIVE ERROR_KMEANS:"<< "\n\n" << Relerr_kmeans << "\n\n" << std::endl;
      std::cout<< "\n\n\n\n\n\n" << "SQUARED SUM ERROR_KMEANS:"<< "\n\n" << Sqsumerr_kmeans << "\n\n" << std::endl;

      helpers::write_matrix("MAX_ERROR_KMEANS.txt", Maxerr_kmeans);
      helpers::write_matrix("RELATIVE_ERROR_KMEANS.txt", Relerr_kmeans);
      helpers::write_matrix("sqsumerror_KMEANS.txt", Sqsumerr_kmeans);
    }

    for(int y=0; y<t+1; ++y){
      Rel_err_GLOBAL(y,m-start)=Relerr1(y);
    }
    helpers::write_matrix("REL_ERROR_GLOBAL.txt", Rel_err_GLOBAL);

    for(int y=0; y<t+1; ++y){
      Max_err_GLOBAL(y,m-start)=Maxerr1(y);
    }
    helpers::write_matrix("MAX_ERROR_GLOBAL.txt", Max_err_GLOBAL);

    for(int y=0; y<t+1; ++y){
      Sqsumerr_GLOBAL(y,m-start)=Sqsumerr1(y);
    }
    helpers::write_matrix("SQ_SUM_GLOBAL.txt", Sqsumerr_GLOBAL);

  }

}





//Analytical resolution of the inverse by prescription of orthotropic in-plane growth to both layers 
//(see 2017 PNAS paper "Growth patterns for shape-shifting elastic bilayers")
void Sim_InverseBilayer::inverseTestExact(const Real deformRad)
{
    tag = "inverseTestExact";

    // initialize the mesh and define target surface
    initInverseProblem();

    const Real E = 1;
    const Real nu = parser.parse<Real>("-nu", 0.33);

    const int nFaces = mesh.getNumberOfFaces();

    Eigen::VectorXd GrowthFacs_top_1;
    GrowthFacs_top_1.resize(nFaces);

    Eigen::VectorXd GrowthFacs_top_2;
    GrowthFacs_top_2.resize(nFaces);

    Eigen::MatrixXd GrowthDirs_top;
    GrowthDirs_top.resize(nFaces, 3);

    Eigen::VectorXd GrowthFacs_bot_1;
    GrowthFacs_bot_1.resize(nFaces);

    Eigen::VectorXd GrowthFacs_bot_2;
    GrowthFacs_bot_2.resize(nFaces);

    Eigen::MatrixXd GrowthDirs_bot;
    GrowthDirs_bot.resize(nFaces, 3);

    const Real h_total = parser.parse<Real>("-h_total", 0.00302);
    const Real h_bot = h_total;
    const Real h_top = h_total;

    const std::string growth_type = parser.parse<std::string>("-growth_type","");

    MaterialProperties_Iso_Constant matprop_top(E, nu, h_top);
    MaterialProperties_Iso_Constant matprop_bot(E, nu, h_bot);

    CombinedOperator_Parametric_InverseGrowth_Bilayer<tMesh, Material_Isotropic> engOp_inverse(matprop_bot, matprop_top);

    const bool flipLayers = parser.parse<bool>("-fliplayers", false);
    std::vector<GrowthState> growth_bot, growth_top;

    const bool just_curvature = parser.parse<bool>("-just_curvature", false);
    computeGrowthForCurrentState(matprop_bot, matprop_top, growth_bot, growth_top, true, flipLayers, just_curvature);



    for(int i=0;i<nFaces;++i)
    {
          const DecomposedGrowthState & decomp_bot = growth_bot[i].getDecomposedFinalState();
          const DecomposedGrowthState & decomp_top = growth_top[i].getDecomposedFinalState();

          // exract the growth factors
          const Real s1_bot = decomp_bot.get_s1();
          const Real s2_bot = decomp_bot.get_s2();

          const Eigen::Vector3d v1_bot = decomp_bot.get_v1();

          const Real s1_top = decomp_top.get_s1();
          const Real s2_top = decomp_top.get_s2();

          const Eigen::Vector3d v1_top = decomp_top.get_v1();

          GrowthFacs_bot_1(i) = s1_bot - 1;
          GrowthFacs_top_1(i) = s1_top - 1;

          GrowthFacs_bot_2(i) = s2_bot - 1;
          GrowthFacs_top_2(i) = s2_top - 1;

          for(int j=0;j<3;++j){
            GrowthDirs_bot(i,j) = v1_bot(j);
            GrowthDirs_top(i,j) = v1_top(j);
          }

    }



    mesh.resetToRestState();

    dumpOrthoNew(growth_bot, growth_top, tag+"_inverse", true);
    // run forward problem
    const bool interpolate_logeucl = parser.parse<bool>("-interp_logeucl", false);

    // define operators
    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // dump 0 swelling rate (initial condition) for nicer movies afterwards
    dumpOrthoNew(growth_bot, growth_top, tag+"_final_"+helpers::ToString(0,3));
    assignGrowthToMetric(growth_bot, growth_top, 0.0, interpolate_logeucl);

    {
      const Real energy_inverse_1 = engOp_inverse.compute(mesh);

      printf("ENERGY_Inverse = %10.10e \n", energy_inverse_1);
    }

    const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
    const Real swelling_step = 1.0/((Real)nSwellingRuns);

    const int startidx = 0;

    for(int s=startidx;s<nSwellingRuns;++s)
    {
        const std::string curTag = tag+"_final_"+helpers::ToString(s+1,3);// s+1 since we already dumped s=0 as the initial condition

        const Real swelling_fac = (s+1)*swelling_step;

        // apply swelling
        assignGrowthToMetric(growth_bot, growth_top, swelling_fac, interpolate_logeucl);

        printf("%d \t %10.10e \t %10.10e \t %10.10e \t %10.10e\n", s, swelling_fac, engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
        // minimize energy
        Real eps = 1e-2;
        minimizeEnergy(engOps, eps);
    }

    dumpOrtho_Princdircurv(growth_bot, growth_top, tag+"_final");
    mesh.writeToFile(tag+"_final");
}





// Resolution of the inverse problem through minimization of the elastic energy functional. 
// Operates with local isotropic in-plane growth. Works well if the target shape is the exact state of equilibrium.
void Sim_InverseBilayer::inverseTestOptim(const Real deformRad)
{
    tag = "inverseTestOptim";

    // initialize the mesh and define target surface
    initInverseProblem();

    // create the energy operators
    const Real E = 1;
    const Real nu = parser.parse<Real>("-nu", 0.33);
    const Real h_total = parser.parse<Real>("-h_total", 0.00302);
    const Real h_bot = h_total;
    const Real h_top = h_total;

    MaterialProperties_Iso_Constant matprop_bot(E, nu, h_bot);
    MaterialProperties_Iso_Constant matprop_top(E, nu, h_top);

    CombinedOperator_Parametric_InverseGrowth_Bilayer<tMesh, Material_Isotropic> engOp_inverse(matprop_bot, matprop_top);

    Real dirichletEnergyFac = parser.parse<Real>("-smoothfac", 0.0);
    DirichletEnergyOperator_Inverse<tMesh, bottom> dirichletEnergyOp_bot(dirichletEnergyFac);
    DirichletEnergyOperator_Inverse<tMesh, top> dirichletEnergyOp_top(dirichletEnergyFac);
    EnergyOperatorList<tMesh> engOp_list({&engOp_inverse, &dirichletEnergyOp_bot, &dirichletEnergyOp_top});

    // compute the exact solution
    const bool flipLayers = parser.parse<bool>("-fliplayers", false);
    std::vector<GrowthState> growth_bot_exact, growth_top_exact;
    computeGrowthForCurrentState(matprop_bot, matprop_top, growth_bot_exact, growth_top_exact, true, flipLayers);
    dumpOrthoNew(growth_bot_exact, growth_top_exact, tag+"_exact_input", true);

    // create an initial guess : average of the two orthotropic growth factors for each layer unless specified otherwise

    const Real constant_init_guess_bot = parser.parse<Real>("-init_guess_bot", -1);
    const Real constant_init_guess_top = parser.parse<Real>("-init_guess_top", -1);

    const int nFaces = mesh.getNumberOfFaces();

    Eigen::VectorXd GrowthFacs_top(nFaces);
    GrowthFacs_top.setZero();
    Eigen::VectorXd GrowthFacs_bot(nFaces);
    GrowthFacs_bot.setZero();

    Eigen::VectorXd initGrowthFacs;
    {
        const int nFaces = mesh.getNumberOfFaces();
        initGrowthFacs.resize(2*nFaces);
        for(int i=0;i<nFaces;++i)
        {
            const DecomposedGrowthState & decomp_bot = growth_bot_exact[i].getDecomposedFinalState();
            const DecomposedGrowthState & decomp_top = growth_top_exact[i].getDecomposedFinalState();

            // exract the growth factors
            const Real s1_bot = decomp_bot.get_s1();
            const Real s2_bot = decomp_bot.get_s2();

            const Real s1_top = decomp_top.get_s1();
            const Real s2_top = decomp_top.get_s2();

            initGrowthFacs(i) = (constant_init_guess_bot > -1 ? constant_init_guess_bot : 0.5*(s1_bot + s2_bot) - 1);
            initGrowthFacs(i + nFaces) = (constant_init_guess_top > -1 ? constant_init_guess_top : 0.5*(s1_top + s2_top) - 1);

            GrowthFacs_bot(i) = initGrowthFacs(i);
            GrowthFacs_top(i) = initGrowthFacs(i + nFaces);
        }
    }



////////////
    GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_bot, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
    GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, GrowthFacs_top, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());


    Real energy_inverse = engOp_inverse.compute(mesh);

    printf("\n\n ENERGY_INVERSE = %10.10e \n\n", energy_inverse);

    initInverseProblem();
//////////////////


    Inverse_Machinery_Wrapper * inverse_wrapper = nullptr;

    const bool growthBound = parser.parse<bool>("-growthbound", false);

    if(growthBound)
    {
        const Real minGrowth = parser.parse<Real>("-mingrowth", -1.0);
        const Real maxGrowth = parser.parse<Real>("-maxgrowth", +5.0);
        // make sure our initial guess respects the bounds
        for(int i=0;i<initGrowthFacs.rows();++i)
            initGrowthFacs(i) = std::min(maxGrowth, std::max(minGrowth, initGrowthFacs(i)));

        inverse_wrapper = new Inverse_Bilayer_Iso_Wrapper_Bound(mesh, engOp_list, initGrowthFacs, minGrowth, maxGrowth);
    }
    else
    {
        inverse_wrapper = new Inverse_Bilayer_Iso_Wrapper(mesh, engOp_list, initGrowthFacs);
    }

    assert(inverse_wrapper != nullptr);

    // minimize
    int retval = 0;
    Real eps;
    const Real epsMin = parser.parse<Real>("-epsmin",std::numeric_limits<Real>::epsilon());

    {
        retval = inverse_wrapper->minimize(tag+"_hlbfgs_diagnostics.dat", epsMin);
        eps = inverse_wrapper->get_lastnorm();
        {
            const Real dirichletEnergy_bot = dirichletEnergyOp_bot.compute(mesh);
            const Real dirichletEnergy_top = dirichletEnergyOp_top.compute(mesh);
            const Real finalEnergy = engOp_inverse.compute(mesh);
            FILE * f = fopen((tag+"_convergence.dat").c_str(), "aw");
            fprintf(f, "%d \t %10.10e \t %10.10e \t %10.10e \t %10.10e \t %10.10e\n",retval, eps, dirichletEnergyFac, dirichletEnergy_bot, dirichletEnergy_top, finalEnergy);
            fclose(f);
        }
    }

    // compute growth state from solution
    mesh.resetToRestState();
    std::vector<GrowthState> growth_bot, growth_top;
    {
        //const int nFaces = mesh.getNumberOfFaces();
        growth_bot.reserve(nFaces);
        growth_top.reserve(nFaces);

        tVecMat2d aforms_bot, aforms_top;
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, inverse_wrapper->getGrowthRates_bot(), aforms_bot);
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, inverse_wrapper->getGrowthRates_top(), aforms_top);

        for(int i=0;i<nFaces;++i)
        {
            const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
            Eigen::MatrixXd rxy_base(3,2);
            rxy_base << rinfo.e1, rinfo.e2;
            growth_bot.emplace_back(rxy_base, aforms_bot[i]);
            growth_top.emplace_back(rxy_base, aforms_top[i]);
        }
    }
    dumpOrthoNew(growth_bot, growth_top, tag+"_inverse", true);

    // run forward problem
    const bool interpolate_logeucl = parser.parse<bool>("-interp_logeucl", false);

    // define operators
    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // dump 0 swelling rate (initial condition) for nicer movies afterwards
    dumpOrthoNew(growth_bot, growth_top, tag+"_final_"+helpers::ToString(0,3));
    assignGrowthToMetric(growth_bot, growth_top, 0.0, interpolate_logeucl);

    const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
    const Real swelling_step = 1.0/((Real)nSwellingRuns);

    const int startidx = 0;//(doRestart ? restartstep - 1 : 0);

    for(int s=startidx;s<nSwellingRuns;++s)
    {
        const std::string curTag = tag+"_final_"+helpers::ToString(s+1,3);// s+1 since we already dumped s=0 as the initial condition

        const Real swelling_fac = (s+1)*swelling_step;//inverseSwelling ? (nSwellingRuns - s - 1)*swelling_step : (s+1)*swelling_step;//swelling_facs[s];

        // apply swelling
        assignGrowthToMetric(growth_bot, growth_top, swelling_fac, interpolate_logeucl);

        printf("%d \t %10.10e \t %10.10e \t %10.10e \t %10.10e\n", s, swelling_fac, engOp_bot.compute(mesh), engOp_top.compute(mesh), engOps.compute(mesh));
        // minimize energy
        Real eps = 1e-2;
        minimizeEnergy(engOps, eps);

        // dump
        dumpOrthoNew(growth_bot, growth_top, curTag);
    }

    delete inverse_wrapper;
}






void Sim_InverseBilayer::Grouping(const int nFaces, const int nClusters, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters, const Eigen::Ref<const Eigen::MatrixXd> growthfactors, const Eigen::Ref<const Eigen::MatrixXi> adj_faces, int t)
{
  Eigen::MatrixXd GrowthFacs_ungrouped(nFaces,2);
  GrowthFacs_ungrouped.setZero();

  const int growthfactors_size = std::pow((nClusters+1),2);

  for(int i=0;i<nFaces;++i)
  {

    GrowthFacs_bot(i) = GrowthFacs_bot(i);
    GrowthFacs_top(i) = GrowthFacs_top(i);

    GrowthFacs_ungrouped(i,0) = GrowthFacs_bot(i);
    GrowthFacs_ungrouped(i,1) = GrowthFacs_top(i);

    Eigen::VectorXd Distances (growthfactors_size);
    Distances.setZero();

    for(int j=0;j<growthfactors_size;++j)
    {
      Distances(j) = std::pow(GrowthFacs_bot(i)-growthfactors(j,0),2) + std::pow(GrowthFacs_top(i)-growthfactors(j,1),2);
    }

    const Real distance = Distances.minCoeff();
    for(int j=0;j<growthfactors_size;++j)
    {
      if(distance == Distances(j))
      {
        GrowthFacs_bot(i) = growthfactors(j,0);
        GrowthFacs_top(i) = growthfactors(j,1);
        Clusters(i) = j;
      }
    }
  }

  Eigen::MatrixXd GrowthFacs_grouped(nFaces,2);
  GrowthFacs_grouped.setZero();
  for(int i=0;i<nFaces;++i)
  {
    GrowthFacs_grouped(i,0) = GrowthFacs_bot(i);
    GrowthFacs_grouped(i,1) = GrowthFacs_top(i);
  }

  helpers::write_matrix( tag + "_growthFacs_UNgrouped.txt", GrowthFacs_ungrouped);
  helpers::write_matrix( tag + "_growthFacs_grouped_" + helpers::ToString(t,2) + ".txt", GrowthFacs_grouped);
  helpers::write_matrix_binary( tag + "_growthFacs_grouped_" + helpers::ToString(t,2) + ".dat", GrowthFacs_grouped);
  helpers::write_matrix( tag + "_groups_" + helpers::ToString(t,2) + ".txt", Clusters);
  helpers::write_matrix_binary(tag + "_groups_" + helpers::ToString(t,2) + ".dat", Clusters);
}






void Sim_InverseBilayer::Kmeans(const int nTries_kmeans, const int nClusters, const int nFaces, const Real h_total, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters)
{

  Eigen::MatrixXi Clusters_kmeans(nFaces,2);
  Eigen::VectorXd GrowthFacs_unclustered_kmeans_1(nFaces);
  Eigen::VectorXd GrowthFacs_unclustered_kmeans_2(nFaces);
  Eigen::VectorXi Invert_faces(nFaces);
  Eigen::MatrixXd Cluster_weight(nClusters+1,2);
  Eigen::MatrixXd Prev_centers(nClusters+1,2);
  Eigen::MatrixXd Raw_centers(nClusters+1,2);

  const int nCenters = ((nClusters+1)*nClusters)/2 + (nClusters+1); //Half of square matrix + main diagonal
  Eigen::MatrixXd Centers(nCenters,2);
  Eigen::VectorXi Cluster_count(nCenters); // actually need nClusters elments, but others are for the purpose of debugging
  Eigen::VectorXd Distances(nCenters);
  Eigen::MatrixXi Combins(nCenters,2); //Combinations of different regimes. The size is chosen for convenience, actually the size must be ((nClusters-1)*nClusters)/2

  Eigen::MatrixXf::Index min_ind;

  Real distance = 0;

  Eigen::VectorXd Error_tries(nTries_kmeans);
  Error_tries.setZero();
  Eigen::MatrixXd Raw_centers_global(nClusters, nTries_kmeans);
  Raw_centers_global.setZero();
  int minerr_indic = 0;
  int trycount = 0;
  int cc;

  // We launch the kmeans algorithm several times and choose the best partition
  while (minerr_indic != 2){

    Clusters_kmeans.setZero();
    GrowthFacs_unclustered_kmeans_1.setZero();
    GrowthFacs_unclustered_kmeans_2.setZero();
    Invert_faces.setZero();
    Distances.setZero();
    Clusters.setZero();
    Cluster_count.setZero();
    Cluster_weight.setZero();
    Prev_centers.setZero();
    Raw_centers.setZero();
    Centers.setZero();
    Combins.setZero();
    cc = 0;


    for(int i=0;i<nFaces;++i)
    {
      if (GrowthFacs_bot(i) > GrowthFacs_top(i)){
        Invert_faces(i) = 0;
        GrowthFacs_unclustered_kmeans_1(i) = GrowthFacs_bot(i);
        GrowthFacs_unclustered_kmeans_2(i) = GrowthFacs_top(i);
      } else {
        Invert_faces(i) = 1;
        GrowthFacs_unclustered_kmeans_1(i) = GrowthFacs_top(i);
        GrowthFacs_unclustered_kmeans_2(i) = GrowthFacs_bot(i);
      }
    }


    //STRUCTURE OF THE CENTROIDS MATRIX
    // zero cluster (one unit)
    // raw clusters (nClusters units)
    // pairs_same (nClusters units)

    // Combinserent ((nClusters-1)*nClusters)/2 units):
    // 2, 1
    // 3, 1
    // 3, 2
    // 4, 1
    // ...
    // n, n-1

    Raw_centers(0,0) = 0.0;
    Raw_centers(0,1) = 0.0;

    // initial guess
    if (minerr_indic != 1){
      for (int j=1; j < nClusters+1; ++j){
        Raw_centers(j,0) = GrowthFacs_unclustered_kmeans_1.maxCoeff()/3.0 + (GrowthFacs_unclustered_kmeans_1.maxCoeff()/3.0*2.0)/(nClusters*20)*(std::rand() % (nClusters*20) + 1);
        Raw_centers(j,1) = Raw_centers(j,0)*(-0.2);
        Raw_centers_global(j,trycount) = Raw_centers(j,0);

        for (int jj=1; jj < j; ++jj){
          if(Raw_centers(j,0) == Raw_centers(jj,0)){
            Raw_centers(j,0) *= 1.1;
          }
        }
      }
    } else {
      for (int j=1; j < nClusters+1; ++j){
        Raw_centers(j,0) = Raw_centers_global(j,min_ind);
        Raw_centers(j,1) = Raw_centers(j,0)*(-0.2);
      }
    }

    for (int i=0; i<nClusters+1; ++i){
      for (int j=0; j<nClusters+1; ++j){
        if (Raw_centers(i,0) >= Raw_centers(j,0)){
          Centers(cc,0) = Raw_centers(i,0) + Raw_centers(j,1);
          Centers(cc,1) = Raw_centers(i,1) + Raw_centers(j,0);
          Combins(cc,0) = i;
          Combins(cc,1) = j;
          cc++;
        }
      }
    }

    // iterative relocation of the centroids
    int grouping_indicator = 0;
    int iter_counter = 0;
    int dd = 0;
    while (grouping_indicator != 1){

      for (int j=1; j<nClusters+1; j++){
        Prev_centers(j,0) = Centers(j,0);
        Prev_centers(j,1) = Centers(j,1);
      }

      Cluster_weight.setZero();
      Cluster_count.setZero();

      // assign points to clusters
      for(int i=0;i<nFaces;++i)
      {
        for (int j=0; j<nCenters; j++){
          Distances(j) = std::pow(GrowthFacs_unclustered_kmeans_1(i)-Centers(j,0),2) + std::pow(GrowthFacs_unclustered_kmeans_2(i)-Centers(j,1),2);
        }
        distance = Distances.minCoeff();
        for (int j=0; j<nCenters; j++){
          if(distance == Distances(j))
          {
            Clusters(i) = j;
          }
        }
      }


      Eigen::VectorXd addweight_1(2);
      Eigen::VectorXd addweight_2(2);

      // compute the center of mass of all clusters
      for(int i=0;i<nFaces;++i)
      {
        for (int j=0; j<nCenters; j++){
            if(Clusters(i) == j){

              addweight_1(0) = 0.5*(Centers(Combins(j,0),0) - Centers(Combins(j,1),1) + GrowthFacs_unclustered_kmeans_1(i));
              addweight_1(1) = 0.5*(Centers(Combins(j,0),1) - Centers(Combins(j,1),0) + GrowthFacs_unclustered_kmeans_2(i));

              addweight_2(0) = 0.5*(Centers(Combins(j,1),0) - Centers(Combins(j,0),1) + GrowthFacs_unclustered_kmeans_2(i));
              addweight_2(1) = 0.5*(Centers(Combins(j,1),1) - Centers(Combins(j,0),0) + GrowthFacs_unclustered_kmeans_1(i));

              Cluster_weight(Combins(j,0),0) = Cluster_weight(Combins(j,0),0) + addweight_1(0);
              Cluster_weight(Combins(j,0),1) = Cluster_weight(Combins(j,0),1) + addweight_1(1);

              Cluster_weight(Combins(j,1),0) = Cluster_weight(Combins(j,1),0) + addweight_2(0);
              Cluster_weight(Combins(j,1),1) = Cluster_weight(Combins(j,1),1) + addweight_2(1);

              Cluster_count(Combins(j,0)) = Cluster_count(Combins(j,0)) + 1;
              Cluster_count(Combins(j,1)) = Cluster_count(Combins(j,1)) + 1;

              //Cluster_count(j)++; //for debugging
            }
         }
      }

      // relocate the centroids
      for (int j=1; j<nClusters+1; j++){ // we are not touching the zero cluster
        if(Cluster_count(j) != 0){
          Centers(j,0) = Cluster_weight(j,0)/Cluster_count(j);
          Centers(j,1) = Cluster_weight(j,1)/Cluster_count(j);
        }
      }

      for (int j=nClusters+1; j<nCenters; j++){
        Centers(j,0) = Centers(Combins(j,0),0) + Centers(Combins(j,1),1);
        Centers(j,1) = Centers(Combins(j,1),0) + Centers(Combins(j,0),1);
      }

      // stop conditions
      grouping_indicator = 1;
      for (int j=1; j<nClusters+1; j++){
        if(Centers(j,0) != Prev_centers(j,0) || Centers(j,1) != Prev_centers(j,1)){
          grouping_indicator = 0;
        }
      }

      // prevent infinite looping
      dd++;
      if(dd == 1000){
        grouping_indicator = 1;
      }
    }

    // choose the best partition
    for(int i=0;i<nFaces;++i)
    {
      Error_tries(trycount) = Error_tries(trycount) + std::pow(GrowthFacs_unclustered_kmeans_1(i)-Centers(Clusters(i),0),2) + std::pow(GrowthFacs_unclustered_kmeans_2(i)-Centers(Clusters(i),1),2);
    }

    if (minerr_indic == 1){
      minerr_indic = 2;
    }

    if (trycount == nTries_kmeans-1){
      Real min_dist = Error_tries.minCoeff(&min_ind);
      //std::cout<< "\n\n" << Error_tries << "\n\n" << std::endl;
      std::cout<< "\n k-means instance providing the minimal error = " << min_ind << "\n\n" << std::endl;
      minerr_indic = 1;
    }

    trycount ++;
  }



  //homogenize the growthfactors inside the segments
  for(int i=0;i<nFaces;++i)
  {
    for (int j=0; j<nCenters; j++)
    {
      if(Clusters(i) == j)
      {
        if (Invert_faces(i) == 0){
          GrowthFacs_bot(i) = Centers(j,0);
          GrowthFacs_top(i) = Centers(j,1);
          Clusters_kmeans(i,0) = Combins(j,0);
          Clusters_kmeans(i,1) = Combins(j,1);
        } else {
          GrowthFacs_top(i) = Centers(j,0);
          GrowthFacs_bot(i) = Centers(j,1);
          Clusters_kmeans(i,0) = Combins(j,1);
          Clusters_kmeans(i,1) = Combins(j,0);
        }
      }
    }
  }


  //compute all possible combinations of regimes, not only the ordered ones as in Centers
  const int growthfactors_size = std::pow((nClusters+1),2);
  Eigen::MatrixXd Centers_topbot (growthfactors_size,2); // 2*(4+6)+4+1 = (one side+different)*same_thing_but_inversed + same_from_both_sides + zero
  Centers_topbot.setZero();
  for (int i=0 ; i<nClusters+1 ; ++i){
    for (int j=0 ; j<nClusters+1 ; ++j){
      Centers_topbot(i*(nClusters+1)+j,0) = Centers(i,0) + Centers(j,1);
      Centers_topbot(i*(nClusters+1)+j,1) = Centers(j,0) + Centers(i,1);
    }
  }
  //then reassign cluster numbers, so that (n,m) and (m,n) is now different clusters
  for(int i=0;i<nFaces;++i)
  {
    for (int j=0; j<std::pow((nClusters+1),2); j++)
    {
      if(GrowthFacs_bot(i) == Centers_topbot(j,0) && GrowthFacs_top(i) == Centers_topbot(j,1))
      {
        Clusters(i) == j;
      }
    }
  }


  // compute the growth factors in terms of the "step" eigenstrain profile and dump everything
  Real epseq;
  Real heq;
  Eigen::MatrixXd h_eps(nClusters,2);
  for (int j=1; j < nClusters+1; ++j){
    Bilayer_2_Trilayer(&epseq, &heq, h_total, Raw_centers(j,0), Raw_centers(j,1));
    h_eps(j-1,0) = heq;
    h_eps(j-1,1) = epseq;
  }
  helpers::write_matrix("Cluster_center_kmeans_trilayer.txt", h_eps);
  helpers::write_matrix("GrowthFacs_unclustered_kmeans_1.txt", GrowthFacs_unclustered_kmeans_1);
  helpers::write_matrix("GrowthFacs_unclustered_kmeans_2.txt", GrowthFacs_unclustered_kmeans_2);

  //std::cout<< "\n\n" << "nClusters = "<< nClusters << "\n\n\n" << std::endl;
  //std::cout<< "\n\n" << "nCenters = "<< nCenters << "\n\n\n" << std::endl;
  //std::cout<< "\n\n" << "Raw_centers"<< "\n" << Raw_centers << "\n\n\n" << std::endl;
  //std::cout<< "\n\n" << "Centers"<< "\n" << Centers << "\n\n\n" << std::endl;

  helpers::write_matrix("Cluster_centers_kmeans.txt", Centers);
  helpers::write_matrix("Clusters_kmeans.txt", Clusters_kmeans);
  helpers::write_matrix("Clusters.txt", Clusters);
  helpers::write_matrix("Invert_faces.txt", Invert_faces);
  helpers::write_matrix("Cluster_count.txt", Cluster_count);

}




void Sim_InverseBilayer::Filtering(const int nFaces, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters, const Eigen::Ref<const Eigen::MatrixXi> adj_faces, int t)
{

    int clean_indic = 0; // if it equas 1 then we stop the iterations
    const int mincluster = Clusters.minCoeff();
    const int maxcluster = Clusters.maxCoeff();

    Eigen::VectorXd GrowthFacs_bot_temp(3);
    Eigen::VectorXd GrowthFacs_top_temp(3);
    int reassignments;
    int neighbors;
    
    //iterative filtering until the stabilization of the pattern
    while (clean_indic == 0){
      
      reassignments = 0;
      
      for (int j=mincluster;j<maxcluster+1;++j){
        for(int i=0;i<nFaces;++i){
          if(Clusters(i)==j){

            const Real growth_bot_old = GrowthFacs_bot(i);
            const Real growth_top_old = GrowthFacs_top(i);

            GrowthFacs_bot_temp.setZero();
            GrowthFacs_top_temp.setZero();

            neighbors = 0;
            for (int j=0;j<3;++j){
              if (adj_faces(i,j)>=0){
                GrowthFacs_bot_temp(neighbors) = GrowthFacs_bot(adj_faces(i,j));
                GrowthFacs_top_temp(neighbors) = GrowthFacs_top(adj_faces(i,j));
                neighbors++;
              } 
            }
            if (neighbors < 3 && GrowthFacs_bot_temp(1) != GrowthFacs_bot(i)){
              GrowthFacs_bot_temp(2) = GrowthFacs_bot_temp(0);
              GrowthFacs_top_temp(2) = GrowthFacs_top_temp(0);
            }


            if (GrowthFacs_bot_temp(0)!=GrowthFacs_bot(i) && (GrowthFacs_bot_temp(0)==GrowthFacs_bot_temp(1) || GrowthFacs_bot_temp(0)==GrowthFacs_bot_temp(2))){
              GrowthFacs_bot(i) = GrowthFacs_bot_temp(0);
              GrowthFacs_top(i) = GrowthFacs_top_temp(0);
              Clusters(i) = Clusters(adj_faces(i,0));
            }
            else if (GrowthFacs_bot_temp(1)!=GrowthFacs_bot(i) && GrowthFacs_bot_temp(1)==GrowthFacs_bot_temp(2)){
              GrowthFacs_bot(i) = GrowthFacs_bot_temp(1);
              GrowthFacs_top(i) = GrowthFacs_top_temp(1);
              Clusters(i) = Clusters(adj_faces(i,1));
            }
            else if ((GrowthFacs_bot_temp(0)!=GrowthFacs_bot(i) && GrowthFacs_bot_temp(1)!=GrowthFacs_bot(i) && GrowthFacs_bot_temp(2)!=GrowthFacs_bot(i)) && 
                (GrowthFacs_bot_temp(0)!=GrowthFacs_bot_temp(1) && GrowthFacs_bot_temp(0)!=GrowthFacs_bot_temp(2) && GrowthFacs_bot_temp(1)!=GrowthFacs_bot_temp(2))){
              GrowthFacs_bot(i) = GrowthFacs_bot_temp(0);
              GrowthFacs_top(i) = GrowthFacs_top_temp(0);
              Clusters(i) = Clusters(adj_faces(i,0));
            }

            if (GrowthFacs_bot(i)!=growth_bot_old || GrowthFacs_top(i)!=growth_top_old){
              reassignments++;
            }         
          }
        }
      }

      if (reassignments == 0){
       clean_indic = 1;
      }
    }

    Eigen::MatrixXd GrowthFacs_clustered(nFaces, 2);
    for(int i=0;i<nFaces;++i)
    {
      GrowthFacs_clustered(i,0) = GrowthFacs_bot(i);
      GrowthFacs_clustered(i,1) = GrowthFacs_top(i);
    }

    helpers::write_matrix( tag + "_growthFacs_clustered_" + helpers::ToString(t,2) + ".txt", GrowthFacs_clustered);
    helpers::write_matrix_binary( tag + "_growthFacs_clustered_" + helpers::ToString(t,2) + ".dat", GrowthFacs_clustered);
    helpers::write_matrix( tag + "_clusters_" + helpers::ToString(t,2) + ".txt", Clusters);
    helpers::write_matrix_binary(tag + "_clusters_" + helpers::ToString(t,2) + ".dat", Clusters);
}




// measure the Huasdorff distance between two shapes and return the maximal distance and the sum of square Distances
void Sim_InverseBilayer::ErrorMap(const std::string fname1, const std::string fname2, const Real k, Real *maxdist, Real* relerror, Real*sqsumerror)
{

  const bool flipX = parser.parse<bool>("-flipx", false);
  const bool flipY = parser.parse<bool>("-flipy", false);
  const bool flipZ = parser.parse<bool>("-flipz", false);
  const bool flipXY = parser.parse<bool>("-flipXY", false);
  const bool flipXminusY = parser.parse<bool>("-flipXminusY", false);

  IOGeometry geometry1(fname1);
  IOGeometry geometry2(fname2);

  Eigen::MatrixXd vertices1, vertices2;
  Eigen::MatrixXi faces1, faces2;
  Eigen::MatrixXb vertices_bc_dummy;

  geometry1.get(vertices1, faces1, vertices_bc_dummy);
  geometry2.get(vertices2, faces2, vertices_bc_dummy);

  const int nFaces1 = faces1.rows();
  const int nFaces2 = faces2.rows();
  const int nVertices1 = vertices1.rows();
  const int nVertices2 = vertices2.rows();
  if((nVertices1 != nVertices2) or (nFaces1 != nFaces2))
  {
      printf("Number of vertices and faces should match : v1, f1 : %d , %d \t v2, f2 : %d, %d  -- returning\n",nVertices1,nFaces1,nVertices2,nFaces2);
  }

  const Real faces_err = (faces1 - faces2).norm();
  if(faces_err > 1e-12)
  {
      printf("Face arrays should be the same - returning\n");
  }

  if(flipX or flipY or flipZ)
  {
      // do it on mesh 1 (does not matter)
      for(int i=0;i<nVertices1;++i)
      {
          if(flipX) vertices1(i,0) *= -1;
          if(flipY) vertices1(i,1) *= -1;
          if(flipZ) vertices1(i,2) *= -1;
      }

  }

  if(flipXY)
  {
      // do it on mesh 1 (does not matter)
      for(int i=0;i<nVertices1;++i)
      {
          Real temp = vertices1(i,0);
          vertices1(i,0) = vertices1(i,1);
          vertices1(i,1) = temp;
      }
  }

  if(flipXminusY)
  {
      // do it on mesh 1 (does not matter)
      for(int i=0;i<nVertices1;++i)
      {
          Real temp = -vertices1(i,0);
          vertices1(i,0) = -vertices1(i,1);
          vertices1(i,1) = temp;
      }
  }

  const bool bDump = parser.parse<bool>("-dump", true);
  const bool bRescale = parser.parse<bool>("-rescale", false);
  ComputeErrorMap computeErrorMap(bDump);
  const Eigen::VectorXd hdist = computeErrorMap.compute(vertices1, vertices2, faces1, bRescale);

  const auto cvertices = mesh.getRestConfiguration().getVertices();
  const auto cface2vertices = mesh.getTopology().getFace2Vertices();
  WriteVTK writer(cvertices, cface2vertices);
  writer.addScalarFieldToVertices(hdist, "error_map");
  writer.write(tag+"_error_map_" + helpers::ToString(k,2));


  *maxdist = hdist.maxCoeff();

  *sqsumerror = 0.0;
  for(int i=0;i<nVertices1;++i)
  {
    *sqsumerror += std::pow(hdist(i),2);
  }

  Eigen::VectorXd deflection(nVertices2);
  for(int i=0;i<nVertices2;++i){
    deflection(i) = vertices2(i,2);
  }
  const Real maxZ = deflection.maxCoeff();
  const Real minZ = deflection.minCoeff();

  *relerror = *maxdist/(maxZ-minZ);
}




void Sim_InverseBilayer::MarginCut(const Real margin_width, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters)
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
    if (Vertices(i,0)<=(MinVert(0)+margin_width) || Vertices(i,0)>=(MaxVert(0)-margin_width) || Vertices(i,1)<=(MinVert(1)+margin_width) || Vertices(i,1)>=(MaxVert(1)-margin_width)){
      IndicV(i) = 1;
    }
  }

  for (int i=0; i<nFaces; ++i){
     if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
       GrowthFacs_bot(i) = 0.0;
       GrowthFacs_top(i) = 0.0;
       Clusters(i) = 0;
     }
  }
}




void Sim_InverseBilayer::GrowthAdjust(const int nFaces, const Real maxratio, Eigen::Ref<Eigen::VectorXd> area_t, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, const Eigen::Ref<const Eigen::VectorXd> mean_t, const Eigen::Ref<const Eigen::VectorXd> princcurv1_t, const int m, const int t, const bool just_curvature)
{

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

  Eigen::VectorXd ratio_curv(nFaces);
  for(int i=0;i<nFaces;++i)
  {
    ratio_curv(i) = mean_t(i)/mean_c(i);
    ratio_curv(i) = (ratio_curv(i) > 0.0 ? ratio_curv(i) : 1.0);
    ratio_curv(i) = (std::abs(ratio_curv(i)) < maxratio ? (std::abs(ratio_curv(i)) > 1.0/maxratio ? ratio_curv(i) : 1.0/maxratio*(std::abs(ratio_curv(i))/ratio_curv(i))) : maxratio*(std::abs(ratio_curv(i))/ratio_curv(i)));
  }

  tVecMat2d firstFF_c,secondFF_c;
  computeQuadraticForms(firstFF_c, secondFF_c);
  Eigen::VectorXd area_c(nFaces);
  Eigen::VectorXd ratio_area(nFaces);
  for(int i=0;i<nFaces;++i)
  {
    area_c(i) = 0.5*std::sqrt(firstFF_c[i].determinant());
    ratio_area(i) = area_t(i)/area_c(i);
  }


  for(int i=0;i<nFaces;++i)
  {
    if (just_curvature){
      Real difference = GrowthFacs_bot(i) - GrowthFacs_top(i);
      GrowthFacs_bot(i) += difference*(ratio_curv(i)-1.0)*0.5;
      GrowthFacs_top(i) -= difference*(ratio_curv(i)-1.0)*0.5;
    } else {
      Real difference = GrowthFacs_bot(i) - GrowthFacs_top(i);
      Real sum = GrowthFacs_bot(i) + GrowthFacs_top(i) + 1.0;
      difference *= ratio_curv(i)*ratio_area(i);
      sum *= ratio_area(i);
      sum -= 1.0;

      GrowthFacs_bot(i) = (sum+difference)*0.5;
      GrowthFacs_top(i) = (sum-difference)*0.5;
    }
  }


  const auto cvertices = mesh.getRestConfiguration().getVertices();
  const auto cface2vertices = mesh.getTopology().getFace2Vertices();
  WriteVTK writer(cvertices, cface2vertices);
  if (just_curvature == false){
    writer.addScalarFieldToFaces(area_c, "base_area");
    writer.addScalarFieldToFaces(area_t, "target_area");
    writer.addScalarFieldToFaces(ratio_area, "ratio_area");
  }
  writer.addScalarFieldToFaces(ratio_curv, "ratio_curv");

  if (m<0){
    writer.write(tag+"_areas_"+helpers::ToString(t,2));
  } else {
    writer.write(tag+"_areas_"+helpers::ToString(m,2)+"_"+helpers::ToString(t,2));
  }
}



void Sim_InverseBilayer::assignGrowthToMetric(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const Real t, const bool interp_logeucl)
{
    const int nFaces = mesh.getNumberOfFaces();
    tVecMat2d & firstFF_bot = mesh.getRestConfiguration().getFirstFundamentalForms<bottom>();
    tVecMat2d & firstFF_top = mesh.getRestConfiguration().getFirstFundamentalForms<top>();

    for(int i=0;i<nFaces;++i)
    {
        firstFF_bot[i] = (interp_logeucl ? growth_bot[i].interpolateLogEucl(t) : growth_bot[i].interpolate(t));
        firstFF_top[i] = (interp_logeucl ? growth_top[i].interpolateLogEucl(t) : growth_top[i].interpolate(t));
    }
}





void Sim_InverseBilayer::computeGrowthForCurrentState(const MaterialProperties_Iso_Constant & matprop_bot, const MaterialProperties_Iso_Constant & matprop_top, std::vector<GrowthState> & growth_bot, std::vector<GrowthState> & growth_top, const bool finalConfig, const bool flipLayers, const bool just_curvature)
{
    const std::string curTag = (finalConfig ? tag+"_final" : tag+"_init");

    // first we dump the mapping from the rest state onto the current configuration
    {
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::MatrixXd endVals(nFaces, 3);
        const auto face2vertices = mesh.getTopology().getFace2Vertices();
        for(int i=0;i<nFaces;++i)
        {
            const TriangleInfo info = mesh.getCurrentConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
            const Eigen::Vector3d face_ctr = info.computeFaceCenter();

            endVals.row(i) = face_ctr;
        }

        const auto cvertices = mesh.getRestConfiguration().getVertices();
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();
        WriteVTK writer(cvertices, cface2vertices);
        writer.addVectorFieldToFaces(endVals, "mapping");
        writer.write(curTag+"_mapping");
    }

    // compute the metric of the current configuration
    mesh.updateDeformedConfiguration();
    tVecMat2d firstFF,secondFF;
    computeQuadraticForms(firstFF, secondFF);

    if (just_curvature){
      mesh.resetToRestState();
      tVecMat2d secondFF_rest;
      computeQuadraticForms(firstFF, secondFF_rest);
    }

    const int nFaces = mesh.getNumberOfFaces();
    Eigen::MatrixXd FFF(nFaces, 3);
    Eigen::MatrixXd SFF(nFaces, 3);
    for(int i=0;i<nFaces;++i)
    {
        FFF(i,0) = firstFF[i](0,0);
        FFF(i,1) = firstFF[i](0,1);
        FFF(i,2) = firstFF[i](1,1);

        SFF(i,0) = secondFF[i](0,0);
        SFF(i,1) = secondFF[i](0,1);
        SFF(i,2) = secondFF[i](1,1);
    }

    const auto cvertices = mesh.getRestConfiguration().getVertices();
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();
    WriteVTK writer(cvertices, cface2vertices);
    writer.addVectorFieldToFaces(FFF, "FFF");
    writer.addVectorFieldToFaces(SFF, "SFF");
    writer.write(curTag+"_fundforms");
    // define the growth states
    //const int nFaces = mesh.getNumberOfFaces();
    if(finalConfig)
    {
        growth_bot.clear();
        growth_top.clear();
        growth_bot.reserve(nFaces);
        growth_top.reserve(nFaces);
    }
    else
    {
        assert((int)growth_bot.size() == nFaces);
        assert((int)growth_top.size() == nFaces);
    }
    for(int i=0;i<nFaces;++i)
    {
        // compute the forms directly
        const Real thickness_bot = matprop_bot.getFaceMaterial(i).getThickness();
        const Real thickness_top = matprop_top.getFaceMaterial(i).getThickness();

	const Real a_prefac = (flipLayers ? -1 : +1) * (4.0*pow(thickness_bot,2)-        thickness_bot*thickness_top+pow(thickness_top,2))/(6.0*thickness_bot);

	const Real b_prefac = (flipLayers ? -1 : +1) * (4.0*pow(thickness_top,2)-        thickness_bot*thickness_top+pow(thickness_bot,2))/(6.0*thickness_top);

        const Eigen::Matrix2d growth_metric_bot = firstFF[i] + a_prefac*secondFF[i];
        const Eigen::Matrix2d growth_metric_top = firstFF[i] - b_prefac*secondFF[i];

        // get triangleinfo for rest config
        const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
        Eigen::MatrixXd rxy_base(3,2);
        rxy_base << rinfo.e1, rinfo.e2;

        if(finalConfig)
        {
            growth_bot.emplace_back(rxy_base, growth_metric_bot);
            growth_top.emplace_back(rxy_base, growth_metric_top);
        }
        else
        {
            growth_bot[i].changeInitialState(growth_metric_bot);
            growth_top[i].changeInitialState(growth_metric_top);
        }
    }

    dumpWithNormals(curTag+"_normcurv");
    dumpWithNormals(curTag+"_normcurv_rest", true);
    dumpOrthoNew(growth_bot, growth_top, curTag+"_growth");


    {
        // compute ratio of triangle sizes and dump
        Eigen::VectorXd base_area(nFaces);
        Eigen::VectorXd target_area(nFaces);
        for(int i=0;i<nFaces;++i)
        {
            const TriangleInfo rinfo = mesh.getRestConfiguration().getTriangleInfoLite(mesh.getTopology(), i);
            Eigen::MatrixXd rxy_base(3,2);
            rxy_base << rinfo.e1, rinfo.e2;

            base_area(i) = 0.5*std::sqrt((rxy_base.transpose() * rxy_base).determinant());
            target_area(i) = 0.5*std::sqrt(firstFF[i].determinant());

        }

        const auto cvertices = mesh.getRestConfiguration().getVertices();
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();
        WriteVTK writer(cvertices, cface2vertices);
        writer.addScalarFieldToFaces(base_area, "base_area");
        writer.addScalarFieldToFaces(target_area, "target_area");
        writer.write(curTag+"_areas");
    }
    // done
}





void Sim_InverseBilayer::dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig)
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




void Sim_InverseBilayer::dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig)
{
    const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();
    WriteVTK writer(cvertices, cface2vertices);

    writer.addScalarFieldToFaces(growthRates_1_bot, "rate1_bot");
    writer.addScalarFieldToFaces(growthRates_2_bot, "rate2_bot");
    writer.addScalarFieldToFaces(growthAngles, "dir_growth");
    writer.addScalarFieldToFaces(growthRates_1_top, "rate1_top");
    writer.addScalarFieldToFaces(growthRates_2_top, "rate2_top");

    writer.write(filename);
}




void Sim_InverseBilayer::dumpOrthoNew(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const std::string filename, const bool restConfig, const bool restGrowth)
{
    const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();
    WriteVTK writer(cvertices, cface2vertices);

    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXd difference_1(nFaces);
    Eigen::VectorXd growthFacs_1_bot(nFaces), growthFacs_2_bot(nFaces);
    Eigen::VectorXd growthFacs_1_top(nFaces), growthFacs_2_top(nFaces);
    Eigen::MatrixXd growthDirs_1_bot(nFaces, 3), growthDirs_2_bot(nFaces, 3);
    Eigen::MatrixXd growthDirs_1_top(nFaces, 3), growthDirs_2_top(nFaces, 3);
    for(int i=0;i<nFaces;++i)
    {
        const DecomposedGrowthState & decomposed_bot = restGrowth ? growth_bot[i].getDecomposedInitState() : growth_bot[i].getDecomposedFinalState();
        const DecomposedGrowthState & decomposed_top = restGrowth ? growth_top[i].getDecomposedInitState() : growth_top[i].getDecomposedFinalState();

        growthFacs_1_bot(i) = decomposed_bot.get_s1();
        growthFacs_2_bot(i) = decomposed_bot.get_s2();

        growthFacs_1_top(i) = decomposed_top.get_s1();
        growthFacs_2_top(i) = decomposed_top.get_s2();

        difference_1(i) = growthFacs_1_bot(i) - growthFacs_1_top(i);

        growthDirs_1_bot.row(i) = decomposed_bot.get_v1();
        growthDirs_2_bot.row(i) = decomposed_bot.get_v2();

        growthDirs_1_top.row(i) = decomposed_top.get_v1();
        growthDirs_2_top.row(i) = decomposed_top.get_v2();
    }

    writer.addScalarFieldToFaces(growthFacs_1_bot, "rate1_bot");
    writer.addScalarFieldToFaces(growthFacs_2_bot, "rate2_bot");
    writer.addVectorFieldToFaces(growthDirs_1_bot, "dir1_bot");
    writer.addVectorFieldToFaces(growthDirs_2_bot, "dir2_bot");
    writer.addScalarFieldToFaces(growthFacs_1_top, "rate1_top");
    writer.addScalarFieldToFaces(growthFacs_2_top, "rate2_top");
    writer.addVectorFieldToFaces(growthDirs_1_top, "dir1_top");
    writer.addVectorFieldToFaces(growthDirs_2_top, "dir2_top");
    writer.addScalarFieldToFaces(difference_1, "difference_1");

    helpers::write_matrix(filename + "_vertices.txt", cvertices);
    helpers::write_matrix(filename + "_face2vertices.txt", cface2vertices);

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

        helpers::write_matrix(filename + "_CurvX.txt", CurvX);
        helpers::write_matrix(filename + "_CurvY.txt", CurvY);
    }

    writer.write(filename);
}




void Sim_InverseBilayer::dumpOrtho_Princdircurv(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const std::string filename, const bool restConfig, const bool restGrowth)
{

    const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();
    WriteVTK writer(cvertices, cface2vertices);

    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXd difference_1(nFaces);
    Eigen::VectorXd growthFacs_1_bot(nFaces), growthFacs_2_bot(nFaces);
    Eigen::VectorXd growthFacs_1_top(nFaces), growthFacs_2_top(nFaces);
    Eigen::MatrixXd growthDirs_1_bot(nFaces, 3), growthDirs_2_bot(nFaces, 3);
    Eigen::MatrixXd growthDirs_1_top(nFaces, 3), growthDirs_2_top(nFaces, 3);
    for(int i=0;i<nFaces;++i)
    {
        const DecomposedGrowthState & decomposed_bot = restGrowth ? growth_bot[i].getDecomposedInitState() : growth_bot[i].getDecomposedFinalState();
        const DecomposedGrowthState & decomposed_top = restGrowth ? growth_top[i].getDecomposedInitState() : growth_top[i].getDecomposedFinalState();

        growthFacs_1_bot(i) = decomposed_bot.get_s1();
        growthFacs_2_bot(i) = decomposed_bot.get_s2();

        growthFacs_1_top(i) = decomposed_top.get_s1();
        growthFacs_2_top(i) = decomposed_top.get_s2();

        difference_1(i) = growthFacs_1_bot(i) - growthFacs_1_top(i);

        growthDirs_1_bot.row(i) = decomposed_bot.get_v1();
        growthDirs_2_bot.row(i) = decomposed_bot.get_v2();

        growthDirs_1_top.row(i) = decomposed_top.get_v1();
        growthDirs_2_top.row(i) = decomposed_top.get_v2();

    }

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
        Eigen::VectorXd Curv1(nFaces);
        Eigen::VectorXd Curv2(nFaces);
        Eigen::VectorXd CurvExpRatio1(nFaces);
        Eigen::VectorXd CurvExpRatio2(nFaces);
        Eigen::VectorXd PrincCurvExpRatio(nFaces);

        ComputeCurvatures<tMesh> computeCurvatures;
        computeCurvatures.computePrincGrowthDir(mesh, gauss, mean, PrincCurv1, PrincCurv2, Curv1, Curv2, growthDirs_1_bot, growthDirs_2_bot);

        for(int i=0;i<nFaces;++i)
        {

          CurvExpRatio1(i) = Curv1(i) / (growthFacs_1_bot(i)-1.0);
          CurvExpRatio2(i) = Curv2(i) / (growthFacs_2_bot(i)-1.0);

          if (std::abs(growthFacs_1_bot(i)-1.0) > std::abs(growthFacs_2_bot(i)-1.0)){
            PrincCurvExpRatio(i) = CurvExpRatio1(i);
          } else {
            PrincCurvExpRatio(i) = CurvExpRatio2(i);
          }
        }

        writer.addScalarFieldToFaces(gauss, "gauss");
        writer.addScalarFieldToFaces(mean, "mean");
        writer.addScalarFieldToFaces(PrincCurv1, "PrincCurv1");
        writer.addScalarFieldToFaces(PrincCurv2, "PrincCurv2");
        writer.addScalarFieldToFaces(Curv1, "Curv1");
        writer.addScalarFieldToFaces(Curv2, "Curv2");
        writer.addScalarFieldToFaces(CurvExpRatio1, "CurvExpRatio1");
        writer.addScalarFieldToFaces(CurvExpRatio2, "CurvExpRatio2");
        writer.addScalarFieldToFaces(PrincCurvExpRatio, "PrincCurvExpRatio");

        helpers::write_matrix(filename + "_Curv1.txt", Curv1);
        helpers::write_matrix(filename + "_Curv2.txt", Curv2);
        helpers::write_matrix(filename + "_growthFacs_1_bot.txt", growthFacs_1_bot);
        helpers::write_matrix(filename + "_growthFacs_2_bot.txt", growthFacs_2_bot);
        helpers::write_matrix(filename + "_growthDirs_1_bot.txt", growthDirs_1_bot);
        helpers::write_matrix(filename + "_growthDirs_2_bot.txt", growthDirs_2_bot);

    }

    writer.addScalarFieldToFaces(growthFacs_1_bot, "rate1_bot");
    writer.addScalarFieldToFaces(growthFacs_2_bot, "rate2_bot");
    writer.addScalarFieldToFaces(growthFacs_1_top, "rate1_top");
    writer.addScalarFieldToFaces(growthFacs_2_top, "rate2_top");
    writer.addVectorFieldToFaces(growthDirs_1_bot, "dir1_bot");
    writer.addVectorFieldToFaces(growthDirs_2_bot, "dir2_bot");
    writer.addVectorFieldToFaces(growthDirs_1_top, "dir1_top");
    writer.addVectorFieldToFaces(growthDirs_2_top, "dir2_top");
    writer.addScalarFieldToFaces(difference_1, "difference_1");

    helpers::write_matrix(filename + "_vertices.txt", cvertices);
    helpers::write_matrix(filename + "_face2vertices.txt", cface2vertices);

    writer.write(filename);
}




void Sim_InverseBilayer::Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot)
{
  *theta_top = heq*epseq*(3*h-2*heq)/std::pow(h,2);
  *theta_bot = heq*epseq*(2*heq-h)/std::pow(h,2);
}



void Sim_InverseBilayer::Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot)
{
  *heq=(theta_top+3.0*theta_bot)*h/((theta_top+theta_bot)*2.0);
  *epseq=std::pow((theta_top+theta_bot),2)/(theta_top+3.0*theta_bot);
}



// create a HLBFGS wrapper
template<typename tMesh>
class Parametrizer_BiLayer_Viability : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

public:

    Parametrizer_BiLayer_Viability(tMesh & mesh_in):
    Parametrizer<tMesh>(mesh_in)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> , const bool reAllocate = false) override
    {
        initSolution(reAllocate);
    }

    void initSolution(const bool reAllocate = false)
    {
        if((data==nullptr) or reAllocate)
        {
            if(data!=nullptr)
            {
                delete [] data;
                data = nullptr;
            }
            const int nVariables = getNumberOfVariables();
            data = new Real[nVariables];

            // assign the initial mesh data
            const int nVertices = mesh.getNumberOfVertices();
            const int nEdges = mesh.getNumberOfEdges();
            const auto vertices = mesh.getCurrentConfiguration().getVertices();
            const auto edgedirs = mesh.getCurrentConfiguration().getEdgeDirectors();
            const auto rvertices = mesh.getRestConfiguration().getVertices();
            for(int i=0;i<nVertices;++i)
                for(int d=0;d<3;++d)
                    data[d*nVertices + i] = vertices(i,d);

            for(int i=0;i<nEdges;++i)
                data[3*nVertices + i] = edgedirs(i);

            for(int i=0;i<nVertices;++i)
                for(int d=0;d<2;++d)
                    data[3*nVertices + nEdges + d*nVertices + i] = rvertices(i,d);

        }
        // also update the mesh
        updateSolution();
    }

    int getNumberOfVariables() const override
    {
        return 5*mesh.getNumberOfVertices() + mesh.getNumberOfEdges();
    }


    void updateSolution() override
    {
        auto vertices = mesh.getCurrentConfiguration().getVertices();
        auto edgedirs = mesh.getCurrentConfiguration().getEdgeDirectors();
        auto rvertices = mesh.getRestConfiguration().getVertices();
        const int nVertices = mesh.getNumberOfVertices();
        const int nEdges = mesh.getNumberOfEdges();
        for(int i=0;i<nVertices;++i)
            for(int d=0;d<3;++d)
                vertices(i,d) = data[d*nVertices + i];

        for(int i=0;i<nEdges;++i)
            edgedirs(i) = data[3*nVertices + i];

        for(int i=0;i<nVertices;++i)
            for(int d=0;d<2;++d)
                rvertices(i,d) = data[3*nVertices + nEdges + d*nVertices + i];

        mesh.getRestConfiguration().setFormsFromVertices(mesh.getTopology(), mesh.getBoundaryConditions());
        mesh.updateDeformedConfiguration();
    }

    void updateGradient(const int nVars, const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        for(int i=0;i<nVars;++i)
            grad_ptr[i] = energyGradient(i);
    }
};
