//
//  Sim_InverseBilayer.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_InverseBilayer_hpp
#define Sim_InverseBilayer_hpp

#include "Sim_Inverse.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "ParametricSurfaceLibrary.hpp"
#include "GrowthHelper.hpp"
#include "EnergyOperator.hpp"
#include "CombinedOperator_Parametric.hpp"

#include "ComputeErrorMap.hpp"

class Sim_InverseBilayer : public Sim_Inverse<BilayerMesh>
{
    typedef BilayerMesh tMesh;

    std::pair<GrowthFacs_Ortho, GrowthFacs_Ortho> computeGrowthFactors(const tVecMat2d & aform_final, const tVecMat2d & bform_final, const tVecMat2d & aform_rest, const MaterialProperties_Iso_Array & matprop);

    void inverseTestOne(const Real deformRad=-1);
    void inverseTestMultiple(const Real deformRad=-1);
    void inverseTestOptim(const Real deformRad=-1);
    void inverseTestExact(const Real deformRad=-1);

    void Grouping(const int nFaces, const int nClusters, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters, const Eigen::Ref<const Eigen::MatrixXd> growthfactors, const Eigen::Ref<const Eigen::MatrixXi> adj_faces, int t = 1);
    void Kmeans(const int nTries_kmeans, const int nClusters, const int nFaces, const Real h_total, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters);
    void Filtering(const int nFaces, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters, const Eigen::Ref<const Eigen::MatrixXi> adj_faces, int t = 0);

    void MarginCut(const Real margin_width, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, Eigen::Ref<Eigen::VectorXi> Clusters);
    void GrowthAdjust(const int nFaces, const Real maxratio, Eigen::Ref<Eigen::VectorXd> area_t, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top, const Eigen::Ref<const Eigen::VectorXd> mean_t, const Eigen::Ref<const Eigen::VectorXd> princcurv1_t, const int m, const int t, const bool just_curvature = false);
    void ErrorMap(const std::string fname1, const std::string fname2, const Real k, Real *maxdist, Real* relerror, Real*squares_sum);

    void Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot);
    void Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot);

    void dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig=false);
    void dumpOrthoNew(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const std::string filename, const bool restConfig = false, const bool restGrowth = false);
    void dumpOrtho_Princdircurv(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const std::string filename, const bool restConfig = false, const bool restGrowth = false);
    void dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig=false);

    void assignGrowthToMetric(const std::vector<GrowthState> & growth_bot, const std::vector<GrowthState> & growth_top, const Real t, const bool logeucl);
    void computeGrowthForCurrentState(const MaterialProperties_Iso_Constant & matprop_bot, const MaterialProperties_Iso_Constant & matprop_top, std::vector<GrowthState> & growth_bot, std::vector<GrowthState> & growth_top, const bool finalConfig, const bool flipLayers = false, const bool just_curvature = false);

public:

    Sim_InverseBilayer(ArgumentParser & parser):
    Sim_Inverse<tMesh>(parser)
    {}

    void init() override;
    void run() override;
};

#endif /* Sim_InverseBilayer_hpp */
