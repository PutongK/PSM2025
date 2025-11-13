//
//  Sim_Bilayer_Growth.hpp
//  Elasticity
//
//  Created by Wim van Rees on 10/27/16. 
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_Bilayer_Growth_hpp
#define Sim_Bilayer_Growth_hpp

#include "Sim.hpp"
#include "Mesh.hpp"

class Sim_Bilayer_Growth : public Sim<BilayerMesh>
{
    typedef BilayerMesh tMesh;
protected:

    void TestCustomGrowth();
    void TestRandomPatterns();
    void initForwardProblem();

    void dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig=false);
    void dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig=false);

    void computeQuadraticForms(tVecMat2d & firstFF, tVecMat2d & secondFF);
    void MarginCut(const Real margin_x, const Real margin_y, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top);
    void Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot);
    void Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot);

public:

    Sim_Bilayer_Growth(ArgumentParser & parser):
    Sim<tMesh>(parser)
    {}

    void init() override;
    void run() override;
};

#endif /* Sim_Bilayer_Growth_hpp */
