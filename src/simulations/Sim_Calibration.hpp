//
//  Sim_Calibration.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_Calibration_hpp
#define Sim_Calibration_hpp

#include "Sim_Inverse.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "ParametricSurfaceLibrary.hpp"
#include "GrowthHelper.hpp"
#include "EnergyOperator.hpp"
#include "CombinedOperator_Parametric.hpp"

#include "ComputeErrorMap.hpp"

class Sim_Calibration : public Sim_Inverse<BilayerMesh>
{
    typedef BilayerMesh tMesh;

    void Almen_Optim(const Real deformRad = -1);
    void Wave_Optim(const Real deformRad = -1);

    void Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot);
    void Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot);

    void dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig=false);
    void dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig=false, bool curvinvert=false);
public:

    Sim_Calibration(ArgumentParser & parser):
    Sim_Inverse<tMesh>(parser)
    {}

    void init() override;
    void run() override;
};

#endif /* Sim_Calibration_hpp */
