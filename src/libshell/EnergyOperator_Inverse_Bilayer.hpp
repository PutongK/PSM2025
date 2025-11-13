//
//  EnergyOperator_Inverse_Bilayer.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/18/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef EnergyOperator_Inverse_Bilayer_h
#define EnergyOperator_Inverse_Bilayer_h

#include "common.hpp"
#include "EnergyOperator_Inverse.hpp"

template<typename tMesh>
class EnergyOperator_Inverse_Bilayer : public EnergyOperator_Inverse<tMesh>
{
protected:
    const EnergyOperator_Inverse<tMesh> & botLayer;
    const EnergyOperator_Inverse<tMesh> & topLayer;
    
    Real computeAll(const tMesh & , Eigen::Ref<Eigen::VectorXd> , const bool ) const override
    {
        helpers::catastrophe("should not be inside computeAll of EnergyOperator_Inverse_Bilayer\n",__FILE__,__LINE__);
        return -1;
    }
    
public:
    EnergyOperator_Inverse_Bilayer(const EnergyOperator_Inverse<tMesh> & botLayer_in, const EnergyOperator_Inverse<tMesh> & topLayer_in):
    EnergyOperator_Inverse<tMesh>(),
    botLayer(botLayer_in),
    topLayer(topLayer_in)
    {}
    
    // without gradient (just energy)
    Real compute(const tMesh & mesh) const override
    {
        const Real energy_bot = botLayer.compute(mesh);
        const Real energy_top = topLayer.compute(mesh);
        return energy_bot + energy_top;
    }
    
    // with gradient (and energy)
    Real compute(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient) const override
    {
        const int nVars_bot = botLayer.getNumberOfVariables(mesh);
        const int nVars_top = topLayer.getNumberOfVariables(mesh);
        
        // map gradients
        Eigen::Map<Eigen::VectorXd> gradient_botLayer(gradient.data(), nVars_bot);
        Eigen::Map<Eigen::VectorXd> gradient_topLayer(gradient.data() + nVars_bot, nVars_top);
        
        const Real energy_bot = botLayer.compute(mesh, gradient_botLayer);
        const Real energy_top = topLayer.compute(mesh, gradient_topLayer);
        return energy_bot + energy_top;
    }
    
    virtual void printProfilerSummary() const override
    {
        botLayer.printProfilerSummary();
        topLayer.printProfilerSummary();
    }
    
    int getNumberOfVariables(const tMesh & mesh) const override
    {
        return botLayer.getNumberOfVariables(mesh) + topLayer.getNumberOfVariables(mesh);
    }
    
    
    
};
#endif /* EnergyOperator_Inverse_Bilayer_h */
