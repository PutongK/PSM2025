//
//  EnergyOperator_Inverse.hpp
//  Elasticity
//
//  Created by Wim van Rees on 8/15/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef EnergyOperator_Inverse_hpp
#define EnergyOperator_Inverse_hpp

#include "common.hpp"
#include "EnergyOperator.hpp"

template<typename tMesh>
class EnergyOperator_Inverse : public EnergyOperator<tMesh>
{
protected:
    
    virtual Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient, const bool computeGradient) const = 0;
    
public:
    
    // without gradient (just energy)
    Real compute(const tMesh & mesh) const override
    {
        Eigen::VectorXd dummy;
        const Real energy = computeAll(mesh, dummy, false);
        return energy;
    }
    
    // with gradient (and energy)
    Real compute(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient) const override
    {
        const Real energy = computeAll(mesh, gradient, true);
        return energy;
    }
    
    virtual void printProfilerSummary() const override
    {
        this->profiler.printSummary();
    }
};


#endif /* EnergyOperator_Inverse_hpp */
