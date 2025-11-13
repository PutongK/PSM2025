//
//  AreaDistortionOperator.hpp
//  Elasticity
//
//  Created by Wim van Rees on 6/9/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#ifndef AreaDistortionOperator_hpp
#define AreaDistortionOperator_hpp

#include "common.hpp"
#include "EnergyOperator.hpp"

template<typename tMesh>
class AreaDistortionOperator : public EnergyOperator_DCS<tMesh>
{
public:
    using EnergyOperator_DCS<tMesh>::compute;
    
protected:
    const Eigen::Ref<const Eigen::VectorXd> targetFaceAreas;
    const Real eng_fac;
    mutable Real lastEnergy;
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, const bool computeGradient) const override;
    
public:
    
    AreaDistortionOperator(const Eigen::Ref<const Eigen::VectorXd> targetFaceAreas_in, const Real eng_fac_in = 1):
    EnergyOperator_DCS<tMesh>(),
    targetFaceAreas(targetFaceAreas_in),
    eng_fac(eng_fac_in),
    lastEnergy(0)
    {}
    
    virtual void addEnergy(std::vector<std::pair<std::string, Real>> & out) const override
    {
        out.push_back(std::make_pair("areadistortion", lastEnergy));
    }
};

#endif /* AreaDistortionOperator_hpp */
