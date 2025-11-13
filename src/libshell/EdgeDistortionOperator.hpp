//
//  EdgeDistortionOperator.hpp
//  Elasticity
//
//  Created by Wim van Rees on 6/9/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#ifndef EdgeDistortionOperator_hpp
#define EdgeDistortionOperator_hpp

#include "common.hpp"
#include "EnergyOperator.hpp"

template<typename tMesh>
class EdgeDistortionOperator : public EnergyOperator_DCS<tMesh>
{
public:
    using EnergyOperator_DCS<tMesh>::compute;
    
protected:
    const Real alpha;
    const Eigen::Ref<const Eigen::VectorXd> targetEdgeLengths;
    mutable Real lastEnergy;
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, const bool computeGradient) const override;
    
public:
    
    EdgeDistortionOperator(const Real alpha_in, const Eigen::Ref<const Eigen::VectorXd> targetEdgeLengths_in):
    EnergyOperator_DCS<tMesh>(),
    alpha(alpha_in),
    targetEdgeLengths(targetEdgeLengths_in),
    lastEnergy(0)
    {}
    
    virtual void addEnergy(std::vector<std::pair<std::string, Real>> & out) const override
    {
        out.push_back(std::make_pair("edgedistortion", lastEnergy));
    }
};

#endif /* EdgeDistortionOperator_hpp */
