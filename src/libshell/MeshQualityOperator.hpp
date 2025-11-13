//
//  MeshQualityOperator.hpp
//  Elasticity
//
//  Created by Wim van Rees on 12/2/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef MeshQualityOperator_hpp
#define MeshQualityOperator_hpp

#include "common.hpp"
#include "EnergyOperator.hpp"

template<typename tMesh>
class MeshQualityOperator : public EnergyOperator_DCS<tMesh>
{
public:
    using EnergyOperator_DCS<tMesh>::compute;
    
protected:
    const Real area_prefac;
    const int area_exponent;
    const bool keepNormalToTarget;
    const std::function<Eigen::Vector3d(const Eigen::Vector3d)> target_normal_func;
    const Real normal_prefac;
    const Real normal_k;
    
    const std::function<Eigen::Vector3d(const Eigen::Vector3d)> trafo_func;

    mutable Real lastEnergy;
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, const bool computeGradient) const override;
    
public:
    
    MeshQualityOperator(const Real area_prefac_in = 1, const int area_exponent_in = 2):
    EnergyOperator_DCS<tMesh>(),
    area_prefac(area_prefac_in),
    area_exponent(area_exponent_in),
    keepNormalToTarget(false),
    target_normal_func([](Eigen::Vector3d ){return (Eigen::Vector3d() << 0,0,1).finished();}),
    normal_prefac(0),
    normal_k(0),
    trafo_func([](Eigen::Vector3d pos){return pos;}),
    lastEnergy(0)
    {}
    
    MeshQualityOperator(const Real area_prefac_in, const int area_exponent_in, const std::function<Eigen::Vector3d(const Eigen::Vector3d)> target_normal_func, const Real normal_prefac_in = 100, const Real normal_k_in = 100):
    EnergyOperator_DCS<tMesh>(),
    area_prefac(area_prefac_in),
    area_exponent(area_exponent_in),
    keepNormalToTarget(true),
    target_normal_func(target_normal_func),
    normal_prefac(normal_prefac_in),
    normal_k(normal_k_in),
    trafo_func([](Eigen::Vector3d pos){return pos;}),
    lastEnergy(0)
    {}
    
    MeshQualityOperator(const Real area_prefac_in, const int area_exponent_in, const std::function<Eigen::Vector3d(const Eigen::Vector3d)> target_normal_func, const Real normal_prefac_in, const Real normal_k_in, const std::function<Eigen::Vector3d(const Eigen::Vector3d)> trafo_func):
    EnergyOperator_DCS<tMesh>(),
    area_prefac(area_prefac_in),
    area_exponent(area_exponent_in),
    keepNormalToTarget(true),
    target_normal_func(target_normal_func),
    normal_prefac(normal_prefac_in),
    normal_k(normal_k_in),
    trafo_func(trafo_func),
    lastEnergy(0)
    {}
            
    virtual void addEnergy(std::vector<std::pair<std::string, Real>> & out) const override
    {
        out.push_back(std::make_pair("meshquality", lastEnergy));
    }
};


#endif /* MeshQualityOperator_hpp */
