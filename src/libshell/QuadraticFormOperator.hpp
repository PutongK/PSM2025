//
//  QuadraticFormOperator.hpp
//  Elasticity
//
//  Created by Wim van Rees on 1/30/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#ifndef QuadraticFormOperator_hpp
#define QuadraticFormOperator_hpp

#include "common.hpp"
#include "EnergyOperator.hpp"
#include "MaterialProperties.hpp"

template<typename tMesh, typename tMaterialType = Material_Isotropic>
class QuadraticFormOperator : public EnergyOperator_DCS<tMesh>
{
public:
    using EnergyOperator_DCS<tMesh>::compute;
    
protected:
    const MaterialProperties<tMaterialType> & material_properties;
//    const Eigen::Ref<const Eigen::MatrixXd> target_vertices;
    const tVecMat2d & target_firstFF, target_secondFF;
    
    const int exponent;
    
    const Real weight_rates;
    const Real weight_surface;
    
    const Real minBound;
    const Real maxBound;
    const bool restConfigGradients;
    
    mutable Real lastEnergy;
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, const bool computeGradient) const override
    {
        assert(not restConfigGradients);
        Eigen::MatrixXd dummy;
        return computeAll(mesh, gradient_vertices, gradient_edges, dummy, computeGradient);
    }
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> gradient_edges, Eigen::Ref<Eigen::MatrixXd> gradient_restvertices, const bool computeGradient) const;
    
public:


    QuadraticFormOperator(const MaterialProperties<tMaterialType> & material_properties_in, const tVecMat2d & target_firstFF, const tVecMat2d & target_secondFF, const int exponent_in, const Real weight_rates_in, const Real weight_surface_in, const Real minBound_in, const Real maxBound_in, const bool restConfigGradients_in = false):
    EnergyOperator_DCS<tMesh>(),
    material_properties(material_properties_in),
    target_firstFF(target_firstFF),
    target_secondFF(target_secondFF),
    exponent(exponent_in),
    weight_rates(weight_rates_in),
    weight_surface(weight_surface_in),
    minBound(minBound_in),
    maxBound(maxBound_in),
    restConfigGradients(restConfigGradients_in),
    lastEnergy(0)
    {
    }
    
    
    virtual Real compute(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient) const override
    {
        assert(mesh.getNumberOfVertices() > 0);
        
        Real energy;
        if(not restConfigGradients)
        {
            assert(gradient.rows() >= 3*mesh.getNumberOfVertices() + mesh.getNumberOfEdges());
            
            const int nVertices = mesh.getNumberOfVertices();
            const int nEdges = mesh.getNumberOfEdges();
            
            // map the gradient vector into two separate parts: a column-major matrix and a vector
            // memory layout is [v0x, v1x, .. ,vNx, v0y, v1y, .. ,vNy, v0z, v1z, .. ,vNz, e0, e1, .. ,eN]^T
            Eigen::Map<Eigen::MatrixXd> grad_vertices(gradient.data(), nVertices, 3);
            Eigen::Map<Eigen::VectorXd> grad_edges(gradient.data() + 3*nVertices, nEdges);
            energy = computeAll(mesh, grad_vertices, grad_edges, true);
        }
        else
        {
            assert(gradient.rows() >= 5*mesh.getNumberOfVertices() + mesh.getNumberOfEdges());
            
            const int nVertices = mesh.getNumberOfVertices();
            const int nEdges = mesh.getNumberOfEdges();
            
            // map the gradient vector into two separate parts: a column-major matrix and a vector
            // memory layout is [v0x, v1x, .. ,vNx, v0y, v1y, .. ,vNy, v0z, v1z, .. ,vNz, e0, e1, .. ,eN]^T
            Eigen::Map<Eigen::MatrixXd> grad_vertices(gradient.data(), nVertices, 3);
            Eigen::Map<Eigen::VectorXd> grad_edges(gradient.data() + 3*nVertices, nEdges);
            Eigen::Map<Eigen::MatrixXd> grad_restvertices(gradient.data() + 3*nVertices + nEdges, nVertices, 2);
            energy = computeAll(mesh, grad_vertices, grad_edges, grad_restvertices, true);
        }
        
        return energy;
    }
    
    virtual int getNumberOfVariables(const tMesh & mesh) const override
    {
        return (restConfigGradients ? 5 : 3)*mesh.getNumberOfVertices() + mesh.getNumberOfEdges();
    }
    
    virtual void addEnergy(std::vector<std::pair<std::string, Real>> & out) const override
    {
        out.push_back(std::make_pair("quadraticform", lastEnergy));
    }
};


#endif /* QuadraticFormOperator_hpp */
