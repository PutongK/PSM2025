//
//  DirichletEnergyOperator_Inverse.hpp
//  Elasticity
//
//  Created by Wim van Rees on 9/14/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef DirichletEnergyOperator_Inverse_hpp
#define DirichletEnergyOperator_Inverse_hpp

#include "common.hpp"
#include "EnergyOperator_Inverse.hpp"

template<typename tMesh, MeshLayer layer = single>
class DirichletEnergyOperator_Inverse : public EnergyOperator_Inverse<tMesh>
{
public:
    using EnergyOperator_Inverse<tMesh>::compute;
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;
    
protected:
    // prefactor for the energy
    Real prefactor;
    
    virtual Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient, const bool computeGradient) const override;
    
public:
    
    DirichletEnergyOperator_Inverse(const Real prefac):
    EnergyOperator_Inverse<tMesh>(),
    prefactor(prefac)
    {
    }
    
//    void computePerFaceEnergies(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> per_face_eng) const;
    
    void printProfilerSummary() const override
    {
        if(layer == bottom) std::cout << "=== BOTTOM LAYER === \n";
        if(layer == top)  std::cout << "=== TOP LAYER === \n";
        this->profiler.printSummary();
    }
    
    int getNumberOfVariables(const tMesh & mesh) const override
    {
        return 3*mesh.getNumberOfFaces();
    }
    
    void setPrefactor(const Real prefac_in)
    {
        prefactor = prefac_in;
    }
};

#endif /* DirichletEnergyOperator_Inverse_hpp */
