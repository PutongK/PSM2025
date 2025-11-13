//
//  CombinedOperator_Parametric_InverseGrowth_Bilayer.hpp
//  Elasticity
//
//  Created by Wim M. van Rees on 8/15/18.
//  Copyright © 2018 Wim van Rees. All rights reserved.
//

#ifndef CombinedOperator_Parametric_InverseGrowth_Bilayer_hpp
#define CombinedOperator_Parametric_InverseGrowth_Bilayer_hpp

#include "common.hpp"
#include "EnergyOperator_Inverse.hpp"
#include "MaterialProperties.hpp"

template<typename tMesh, typename tMaterialType = Material_Isotropic>
class CombinedOperator_Parametric_InverseGrowth_Bilayer : public EnergyOperator_Inverse<tMesh>
{
public:
    typedef typename tMesh::tCurrentConfigData tCurrentConfigData;
    
protected:
    const MaterialProperties<tMaterialType> & material_properties_bot; // layer 1
    const MaterialProperties<tMaterialType> & material_properties_top; // layer 2
    
    const bool withGradientThickness;
    const bool withGradientYoung;
    
    Real computeAll(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient, const bool computeGradient) const override;
    
public:

    CombinedOperator_Parametric_InverseGrowth_Bilayer(const MaterialProperties<tMaterialType> & material_properties, const bool withGradientThickness_in = false, const bool withGradientYoung_in = false):
    EnergyOperator_Inverse<tMesh>(),
    material_properties_bot(material_properties),
    material_properties_top(material_properties),
    withGradientThickness(withGradientThickness_in),
    withGradientYoung(withGradientYoung_in)
    {}
    
    CombinedOperator_Parametric_InverseGrowth_Bilayer(const MaterialProperties<tMaterialType> & material_properties_bot, const MaterialProperties<tMaterialType> & material_properties_top, const bool withGradientThickness_in = false, const bool withGradientYoung_in = false):
    EnergyOperator_Inverse<tMesh>(),
    material_properties_bot(material_properties_bot),
    material_properties_top(material_properties_top),
    withGradientThickness(withGradientThickness_in),
    withGradientYoung(withGradientYoung_in)
    {}
    
    int getNumberOfVariables(const tMesh & mesh) const override
    {
        const int nFaces = mesh.getNumberOfFaces();
        int nComponents = 3;
//        if(withGradientThickness) nComponents++;
//        if(withGradientYoung) nComponents++;
        return 2*nComponents*nFaces; // abar (3 components/face) for each layer
    }
};

#endif /* CombinedOperator_Parametric_InverseGrowth_Bilayer_hpp */
