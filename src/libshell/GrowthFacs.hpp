//
//  GrowthFacs.hpp
//  Elasticity
//
//  Created by Wim van Rees on 3/20/17.
//  Copyright © 2017 Wim van Rees. All rights reserved.
//

#ifndef GrowthFacs_hpp
#define GrowthFacs_hpp

#include "common.hpp"
#include "WriteVTK.hpp"

struct GrowthFacs
{
    const int nFaces;
    GrowthFacs(const int nFaces):
    nFaces(nFaces)
    {}
    
    virtual Eigen::Matrix2d gimmeMatrix(const int i) const = 0;
    virtual void addToWriter(WriteVTK & writer, const std::string postfix = "") const = 0;
};


struct GrowthFacs_Iso : GrowthFacs
{
    Eigen::VectorXd growthRates;
    
    GrowthFacs_Iso(const int nFaces):
    GrowthFacs(nFaces)
    {
        growthRates.resize(nFaces);
    }
    
    Eigen::Matrix2d gimmeMatrix(const int i) const override
    {
        const Eigen::Matrix2d retval = std::pow(growthRates(i) + 1, 2) * Eigen::Matrix2d::Identity();
        return retval;
    }
    
    void addToWriter(WriteVTK & writer, const std::string postfix = "") const override
    {
        writer.addScalarFieldToFaces(growthRates, "rate" + postfix);
    }
};


struct GrowthFacs_Ortho : GrowthFacs
{
    Eigen::VectorXd growthRates_1, growthRates_2, growthAngles;
    
    GrowthFacs_Ortho(const int nFaces):
    GrowthFacs(nFaces)
    {
        growthRates_1.resize(nFaces);
        growthRates_2.resize(nFaces);
        growthAngles.resize(nFaces);
    }
    
    Eigen::Matrix2d gimmeMatrix(const int i) const override
    {
        const Eigen::Vector2d dir_1 = (Eigen::Vector2d() << std::cos(growthAngles(i)) , std::sin(growthAngles(i))).finished();
        const Eigen::Vector2d dir_2 = (Eigen::Vector2d() << -std::sin(growthAngles(i)) , std::cos(growthAngles(i))).finished();
        const Eigen::Matrix2d retval = std::pow(growthRates_1(i) + 1, 2) * dir_1 * dir_1.transpose() + std::pow(growthRates_2(i) + 1, 2) * dir_2 * dir_2.transpose();
        return retval;
    }
    
    void addToWriter(WriteVTK & writer, const std::string postfix = "") const override
    {
        writer.addScalarFieldToFaces(growthRates_1, "rate1" + postfix);
        writer.addScalarFieldToFaces(growthRates_2, "rate2" + postfix);
        writer.addScalarFieldToFaces(growthAngles, "dir" + postfix);
    }
};

struct GrowthFacs_Ortho_Shell : GrowthFacs
{
    Eigen::VectorXd growthRates_1, growthRates_2;
    Eigen::MatrixXd growthAngles;
    
    GrowthFacs_Ortho_Shell(const int nFaces):
    GrowthFacs(nFaces)
    {
        growthRates_1.resize(nFaces);
        growthRates_2.resize(nFaces);
        growthAngles.resize(nFaces, 3);
    }
    
    Eigen::Matrix2d gimmeMatrix(const int ) const override
    {
        std::cout << "gimmeMatrix not implemented yet for GrowthFacs_Ortho_Shell" << std::endl;
        return Eigen::Matrix2d::Constant(0);
    }
    
    void addToWriter(WriteVTK & writer, const std::string postfix = "") const override
    {
        writer.addScalarFieldToFaces(growthRates_1, "rate1" + postfix);
        writer.addScalarFieldToFaces(growthRates_2, "rate2" + postfix);
        writer.addVectorFieldToFaces(growthAngles, "dir" + postfix);
    }
};


#endif /* GrowthFacs_hpp */
