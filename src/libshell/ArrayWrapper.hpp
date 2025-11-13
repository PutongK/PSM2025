//
//  ArrayWrapper.hpp
//  Elasticity
//
//  Created by Wim van Rees on 8/5/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef ArrayWrapper_hpp
#define ArrayWrapper_hpp

#include "common.hpp"

struct ArrayWrapper
{
    Eigen::VectorXd array;
    Real prefac;
    
    virtual Real func(const Real x) const = 0;
    virtual Real d_func(const Real x) const = 0;
    
    ArrayWrapper(const int nFaces, const Real initVal):
    array(Eigen::VectorXd::Constant(nFaces, initVal)),
    prefac(1.0)
    {}
    
    virtual void assign(const Eigen::Ref<const Eigen::VectorXd> parameter)
    {
        for(int i=0;i<array.rows();++i)
            array(i) = prefac * func(parameter(i));
    }
    
    virtual void applyChainRule(const Eigen::Ref<const Eigen::VectorXd> parameter, const Eigen::Ref<const Eigen::VectorXd> gradient_val, Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        // apply chain rule in different array
        for(int i=0;i<array.rows();++i)
            gradient_parameter(i) = gradient_val(i) * prefac * d_func(parameter(i));
    }
    
    virtual void applyChainRule(const Eigen::Ref<const Eigen::VectorXd> parameter, Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        // apply chain rule to the same array
        for(int i=0;i<array.rows();++i)
            gradient_parameter(i) *= prefac * d_func(parameter(i));
    }
    
    virtual void setPrefac(const Real prefac_in)
    {
        prefac = prefac_in;
    }
    
    int getSize() const
    {
        return array.rows();
    }
};

struct PeriodicBoundedArray : public ArrayWrapper
{

    const Real minVal, maxVal;
    
    Real func(const Real x) const override
    {
        return prefac * (minVal + 0.5*(maxVal - minVal)*(std::sin(x) + 1.0));
    }
    
    Real d_func(const Real x) const override
    {
        return 0.5*prefac*(maxVal - minVal)*std::cos(x);
    }
    
    PeriodicBoundedArray(const int nFaces, const Real minVal, const Real maxVal, const Real initVal=0):
    ArrayWrapper(nFaces, initVal),
    minVal(minVal),
    maxVal(maxVal)
    {}
};

struct IdentityArray : public ArrayWrapper
{
    Real func(const Real x) const override
    {
        return x;
    }
    
    Real d_func(const Real  ) const override
    {
        return 1;
    }
    
    IdentityArray(const int nFaces, const Real , const Real , const Real initVal=0):
    ArrayWrapper(nFaces, initVal)
    {}
};

#if 1==0
struct LowerBoundedArray
{
    Eigen::VectorXd array;
    const Real minVal;
    Real prefac;
    
    Real func(const Real x) const
    {
        return prefac * (minVal + x*x);
    }
    
    Real d_func(const Real x) const
    {
        return 2.0*prefac*x;
    }
    
    PeriodicBoundedArray(const int nFaces, const Real minVal, const Real initVal=0):
    array(Eigen::VectorXd::Constant(nFaces, initVal)),
    minVal(minVal),
    prefac(1.0)
    {}
    
    void assign(const Eigen::Ref<const Eigen::VectorXd> parameter)
    {
        for(int i=0;i<array.rows();++i)
            array(i) = func(parameter(i));
    }
    
    void applyChainRule(const Eigen::Ref<const Eigen::VectorXd> parameter, const Eigen::Ref<const Eigen::VectorXd> gradient_val, Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        for(int i=0;i<array.rows();++i)
            gradient_parameter(i) = gradient_val(i)*d_func(parameter(i));
    }
    
    void applyChainRule(const Eigen::Ref<const Eigen::VectorXd> parameter, Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        for(int i=0;i<array.rows();++i)
            gradient_parameter(i) *= d_func(parameter(i));
    }
    
    void setPrefac(const Real prefac_in)
    {
        prefac = prefac_in;
    }
};
#endif



#endif /* ArrayWrapper_hpp */
