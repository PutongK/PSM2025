//
//  Parametrizer_InverseGrowth.hpp
//  Elasticity
//
//  Created by Wim van Rees on 8/17/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef Parametrizer_InverseGrowth_h
#define Parametrizer_InverseGrowth_h

#include "Parametrizer.hpp"
#include "GrowthHelper.hpp"
#include "MaterialProperties.hpp"

struct DataWrapper
{
    // assume psi = f(phi) where f is some mapping function, psi is the actual data (eg growthrates) and phi is the underlying parameters we optimize
    Eigen::VectorXd data;
    Eigen::Map<Eigen::VectorXd> params;

    virtual Real func(const Real x) const = 0;
    virtual Real d_func(const Real x) const = 0;
    virtual Real func_inverse(const Real x) const { return x; };

    DataWrapper(const int nVals):
    data(nVals),
    params(nullptr,0)
    {}

    void linkParams(Real * data, const int nData)
    {
        new (&params) Eigen::Map<Eigen::VectorXd>(data,nData);
    }

    virtual void assign_inverse()
    {
        // assume that params contains the actual data value and we need to map that properly
        for(int i=0;i<params.rows();++i)
        {
            params(i) = func_inverse(params(i));
            data(i) = func(params(i));
        }
    }

    virtual void assign()
    {
        for(int i=0;i<params.rows();++i)
            data(i) = func(params(i));
    }

    virtual void applyChainRule(const Eigen::Ref<const Eigen::VectorXd> gradient_val, Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        // apply chain rule in different array
        for(int i=0;i<params.rows();++i)
            gradient_parameter(i) = gradient_val(i) * d_func(params(i));
    }

    virtual void applyChainRule(Eigen::Ref<Eigen::VectorXd> gradient_parameter) const
    {
        // apply chain rule to the same array
        //std::cout<<"hhhh = "<< params.rows() <<std::endl;
        for(int i=0;i<params.rows();++i)
            gradient_parameter(i) *= d_func(params(i));
    }

};

struct LowerUpperBound : DataWrapper
{
    const Real minVal;
    const Real maxVal;

    LowerUpperBound(const int nVals, const Real lowerBound, const Real upperBound):
    DataWrapper(nVals),
    minVal(lowerBound),
    maxVal(upperBound)
    {}

    Real func(const Real x) const override
    {
        return (minVal + 0.5*(maxVal - minVal)*(std::sin(x) + 1.0));
    }

    Real d_func(const Real x) const override
    {
        return 0.5*(maxVal - minVal)*std::cos(x);
    }

    Real func_inverse(const Real x) const override
    {
        const Real arg_raw = 2.0*(x - minVal)/(maxVal - minVal) - 1;
        const Real eps = std::numeric_limits<Real>::epsilon();
        const Real arg = std::min(+1.0 - eps, std::max(-1.0 + eps, arg_raw)); // make sure we are between bounds
        return std::asin(arg);
    }
};




/* =============================== BILAYER METHODS BELOW ============================*/

template<typename tMesh>
class Parametrizer_Bilayer_IsoGrowth_Base : public Parametrizer<tMesh>
{

public:
    Parametrizer_Bilayer_IsoGrowth_Base(tMesh & mesh_in):
    Parametrizer<tMesh>(mesh_in)
    {
    }

    virtual const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const = 0;
    virtual const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const = 0;
};

template<typename tMesh>
class Parametrizer_Bilayer_IsoGrowth : public Parametrizer_Bilayer_IsoGrowth_Base<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

public:

    Parametrizer_Bilayer_IsoGrowth(tMesh & mesh_in):
    Parametrizer_Bilayer_IsoGrowth_Base<tMesh>(mesh_in)
    {
    }

    int getNumberOfVariables() const override
    {
        return 2*mesh.getNumberOfFaces(); // two swelling rate per face
    }

    void updateSolution() override
    {
        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthRates_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_top(data + nFaces, nFaces);

        // use helper method to assign the first fundamental form directly
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_bot, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_top, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();
        // wrapper around solution
        const Eigen::Map<const Eigen::VectorXd> growthRates_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_top(data + nFaces, nFaces);

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_top(grad_ptr + nFaces, nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, growthRates_bot, gradEng_abar_bot, gradEng_growthRate_bot);
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, growthRates_top, gradEng_abar_top, gradEng_growthRate_top);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const override
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const override
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + nFaces, nFaces);
        return dataptr;
    }
};

template<typename tMesh>
class Parametrizer_Bilayer_IsoGrowth_WithBounds : public Parametrizer_Bilayer_IsoGrowth_Base<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:
    LowerUpperBound lowerupper_bot, lowerupper_top;

public:

    Parametrizer_Bilayer_IsoGrowth_WithBounds(tMesh & mesh_in, const Real min_growth, const Real max_growth):
    Parametrizer_Bilayer_IsoGrowth_Base<tMesh>(mesh_in),
    lowerupper_bot(mesh_in.getNumberOfFaces(), min_growth, max_growth),
    lowerupper_top(mesh_in.getNumberOfFaces(), min_growth, max_growth)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the growth
        lowerupper_bot.linkParams(data, nFaces);
        lowerupper_top.linkParams(data + nFaces, nFaces);

        // compute the parameters from the thickness
        lowerupper_bot.assign_inverse();
        lowerupper_top.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 2*mesh.getNumberOfFaces(); // two swelling rates
    }

    void updateSolution() override
    {
        // compute the growth rates from the parameters
        lowerupper_bot.assign();
        lowerupper_top.assign();

        // use helper method to assign the first fundamental form directly
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, lowerupper_bot.data, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, lowerupper_top.data, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_top(grad_ptr + nFaces, nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, lowerupper_bot.data, gradEng_abar_bot, gradEng_growthRate_bot);
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, lowerupper_top.data, gradEng_abar_top, gradEng_growthRate_top);

        // apply chain rule
        lowerupper_bot.applyChainRule(gradEng_growthRate_bot);
        lowerupper_top.applyChainRule(gradEng_growthRate_top);

    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const override
    {
        return lowerupper_bot.data;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const override
    {
        return lowerupper_top.data;
    }
};




template<typename tMesh>
class Parametrizer_Bilayer_IsoOrthoGrowth_WithBounds : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:
    const Real min_growth_abs, max_growth_abs;
    LowerUpperBound lowerupper_AR, lowerupper_s_bot, lowerupper_s_top;

    void computeGrowthRates(const Eigen::Ref<const Eigen::VectorXd> aspectRatio, const Eigen::Ref<const Eigen::VectorXd> scaleFac, Eigen::VectorXd & growthRates_1, Eigen::VectorXd & growthRates_2) const
    {
        const int nFaces = aspectRatio.rows();

        assert(scaleFac.rows() == nFaces);
        assert(growthRates_1.rows() == nFaces);
        assert(growthRates_2.rows() == nFaces);

        for(int i=0;i<nFaces;++i)
        {
            growthRates_1(i) = (1.0 - scaleFac(i)) * min_growth_abs * aspectRatio(i) + scaleFac(i) * max_growth_abs - 1; // growth_1 is the larger one
            growthRates_2(i) = (1.0 - scaleFac(i)) * min_growth_abs + scaleFac(i) * max_growth_abs / aspectRatio(i) - 1; // growth_2 is the smaller one
            // we subtract -1 to make it relative again : but the aspect ratio identity (s1/s2 = AR) holds for the absolute values of growth
        }
    }

public:

    Parametrizer_Bilayer_IsoOrthoGrowth_WithBounds(tMesh & mesh_in, const Real min_growth_in, const Real max_growth_in):
    Parametrizer<tMesh>(mesh_in),
    min_growth_abs(min_growth_in+1), // always > 0
    max_growth_abs(max_growth_in+1), // always > 0
    lowerupper_AR(mesh_in.getNumberOfFaces(), std::max(1.0, min_growth_abs/max_growth_abs), max_growth_abs/min_growth_abs), // s2 < s1 --> AR >= 1
    lowerupper_s_bot(mesh_in.getNumberOfFaces(), 0, 1),
    lowerupper_s_top(mesh_in.getNumberOfFaces(), 0, 1)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the growth
        lowerupper_AR.linkParams(data + nFaces, nFaces);
        lowerupper_s_bot.linkParams(data + 2*nFaces, nFaces);
        lowerupper_s_top.linkParams(data + 3*nFaces, nFaces);

        // compute the parameters from the initial condition
        lowerupper_AR.assign_inverse();
        lowerupper_s_bot.assign_inverse();
        lowerupper_s_top.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 4*mesh.getNumberOfFaces(); // one layer isotropic, other layer anisotropic
    }

    void updateSolution() override
    {
        // compute the values from the parameters
        lowerupper_AR.assign();
        lowerupper_s_bot.assign();
        lowerupper_s_top.assign();

        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles(data, nFaces);
        const Eigen::Ref<const Eigen::VectorXd> growthRates_AR(lowerupper_AR.data);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac_bot(lowerupper_s_bot.data);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac_top(lowerupper_s_top.data);

        Eigen::VectorXd growthRates_1_bot(nFaces), growthRates_2_bot(nFaces);
        Eigen::VectorXd growthRates_1_top(nFaces), growthRates_2_top(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_bot, growthRates_1_bot, growthRates_2_bot);
        computeGrowthRates(growthRates_AR, scaleFac_top, growthRates_1_top, growthRates_2_top);

        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_bot, growthRates_2_bot, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles, growthRates_1_top, growthRates_2_top, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // create a wrapper around the solution pointer
        const Eigen::Map<const Eigen::VectorXd> growthAngles(data, nFaces);
        const Eigen::Ref<const Eigen::VectorXd> growthRates_AR(lowerupper_AR.data);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac_bot(lowerupper_s_bot.data);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac_top(lowerupper_s_top.data);

        Eigen::VectorXd growthRates_1_bot(nFaces), growthRates_2_bot(nFaces);
        Eigen::VectorXd growthRates_1_top(nFaces), growthRates_2_top(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_bot, growthRates_1_bot, growthRates_2_bot);
        computeGrowthRates(growthRates_AR, scaleFac_top, growthRates_1_top, growthRates_2_top);

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRates_AR(grad_ptr + nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_scaleFac_bot(grad_ptr + 2*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_scaleFac_top(grad_ptr + 3*nFaces, nFaces);

        // tmp vectors
        Eigen::VectorXd gradEng_growthRates_1_bot(nFaces), gradEng_growthRates_2_bot(nFaces);
        Eigen::VectorXd gradEng_growthRates_1_top(nFaces), gradEng_growthRates_2_top(nFaces);
        Eigen::VectorXd gradEng_growthAngles_bot(nFaces), gradEng_growthAngles_top(nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles, growthRates_1_bot, growthRates_2_bot, gradEng_abar_bot, gradEng_growthAngles_bot, gradEng_growthRates_1_bot, gradEng_growthRates_2_bot);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles, growthRates_1_top, growthRates_2_top, gradEng_abar_top, gradEng_growthAngles_top, gradEng_growthRates_1_top, gradEng_growthRates_2_top);

        // apply the last chain rule for the actual values we optimize for
        for(int i=0;i<nFaces;++i)
        {
            gradEng_growthAngles(i) = gradEng_growthAngles_bot(i) + gradEng_growthAngles_top(i);

            const Real AR_1_bot = gradEng_growthRates_1_bot(i) * (1.0 - scaleFac_bot(i)) * min_growth_abs;
            const Real AR_2_bot = gradEng_growthRates_2_bot(i) * (-1.0 * scaleFac_bot(i) * max_growth_abs / std::pow(growthRates_AR(i),2));
            const Real AR_1_top = gradEng_growthRates_1_top(i) * (1.0 - scaleFac_top(i)) * min_growth_abs;
            const Real AR_2_top = gradEng_growthRates_2_top(i) * (-1.0 * scaleFac_top(i) * max_growth_abs / std::pow(growthRates_AR(i),2));
            gradEng_growthRates_AR(i) = AR_1_bot + AR_2_bot + AR_1_top + AR_2_top;

            const Real scaleFac_1_bot = gradEng_growthRates_1_bot(i) * (- min_growth_abs * growthRates_AR(i) + max_growth_abs);
            const Real scaleFac_2_bot = gradEng_growthRates_2_bot(i) * (- min_growth_abs + max_growth_abs / growthRates_AR(i));
            const Real scaleFac_1_top = gradEng_growthRates_1_top(i) * (- min_growth_abs * growthRates_AR(i) + max_growth_abs);
            const Real scaleFac_2_top = gradEng_growthRates_2_top(i) * (- min_growth_abs + max_growth_abs / growthRates_AR(i));

            gradEng_scaleFac_bot(i) = scaleFac_1_bot + scaleFac_2_bot;
            gradEng_scaleFac_top(i) = scaleFac_1_top + scaleFac_2_top;
        }

        // apply chain rule for the wrapper function
        lowerupper_AR.applyChainRule(gradEng_growthRates_AR);
        lowerupper_s_bot.applyChainRule(gradEng_scaleFac_bot);
        lowerupper_s_top.applyChainRule(gradEng_scaleFac_top);

        /*
         * to explain the last chain rules:
         * we have E = f( abar_b(s1b, s2b, alphab), abar_t(s1t, s2t, alphat) )
         * where s1t = s * (s1b + 1) - 1 and s2t = s * (s2b + 1) - 1 ; alphat = alphab
         * then :
         * grad_s1b[E] = df/dabar_b * dabar_b/ds1b + df/dabar_t * dabar_t/ds1t * ds1t/ds1b = gradEng_growthRates_1_bot + gradEng_growthRates_1_top * s
         * since ds1t / ds1b = s
         * same for grad_s2b
         *
         * grad_alphab[E] = df/dabar_b * dabar_b/dalphab + df/dabar_t * dabar_t/dalphat * dalphat / dalphab = gradEng_growthAngles_bot + gradEng_growthAngles_top * 1
         * since dalphat / dalphab = 1
         *
         * finally
         * grad_s[E] = df/dabar_t * ( dabar_t / ds1t * ds1t / ds  + dabar_t / ds2t * ds2t / ds) = gradEng_growthRates_1_top * (growthRates_1_bot + 1)+ gradEng_growthRates_2_top * (growthRates_2_bot + 1)
         * since ds1t/ds = (s1b + 1) and ds2t/ds = (s2b + 1)

         */
    }

    const Eigen::Ref<const Eigen::VectorXd> getScaleFac_bot() const
    {
        return lowerupper_s_bot.data;
    }

    const Eigen::Ref<const Eigen::VectorXd> getScaleFac_top() const
    {
        return lowerupper_s_top.data;
    }

    const Eigen::Ref<const Eigen::VectorXd> getAspectRatio() const
    {
        return lowerupper_AR.data;
    }

    const Eigen::VectorXd getGrowthAngles_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces); // same as bot
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces); // same as bot
        return dataptr;
    }

    const Eigen::VectorXd getGrowthRates_1_bot() const
    {
        const auto scaleFac_bot = getScaleFac_bot();
        const auto growthRates_AR = getAspectRatio();
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd growthRates_1_bot(nFaces), growthRates_2_bot(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_bot, growthRates_1_bot, growthRates_2_bot);
        return growthRates_1_bot;
    }

    const Eigen::VectorXd getGrowthRates_2_bot() const
    {
        const auto scaleFac_bot = getScaleFac_bot();
        const auto growthRates_AR = getAspectRatio();
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd growthRates_1_bot(nFaces), growthRates_2_bot(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_bot, growthRates_1_bot, growthRates_2_bot);
        return growthRates_2_bot;
    }

    const Eigen::VectorXd getGrowthRates_1_top() const
    {
        const auto scaleFac_top = getScaleFac_top();
        const auto growthRates_AR = getAspectRatio();
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd growthRates_1_top(nFaces), growthRates_2_top(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_top, growthRates_1_top, growthRates_2_top);
        return growthRates_1_top;
    }

    const Eigen::VectorXd getGrowthRates_2_top() const
    {
        const auto scaleFac_top = getScaleFac_top();
        const auto growthRates_AR = getAspectRatio();
        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd growthRates_1_top(nFaces), growthRates_2_top(nFaces);
        computeGrowthRates(growthRates_AR, scaleFac_top, growthRates_1_top, growthRates_2_top);
        return growthRates_2_top;
    }
};



template<typename tMesh>
class Parametrizer_Bilayer_OrthoGrowth_Angles : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

    const Real s1, s2;
public:

    Parametrizer_Bilayer_OrthoGrowth_Angles(tMesh & mesh_in, const Real s1_in, const Real s2_in):
    Parametrizer<tMesh>(mesh_in),
    s1(s1_in),
    s2(s2_in)
    {
    }

    int getNumberOfVariables() const override
    {
        return 2*mesh.getNumberOfFaces(); // two layers, each layer has one angle
    }

    void updateSolution() override
    {
        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + nFaces, nFaces);

        // use helper method to assign the first fundamental form directly
        const Eigen::VectorXd growthRates_1 = Eigen::VectorXd::Constant(nFaces, s1);
        const Eigen::VectorXd growthRates_2 = Eigen::VectorXd::Constant(nFaces, s2);

        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_bot, growthRates_1, growthRates_2, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_top, growthRates_1, growthRates_2, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();
        // wrapper around solution
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + nFaces, nFaces);

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_top(grad_ptr + nFaces, nFaces);

        const Eigen::VectorXd growthRates_1 = Eigen::VectorXd::Constant(nFaces, s1);
        const Eigen::VectorXd growthRates_2 = Eigen::VectorXd::Constant(nFaces, s2);
        Eigen::VectorXd gradEng_growthRates_1_dummy(nFaces), gradEng_growthRates_2_dummy(nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_bot, growthRates_1, growthRates_2, gradEng_abar_bot, gradEng_growthAngles_bot, gradEng_growthRates_1_dummy, gradEng_growthRates_2_dummy);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_top, growthRates_1, growthRates_2, gradEng_abar_top, gradEng_growthAngles_top, gradEng_growthRates_1_dummy, gradEng_growthRates_2_dummy);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + nFaces, nFaces);
        return dataptr;
    }
};


template<typename tMesh>
class Parametrizer_Bilayer_OrthoGrowth_Angles_Rescale : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

    const Real s1_over_s2;
    LowerUpperBound lowerupper_scalefac;

public:

    Parametrizer_Bilayer_OrthoGrowth_Angles_Rescale(tMesh & mesh_in, const Real s1_over_s2_in, const Real min_growth = -1, const Real max_growth = +1):
    Parametrizer<tMesh>(mesh_in),
    s1_over_s2(s1_over_s2_in),
    lowerupper_scalefac(1, min_growth, max_growth)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the lowerupper container -- the rest stays unchanged
        lowerupper_scalefac.linkParams(data + 2*nFaces, 1);

        // compute the parameters from the initial condition
        lowerupper_scalefac.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 2*mesh.getNumberOfFaces() + 1; // two layers, each layer has one angle -- plus one scale factor (lets say the scale factor determines s1)
    }

    void updateSolution() override
    {
        lowerupper_scalefac.assign();

        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + nFaces, nFaces);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac(lowerupper_scalefac.data);

        // use helper method to assign the first fundamental form directly
        const Eigen::VectorXd growthRates_1 = Eigen::VectorXd::Constant(nFaces, scaleFac(0));
        const Eigen::VectorXd growthRates_2 = Eigen::VectorXd::Constant(nFaces, scaleFac(0)/s1_over_s2);

        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_bot, growthRates_1, growthRates_2, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_top, growthRates_1, growthRates_2, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();
        // wrapper around solution
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + nFaces, nFaces);
        const Eigen::Ref<const Eigen::VectorXd> scaleFac(lowerupper_scalefac.data);

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_top(grad_ptr + nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthScaleFac(grad_ptr + 2*nFaces, 1);

        const Eigen::VectorXd growthRates_1 = Eigen::VectorXd::Constant(nFaces, scaleFac(0));
        const Eigen::VectorXd growthRates_2 = Eigen::VectorXd::Constant(nFaces, scaleFac(0)/s1_over_s2);
        Eigen::VectorXd gradEng_growthRates_1_bot(nFaces), gradEng_growthRates_2_bot(nFaces);
        Eigen::VectorXd gradEng_growthRates_1_top(nFaces), gradEng_growthRates_2_top(nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_bot, growthRates_1, growthRates_2, gradEng_abar_bot, gradEng_growthAngles_bot, gradEng_growthRates_1_bot, gradEng_growthRates_2_bot);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_top, growthRates_1, growthRates_2, gradEng_abar_top, gradEng_growthAngles_top, gradEng_growthRates_1_top, gradEng_growthRates_2_top);

        // finally we compute the gradient with respect to the scale factor 's': ds1_b/ds = ds1_t/ds = 1 ; ds2_b/ds = ds2_t/ds = 1/s1_over_s1 (since s1 = s, s2 = s/s1_over_s2) --> sum over all faces
        gradEng_growthScaleFac(0) = 0;
        for(int i=0;i<nFaces;++i)
        {
            gradEng_growthScaleFac(0) += gradEng_growthRates_1_bot(i) + gradEng_growthRates_1_top(i) + 1/(s1_over_s2) * (gradEng_growthRates_2_bot(i) + gradEng_growthRates_2_top(i));
        }

        lowerupper_scalefac.applyChainRule(gradEng_growthScaleFac);
    }

    const Eigen::VectorXd getGrowthRates_1() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::VectorXd growthRates_1 = Eigen::VectorXd::Constant(nFaces, lowerupper_scalefac.data(0));
        return growthRates_1;
    }

    const Eigen::VectorXd getGrowthRates_2() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> scaleFac(data + 2*nFaces, 1);
        const Eigen::VectorXd growthRates_2 = Eigen::VectorXd::Constant(nFaces, lowerupper_scalefac.data(0)/s1_over_s2);
        return growthRates_2;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + nFaces, nFaces);
        return dataptr;
    }
};

template<typename tMesh>
class Parametrizer_Bilayer_OrthoGrowth : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

public:

    Parametrizer_Bilayer_OrthoGrowth(tMesh & mesh_in):
    Parametrizer<tMesh>(mesh_in)
    {
    }

    int getNumberOfVariables() const override
    {
        return 6*mesh.getNumberOfFaces(); // two layers, each layer has two rates and one angle
    }

    void updateSolution() override
    {
        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_1_bot(data + nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_2_bot(data + 2*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_1_top(data + 4*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_2_top(data + 5*nFaces, nFaces);

        // use helper method to assign the first fundamental form directly
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_bot, growthRates_1_bot, growthRates_2_bot, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_top, growthRates_1_top, growthRates_2_top, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();
        // wrapper around solution
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_1_bot(data + nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_2_bot(data + 2*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_1_top(data + 4*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_2_top(data + 5*nFaces, nFaces);

        // wrapper around abar
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 3*nFaces, nFaces, 3);

        // wrapper around gradient
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRates_1_bot(grad_ptr + nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRates_2_bot(grad_ptr + 2*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngles_top(grad_ptr + 3*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRates_1_top(grad_ptr + 4*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRates_2_top(grad_ptr + 5*nFaces, nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_bot, growthRates_1_bot, growthRates_2_bot, gradEng_abar_bot, gradEng_growthAngles_bot, gradEng_growthRates_1_bot, gradEng_growthRates_2_bot);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_top, growthRates_1_top, growthRates_2_top, gradEng_abar_top, gradEng_growthAngles_top, gradEng_growthRates_1_top, gradEng_growthRates_2_top);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_1_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + nFaces, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_2_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + 2*nFaces, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + 3*nFaces, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_1_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + 4*nFaces, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_2_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + 5*nFaces, nFaces);
        return dataptr;
    }
};



template<typename tMesh>
class Parametrizer_BiLayer_IsoGrowth_WithYoungBound : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:
    MaterialProperties_Iso_Array & material_properties_bot;
    MaterialProperties_Iso_Array & material_properties_top;
    LowerUpperBound lowerupper_bot_Young;
    LowerUpperBound lowerupper_top_Young;

public:

    Parametrizer_BiLayer_IsoGrowth_WithYoungBound(tMesh & mesh_in, MaterialProperties_Iso_Array & material_properties_bot_in, MaterialProperties_Iso_Array & material_properties_top_in, const Real min_Young, const Real max_Young):
    Parametrizer<tMesh>(mesh_in),
    material_properties_bot(material_properties_bot_in),
    material_properties_top(material_properties_top_in),
    lowerupper_bot_Young(mesh_in.getNumberOfFaces(), min_Young, max_Young),
    lowerupper_top_Young(mesh_in.getNumberOfFaces(), min_Young, max_Young)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the Young
        lowerupper_bot_Young.linkParams(data + 2*nFaces, nFaces);
        lowerupper_top_Young.linkParams(data + 3*nFaces, nFaces);

        // compute the parameters from the Youngs values
        lowerupper_bot_Young.assign_inverse();
        lowerupper_top_Young.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 4*mesh.getNumberOfFaces(); // two swelling rate and two Young per face
    }

    void updateSolution() override
    {
        // compute the Young's modulus from the parameters
        lowerupper_bot_Young.assign();
        lowerupper_top_Young.assign();

        // create a wrapper around the solution pointer
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthRates_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_top(data + nFaces, nFaces);

        // use helper method to assign the first fundamental form directly
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_bot, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_top, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());

        // copy the Young's moduli
        for(int i=0;i<nFaces;++i)
        {
            material_properties_bot.materials[i].Young = lowerupper_bot_Young.data(i);
            material_properties_top.materials[i].Young = lowerupper_top_Young.data(i);
        }
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // wrapper around solution
        const Eigen::Map<const Eigen::VectorXd> growthRates_bot(data, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthRates_top(data + nFaces, nFaces);

        // wrapper around abar
        // energygradient --> abarbot (3n), youngbot (n), abartop (3n), youngtop (n)
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_bot(energyGradient.data() + 3*nFaces, nFaces);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 4*nFaces, nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_top(energyGradient.data() + 7*nFaces, nFaces);

        // wrapper around gradient : order should be growth_bot, growth_top, young_bot, young_top
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_top(grad_ptr + nFaces, nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, growthRates_bot, gradEng_abar_bot, gradEng_growthRate_bot);
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, growthRates_top, gradEng_abar_top, gradEng_growthRate_top);

        Eigen::Map<Eigen::VectorXd> gradEng_bot_Young(grad_ptr + 2*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_top_Young(grad_ptr + 3*nFaces, nFaces);

        gradEng_bot_Young = gradEng_Y_bot;
        gradEng_top_Young = gradEng_Y_top;

        // chain rule for the Young's modulus (bound parametres)
        lowerupper_bot_Young.applyChainRule(gradEng_bot_Young);
        lowerupper_top_Young.applyChainRule(gradEng_top_Young);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data, nFaces);
        return dataptr;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> dataptr(data + nFaces, nFaces);
        return dataptr;
    }

};


template<typename tMesh>
class Parametrizer_BiLayer_IsoGrowthYoung_WithBounds : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:
    MaterialProperties_Iso_Array & material_properties_bot;
    MaterialProperties_Iso_Array & material_properties_top;
    LowerUpperBound lowerupper_bot_iso;
    LowerUpperBound lowerupper_top_iso;
    LowerUpperBound lowerupper_bot_Young;
    LowerUpperBound lowerupper_top_Young;

public:

    Parametrizer_BiLayer_IsoGrowthYoung_WithBounds(tMesh & mesh_in, MaterialProperties_Iso_Array & material_properties_bot_in, MaterialProperties_Iso_Array & material_properties_top_in, const Real min_growth, const Real max_growth, const Real min_Young, const Real max_Young):
    Parametrizer<tMesh>(mesh_in),
    material_properties_bot(material_properties_bot_in),
    material_properties_top(material_properties_top_in),
    lowerupper_bot_iso(mesh_in.getNumberOfFaces(), min_growth, max_growth),
    lowerupper_top_iso(mesh_in.getNumberOfFaces(), min_growth, max_growth),
    lowerupper_bot_Young(mesh_in.getNumberOfFaces(), min_Young, max_Young),
    lowerupper_top_Young(mesh_in.getNumberOfFaces(), min_Young, max_Young)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the growth
        lowerupper_bot_iso.linkParams(data + 0*nFaces, nFaces);
        lowerupper_top_iso.linkParams(data + 1*nFaces, nFaces);

        // compute the parameters from the growth values
        lowerupper_bot_iso.assign_inverse();
        lowerupper_top_iso.assign_inverse();

        // link the data array to the Young
        lowerupper_bot_Young.linkParams(data + 2*nFaces, nFaces);
        lowerupper_top_Young.linkParams(data + 3*nFaces, nFaces);

        // compute the parameters from the Youngs values
        lowerupper_bot_Young.assign_inverse();
        lowerupper_top_Young.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 4*mesh.getNumberOfFaces(); // two swelling rate and two Young per face
    }

    void updateSolution() override
    {
        // compute the growth from the parameters
        lowerupper_bot_iso.assign();
        lowerupper_top_iso.assign();

        // compute the Young's modulus from the parameters
        lowerupper_bot_Young.assign();
        lowerupper_top_Young.assign();

        // use helper method to assign the first fundamental form directly
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, lowerupper_bot_iso.data, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());
        GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, lowerupper_top_iso.data, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());

        // copy the Young's moduli
        const int nFaces = mesh.getNumberOfFaces();
        for(int i=0;i<nFaces;++i)
        {
            material_properties_bot.materials[i].Young = lowerupper_bot_Young.data(i);
            material_properties_top.materials[i].Young = lowerupper_top_Young.data(i);
        }
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();


        // wrapper around abar
        // energygradient --> abarbot (3n), youngbot (n), abartop (3n), youngtop (n)
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_bot(energyGradient.data() + 3*nFaces, nFaces);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 4*nFaces, nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_top(energyGradient.data() + 7*nFaces, nFaces);

        // wrapper around gradient : order should be growth_bot, growth_top, young_bot, young_top
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_bot(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthRate_top(grad_ptr + nFaces, nFaces);

        // chain rule : propagate grad rate into grad_ptr
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, lowerupper_bot_iso.data, gradEng_abar_bot, gradEng_growthRate_bot);
        GrowthHelper<tMesh>::computeAbarsIsoGrowth_Gradient(mesh, lowerupper_top_iso.data, gradEng_abar_top, gradEng_growthRate_top);

        // chain rule for the growth rates (bound parametres)
        lowerupper_bot_iso.applyChainRule(gradEng_growthRate_bot);
        lowerupper_top_iso.applyChainRule(gradEng_growthRate_top);

        // wrapper around gradient for Young's moduli
        Eigen::Map<Eigen::VectorXd> gradEng_bot_Young(grad_ptr + 2*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_top_Young(grad_ptr + 3*nFaces, nFaces);

        gradEng_bot_Young = gradEng_Y_bot;
        gradEng_top_Young = gradEng_Y_top;

        // chain rule for the Young's modulus (bound parametres)
        lowerupper_bot_Young.applyChainRule(gradEng_bot_Young);
        lowerupper_top_Young.applyChainRule(gradEng_top_Young);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_bot() const
    {
        return lowerupper_bot_iso.data;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthRates_top() const
    {
        return lowerupper_top_iso.data;
    }

};



template<typename tMesh>
class Parametrizer_BiLayer_IsoGrowthYoungRatio : public Parametrizer<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:
    MaterialProperties_Iso_Array & material_properties_bot;
    MaterialProperties_Iso_Array & material_properties_top;
    const Real iso_growth;
    const Real ortho_growth_1;
    const Real ortho_growth_2;
    const Real Young_ref;

    LowerUpperBound lowerupper_bot_iso;// moving from isotropic (0) to orthotropic (1)
    LowerUpperBound lowerupper_top_iso;
    LowerUpperBound lowerupper_Young_ratio; // ratio of Young's modulus in top layer vs bottom layer

public:

    Parametrizer_BiLayer_IsoGrowthYoungRatio(tMesh & mesh_in, MaterialProperties_Iso_Array & material_properties_bot_in, MaterialProperties_Iso_Array & material_properties_top_in, const Real iso_growth_in, const Real ortho_growth_1_in, const Real ortho_growth_2_in, const Real Young_ref_in, const Real min_Young_ratio_in, const Real max_Young_ratio_in):
    Parametrizer<tMesh>(mesh_in),
    material_properties_bot(material_properties_bot_in),
    material_properties_top(material_properties_top_in),
    iso_growth(iso_growth_in),
    ortho_growth_1(ortho_growth_1_in),
    ortho_growth_2(ortho_growth_2_in),
    Young_ref(Young_ref_in),
    lowerupper_bot_iso(mesh_in.getNumberOfFaces(), 0, 1),
    lowerupper_top_iso(mesh_in.getNumberOfFaces(), 0, 1),
    lowerupper_Young_ratio(mesh_in.getNumberOfFaces(), std::sqrt(min_Young_ratio_in), std::sqrt(max_Young_ratio_in))
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        // solution data contains:
        // bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio
        // Young_ratio = Young_top / Young_bottom

        const int nFaces = mesh.getNumberOfFaces();

        // initialize the parameters array
        Parametrizer<tMesh>::initSolution(initVals, reAllocate);

        // link the data array to the growth
        lowerupper_bot_iso.linkParams(data + 0*nFaces, nFaces);
        // 1*nFaces = bottom angle
        lowerupper_top_iso.linkParams(data + 2*nFaces, nFaces);
        // 3*nFaces = top angle
        lowerupper_Young_ratio.linkParams(data + 4*nFaces, nFaces);

        // compute the parameters from the initial values
        lowerupper_bot_iso.assign_inverse();
        lowerupper_top_iso.assign_inverse();
        lowerupper_Young_ratio.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 5*mesh.getNumberOfFaces(); // per face : two angles and two AR (each layer), and one Young ratio
    }

    virtual void computeGrowthFactors(const Eigen::Ref<const Eigen::VectorXd> lowerupper_iso, Eigen::VectorXd & growthFactors_1, Eigen::VectorXd & growthFactors_2) const
    {
        const int nFaces = mesh.getNumberOfFaces();
        growthFactors_1.resize(nFaces);
        growthFactors_2.resize(nFaces);

        for(int i=0;i<nFaces;++i)
        {
            const Real aspectRatio = lowerupper_iso(i);
            growthFactors_1(i) = iso_growth + (ortho_growth_1 - iso_growth) * aspectRatio;
            growthFactors_2(i) = iso_growth + (ortho_growth_2 - iso_growth) * aspectRatio;
        }
    }

    void updateSolution() override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // compute the values from the parameters
        lowerupper_bot_iso.assign();
        lowerupper_top_iso.assign();
        lowerupper_Young_ratio.assign();

        // get references to the angles
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data + 1*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);

        // allocate data for the growth factors
        Eigen::VectorXd growthFactors_1, growthFactors_2;
        // compute bottom layer growth factors and assign
        computeGrowthFactors(lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_bot, growthFactors_1, growthFactors_2, mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>());

        // same for top layer
        computeGrowthFactors(lowerupper_top_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth(mesh, growthAngles_top, growthFactors_1, growthFactors_2, mesh.getRestConfiguration().template getFirstFundamentalForms<top>());


        // assign the Young's moduli
        for(int i=0;i<nFaces;++i)
        {
            const Real Young_ratio = lowerupper_Young_ratio.data(i);
            material_properties_bot.materials[i].Young = Young_ref * Young_ratio;
            material_properties_top.materials[i].Young = Young_ref / Young_ratio;
        }
    }

    void updateGradient(const int , const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        const int nFaces = mesh.getNumberOfFaces();

        // get solution
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data + 1*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);

        // wrapper around abar
        // energygradient --> abarbot (3n), youngbot (n), abartop (3n), youngtop (n)
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_bot(energyGradient.data() + 3*nFaces, nFaces);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 4*nFaces, nFaces, 3);
        const Eigen::Map<const Eigen::VectorXd> gradEng_Y_top(energyGradient.data() + 7*nFaces, nFaces);

        // wrapper around gradient :  bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio

        Eigen::Map<Eigen::VectorXd> gradEng_bot_iso(grad_ptr, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngle_bot(grad_ptr + nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_top_iso(grad_ptr + 2*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_growthAngle_top(grad_ptr + 3*nFaces, nFaces);
        Eigen::Map<Eigen::VectorXd> gradEng_YoungRatio(grad_ptr + 4*nFaces, nFaces);

        // chain rule : compute the gradient wrt orthotropic growth factors (and angle) for bot layer
        {
            // first (re)compute the solution
            Eigen::VectorXd growthFactors_1, growthFactors_2;

            // compute bottom layer growth factors
            computeGrowthFactors(lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);

            // allocate space for gradient
            Eigen::VectorXd gradEng_growthFac_1(nFaces), gradEng_growthFac_2(nFaces);

            GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_bot, growthFactors_1, growthFactors_2, gradEng_abar_bot, gradEng_growthAngle_bot, gradEng_growthFac_1, gradEng_growthFac_2);

            // chain rule (by hand)
            const Real prefac_1 = (ortho_growth_1 - iso_growth);
            const Real prefac_2 = (ortho_growth_2 - iso_growth);
            for(int i=0;i<nFaces;++i)
            {
                gradEng_bot_iso(i) = gradEng_growthFac_1(i) * prefac_1 + gradEng_growthFac_2(i) * prefac_2;
            }

            // apply chain rule for parameterization
            lowerupper_bot_iso.applyChainRule(gradEng_bot_iso);

            //// SAME EXACT THING FOR TOP LAYER (without reallocating arrays)

            // compute top layer growth factors
            computeGrowthFactors(lowerupper_top_iso.data, growthFactors_1, growthFactors_2);

            GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_top, growthFactors_1, growthFactors_2, gradEng_abar_top, gradEng_growthAngle_top, gradEng_growthFac_1, gradEng_growthFac_2);

            // chain rule (by hand)
            for(int i=0;i<nFaces;++i)
            {
                gradEng_top_iso(i) = gradEng_growthFac_1(i) * prefac_1 + gradEng_growthFac_2(i) * prefac_2;
            }

            // apply chain rule for parameterization
            lowerupper_top_iso.applyChainRule(gradEng_top_iso);
        }

        // now the Young's modulus
        for(int i=0;i<nFaces;++i)
        {
            material_properties_bot.materials[i].Young = Young_ref * lowerupper_Young_ratio.data(i);
            material_properties_top.materials[i].Young = Young_ref / lowerupper_Young_ratio.data(i);

            gradEng_YoungRatio(i) = Young_ref * ( gradEng_Y_bot(i) - gradEng_Y_top(i) / std::pow(lowerupper_Young_ratio.data(i),2) );
        }
        // chain rule for the Young's modulus (bound parametres)
        lowerupper_Young_ratio.applyChainRule(gradEng_YoungRatio);
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_bot() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data + 1*nFaces, nFaces);
        return growthAngles_bot;
    }

    const Eigen::VectorXd getGrowthRates_1_bot() const
    {
        Eigen::VectorXd growthFactors_1, growthFactors_2;
        computeGrowthFactors(lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);
        return growthFactors_1;
    }

    const Eigen::VectorXd getGrowthRates_2_bot() const
    {
        Eigen::VectorXd growthFactors_1, growthFactors_2;
        computeGrowthFactors(lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);
        return growthFactors_2;
    }

    const Eigen::Ref<const Eigen::VectorXd> getGrowthAngles_top() const
    {
        const int nFaces = mesh.getNumberOfFaces();
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);
        return growthAngles_top;
    }

    const Eigen::VectorXd getGrowthRates_1_top() const
    {
        Eigen::VectorXd growthFactors_1, growthFactors_2;
        computeGrowthFactors(lowerupper_top_iso.data, growthFactors_1, growthFactors_2);
        return growthFactors_1;
    }

    const Eigen::VectorXd getGrowthRates_2_top() const
    {
        Eigen::VectorXd growthFactors_1, growthFactors_2;
        computeGrowthFactors(lowerupper_top_iso.data, growthFactors_1, growthFactors_2);
        return growthFactors_2;
    }
};



template<typename tMesh>
class Parametrizer_BiLayer_IsoGrowthYoungRatioTemperature : public Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

    LowerUpperBound lowerupper_temp; // one temperature for entire mesh

public:

    Parametrizer_BiLayer_IsoGrowthYoungRatioTemperature(tMesh & mesh_in, MaterialProperties_Iso_Array & material_properties_bot_in, MaterialProperties_Iso_Array & material_properties_top_in, const Real iso_growth_in, const Real ortho_growth_1_in, const Real ortho_growth_2_in, const Real Young_ref_in, const Real min_Young_ratio_in, const Real max_Young_ratio_in, const Real minTemp_in, const Real maxTemp_in):
    Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>(mesh_in, material_properties_bot_in, material_properties_top_in, iso_growth_in, ortho_growth_1_in, ortho_growth_2_in, Young_ref_in, min_Young_ratio_in, max_Young_ratio_in),
    lowerupper_temp(1, minTemp_in, maxTemp_in)
    {
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        // solution data contains:
        // bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio, temperature (scalar)
        // Young_ratio = Young_top / Young_bottom

        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::initSolution(initVals, reAllocate);

        const int nFaces = mesh.getNumberOfFaces();
        lowerupper_temp.linkParams(data + 5*nFaces, 1);

        // compute the parameters from the initial values
        lowerupper_temp.assign_inverse();
    }

    int getNumberOfVariables() const override
    {
        return 5*mesh.getNumberOfFaces() + 1; // per face : two angles and two AR (each layer), and one Young ratio. also xy/coordinates of rest configuration
    }

    virtual void computeGrowthFactors(const Eigen::Ref<const Eigen::VectorXd> lowerupper_iso, Eigen::VectorXd & growthFactors_1, Eigen::VectorXd & growthFactors_2) const override
    {
        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::computeGrowthFactors(lowerupper_iso, growthFactors_1, growthFactors_2);
        const Real temperature = lowerupper_temp.data(0);
        growthFactors_1 *= temperature;
        growthFactors_2 *= temperature;
    }

    void updateSolution() override
    {
        lowerupper_temp.assign();
        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::updateSolution();
    }

    void updateGradient(const int nVars, const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        // update the gradient with respect to all (before temperature)
        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::updateGradient(nVars, energyGradient, grad_ptr);

        // correct for temperature
        const int nFaces = mesh.getNumberOfFaces();
        const Real temperature = lowerupper_temp.data(0);

        for(int i=0;i<nFaces;++i)
        {
            grad_ptr[i] *= temperature;
            grad_ptr[i + 2*nFaces] *= temperature;
        }

        // now derivative wrt temperature
        // wrapper around gradient :  bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio, temperature(scalar)
        Real & gradEng_temp = *(grad_ptr + 5*nFaces);
        gradEng_temp = 0.0;

        // wrapper around abar
        // energygradient --> abarbot (3n), youngbot (n), abartop (3n), youngtop (n)
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 4*nFaces, nFaces, 3);

        // compute the gradient wrt growthRates_bot
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data + 1*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);
        Eigen::VectorXd growthFactors_1, growthFactors_2;


        Eigen::VectorXd gradEng_growthAngle(nFaces), gradEng_growthFac_1(nFaces), gradEng_growthFac_2(nFaces);

        // bottom layer factors
        this->computeGrowthFactors(this->lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_bot, growthFactors_1, growthFactors_2, gradEng_abar_bot, gradEng_growthAngle, gradEng_growthFac_1, gradEng_growthFac_2);

        for(int i=0;i<nFaces;++i)
            gradEng_temp += gradEng_growthFac_1(i) * growthFactors_1(i) + gradEng_growthFac_2(i) * growthFactors_2(i);

        // top layer factors
        this->computeGrowthFactors(this->lowerupper_top_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_Gradient(mesh, growthAngles_top, growthFactors_1, growthFactors_2, gradEng_abar_top, gradEng_growthAngle, gradEng_growthFac_1, gradEng_growthFac_2);

        for(int i=0;i<nFaces;++i)
            gradEng_temp += gradEng_growthFac_1(i) * growthFactors_1(i) + gradEng_growthFac_2(i) * growthFactors_2(i);


        gradEng_temp /= temperature;

        Eigen::VectorXd gradEng_temp_vec(1);
        gradEng_temp_vec(0) = gradEng_temp;
        lowerupper_temp.applyChainRule(gradEng_temp_vec);

        // store in the grad_ptr reference
        gradEng_temp = gradEng_temp_vec(0);
        // done
    }
};






template<typename tMesh>
class Parametrizer_BiLayer_IsoGrowthYoungRatioVertices : public Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>
{
public:
    using Parametrizer<tMesh>::mesh; // to avoid having to type this-> all the time
    using Parametrizer<tMesh>::data; // to avoid having to type this-> all the time

protected:

public:

    Parametrizer_BiLayer_IsoGrowthYoungRatioVertices(tMesh & mesh_in, MaterialProperties_Iso_Array & material_properties_bot_in, MaterialProperties_Iso_Array & material_properties_top_in, const Real iso_growth_in, const Real ortho_growth_1_in, const Real ortho_growth_2_in, const Real Young_ref_in, const Real min_Young_ratio_in, const Real max_Young_ratio_in):
    Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>(mesh_in, material_properties_bot_in, material_properties_top_in, iso_growth_in, ortho_growth_1_in, ortho_growth_2_in, Young_ref_in, min_Young_ratio_in, max_Young_ratio_in)
    {
    }


    void assignRestVerticesToMesh()
    {
        const int nFaces = mesh.getNumberOfFaces();
        const int nVertices = mesh.getNumberOfVertices();

        const Eigen::Map<const Eigen::MatrixXd> restvertices_data(data + 5*nFaces, nVertices, 2);
        auto rverts = mesh.getRestConfiguration().getVertices();

        for(int i=0;i<nVertices;++i)
            for(int d=0;d<2;++d)
                rverts(i,d) = restvertices_data(i,d);

        // no need to update if we recompute the abar afterwards
    }

    void initSolution(const Eigen::Ref<const Eigen::VectorXd> initVals, const bool reAllocate = false) override
    {
        // solution data contains:
        // bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio, restvertices_x, restvertices_y
        // Young_ratio = Young_top / Young_bottom

        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::initSolution(initVals, reAllocate);

        // dont need this : data already copied in Parametrizer::initSolution
//        const int nFaces = mesh.getNumberOfFaces();
//        const int nVertices = mesh.getNumberOfVertices();
//        Eigen::Map<Eigen::MatrixXd> restvertices_data(data + 5*nFaces, nVertices, 2);
//        const Eigen::Map<const Eigen::MatrixXd> restvertices_in(initVals.data() + 5*nFaces, nVertices, 2);
//        restvertices_data = restvertices_in;
    }

    int getNumberOfVariables() const override
    {
        return 5*mesh.getNumberOfFaces() + 2*mesh.getNumberOfVertices(); // per face : two angles and two AR (each layer), and one Young ratio. also xy/coordinates of rest configuration
    }

    void updateSolution() override
    {
        // first assign rest vertices
        assignRestVerticesToMesh();

        // then do the update of the rest of the fields
        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::updateSolution();
    }

    void updateGradient(const int nVars, const Eigen::Ref<const Eigen::VectorXd> energyGradient, Real * const grad_ptr) override
    {
        // update the gradient with respect to all non-vertex values
        Parametrizer_BiLayer_IsoGrowthYoungRatio<tMesh>::updateGradient(nVars, energyGradient, grad_ptr);

        // then with respect to the vertices
        const int nFaces = mesh.getNumberOfFaces();
        const int nVertices = mesh.getNumberOfVertices();

        // wrapper around abar
        // energygradient --> abarbot (3n), youngbot (n), abartop (3n), youngtop (n)
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_bot(energyGradient.data(), nFaces, 3);
        const Eigen::Map<const Eigen::MatrixXd> gradEng_abar_top(energyGradient.data() + 4*nFaces, nFaces, 3);

        // wrapper around gradient :  bottom_AR, bottom_angle, top_AR, top_angle, Young_ratio, vertices
        Eigen::Map<Eigen::MatrixXd> gradEng_verts(grad_ptr + 5*nFaces, nVertices, 2);
        gradEng_verts.setZero(); // set to zero because we do plus-equal next

        // also need the angles and factors
        const Eigen::Map<const Eigen::VectorXd> growthAngles_bot(data + 1*nFaces, nFaces);
        const Eigen::Map<const Eigen::VectorXd> growthAngles_top(data + 3*nFaces, nFaces);
        Eigen::VectorXd growthFactors_1, growthFactors_2;

        // bottom layer factors
        this->computeGrowthFactors(this->lowerupper_bot_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_GradientVertices(mesh, growthAngles_bot, growthFactors_1, growthFactors_2, gradEng_abar_bot, gradEng_verts);

        // top layer factors
        this->computeGrowthFactors(this->lowerupper_top_iso.data, growthFactors_1, growthFactors_2);
        GrowthHelper<tMesh>::computeAbarsOrthoGrowth_GradientVertices(mesh, growthAngles_top, growthFactors_1, growthFactors_2, gradEng_abar_top, gradEng_verts);

        // done
    }
};

#endif /* Parametrizer_InverseGrowth_h */
