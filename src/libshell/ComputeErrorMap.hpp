//
//  ComputeErrorMap.hpp
//  Elasticity
//
//  Created by Wim van Rees on 12/28/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef ComputeErrorMap_hpp
#define ComputeErrorMap_hpp

#include "common.hpp"

class ComputeErrorMap
{
protected:
    const bool bDump;

public:

    ComputeErrorMap(const bool bDump = false):
    bDump(bDump)
    {}

    Eigen::VectorXd compute(const Eigen::Ref<const Eigen::MatrixXd> vertices_A, const Eigen::Ref<const Eigen::MatrixXd> vertices_B, const Eigen::Ref<const Eigen::MatrixXi> faces, const Real rescale = false) const;
};

#endif /* ComputeErrorMap_hpp */
