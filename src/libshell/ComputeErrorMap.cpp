//
//  ComputeErrorMap.cpp
//  Elasticity
//
//  Created by Wim van Rees on 12/28/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "ComputeErrorMap.hpp"

//#include <igl/hausdorff.h>
#include <Eigen/Geometry>
#include "WriteVTK.hpp"


Eigen::VectorXd ComputeErrorMap::compute(const Eigen::Ref<const Eigen::MatrixXd> vertices_A, const Eigen::Ref<const Eigen::MatrixXd> vertices_B, const Eigen::Ref<const Eigen::MatrixXi> faces, const Real rescale) const
{
    // compute the least-squares transformation between two point sets
    //https://eigen.tuxfamily.org/dox/group__Geometry__Module.html#gab3f5a82a24490b936f8694cf8fef8e60

    const Eigen::MatrixXd trafo = Eigen::umeyama(vertices_A.transpose(), vertices_B.transpose(), rescale);
    const Eigen::Matrix3d rotmat = trafo.block<3,3>(0,0);
    const Eigen::Vector3d transv = trafo.block<3,1>(0,3);
    // now we have the rotation and scaling matrices to transform vertices_A into vertices_B

    // apply the transform
    const int nVertices = vertices_A.rows();
    Eigen::MatrixXd vertices_A_trafo(nVertices,3);
    for(int i=0;i<nVertices;++i)
    {
        const Eigen::Vector3d vertA = vertices_A.row(i);
        const Eigen::Vector3d vertA_trafo = rotmat * vertA + transv;
        vertices_A_trafo.row(i) = vertA_trafo;
    }

    if(bDump)
    {
        WriteVTK writerA(vertices_A, faces);
        writerA.write("haussdorf_A");

        WriteVTK writerB(vertices_B, faces);
        writerB.write("haussdorf_B");

        WriteVTK writerA_trafo(vertices_A_trafo, faces);
        writerA_trafo.write("haussdorf_A_trafo");
    }

    Eigen::VectorXd dist(nVertices);

    // compute the distance
    for(int i=0;i<nVertices;++i)
    {
      dist(i) = sqrt(pow(vertices_A_trafo(i,0)-vertices_B(i,0),2.)+
                     pow(vertices_A_trafo(i,1)-vertices_B(i,1),2.)+
                     pow(vertices_A_trafo(i,2)-vertices_B(i,2),2.));
    }

    return dist;
}
