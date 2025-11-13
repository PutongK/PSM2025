//
//  ComputeCurvatures.cpp
//  Elasticity
//
//  Created by Wim van Rees on 5/20/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "ComputeCurvatures.hpp"
#include "ExtendedTriangleInfo.hpp"
#include "WriteVTK.hpp"

template<typename tMesh>
void ComputeCurvatures<tMesh>::compute(const tMesh & mesh, Eigen::VectorXd & gauss, Eigen::VectorXd & mean) const
{
    // get current state quantities
    const int nFaces = mesh.getNumberOfFaces();
    const auto & currentState = mesh.getCurrentConfiguration();

    for(int i=0;i<nFaces;++i)
    {
        const ExtendedTriangleInfo & info = currentState.getTriangleInfo(i);

        const Eigen::Matrix2d aform = info.computeFirstFundamentalForm();
        const Eigen::Matrix2d bform = info.computeSecondFundamentalForm();

        const Eigen::Matrix2d shapeOp = aform.inverse() * bform;

        const Real gauss_curv = shapeOp.determinant();
        const Real mean_curv = 0.5*shapeOp.trace();

        gauss(i) = gauss_curv;
        mean(i) = mean_curv;
    }
}

template<typename tMesh>
void ComputeCurvatures<tMesh>::computeNew(const tMesh & mesh, Eigen::VectorXd & gauss, Eigen::VectorXd & mean, Eigen::VectorXd & PrincCurv1, Eigen::VectorXd & PrincCurv2) const
{
    // get current state quantities
    const int nFaces = mesh.getNumberOfFaces();
    const auto & currentState = mesh.getCurrentConfiguration();

    for(int i=0;i<nFaces;++i)
    {
        const ExtendedTriangleInfo & info = currentState.getTriangleInfo(i);

        const Eigen::Matrix2d aform = info.computeFirstFundamentalForm();
        const Eigen::Matrix2d bform = info.computeSecondFundamentalForm();

        const Eigen::Matrix2d shapeOp = aform.inverse() * bform;

        const Eigen::VectorXcd eigenv1 = shapeOp.eigenvalues();
        Eigen::VectorXd eigenv = eigenv1.real();

        if (std::abs(eigenv(1))>std::abs(eigenv(0))){
          Real temp = eigenv(0);
          eigenv(0) = eigenv(1);
          eigenv(1) = temp;
        }

        //std::cout << "\n"<< eigenv << "\nolol" << std::endl;

        PrincCurv1(i) = eigenv(0);
        PrincCurv2(i) = eigenv(1);

        const Real gauss_curv = shapeOp.determinant();
        const Real mean_curv = 0.5*shapeOp.trace();

        gauss(i) = gauss_curv;
        mean(i) = mean_curv;
    }
}




template<typename tMesh>
void ComputeCurvatures<tMesh>::computeDir(const tMesh & mesh, Eigen::VectorXd & gauss, Eigen::VectorXd & mean, Eigen::VectorXd & PrincCurv1, Eigen::VectorXd & PrincCurv2, Eigen::VectorXd & CurvX, Eigen::VectorXd & CurvY, Eigen::Vector3d & NewDir1, Eigen::Vector3d & NewDir2) const
{
    // get current state quantities
    const int nFaces = mesh.getNumberOfFaces();
    const auto & currentState = mesh.getCurrentConfiguration();

    Eigen::MatrixXd ProjNewDir1_Arr(nFaces,3); //for plotting
    Eigen::MatrixXd ProjNewDir2_Arr(nFaces,3);

    for(int i=0;i<nFaces;++i)
    {
        const ExtendedTriangleInfo & info = currentState.getTriangleInfo(i);

        const Eigen::Matrix2d aform = info.computeFirstFundamentalForm();
        const Eigen::Matrix2d bform = info.computeSecondFundamentalForm();

        const Eigen::Matrix2d shapeOp = aform.inverse() * bform;

        const Eigen::VectorXcd eigenval_comp = shapeOp.eigenvalues();
        Eigen::VectorXd eigenval = eigenval_comp.real();

        const Eigen::EigenSolver<Eigen::Matrix2d> es(shapeOp);
        const Eigen::Matrix2cd eigenvec_comp = es.eigenvectors();
        const Eigen::Matrix2d eigenvec = eigenvec_comp.real();

        //project vectors NewDir1 and NewDir2 onto the plane of our triangle
        //use v2 as point of reference
        Eigen::Vector3d PointX = (Eigen::Vector3d() <<  (info.v2(0)+NewDir1(0)), (info.v2(1)+NewDir1(1)), (info.v2(2)+NewDir1(2))).finished();
        Eigen::Vector3d PointY = (Eigen::Vector3d() <<  (info.v2(0)+NewDir2(0)), (info.v2(1)+NewDir2(1)), (info.v2(2)+NewDir2(2))).finished();

        const Real A = info.face_normal(0);
        const Real B = info.face_normal(1);
        const Real C = info.face_normal(2);
        const Real D = -(info.face_normal).dot(info.v2);

        Eigen::Vector3d ProjPointX = PointX;
        ProjPointX(2) = -(A*ProjPointX(0) + B*ProjPointX(1) + D)/C;
        Eigen::Vector3d ProjPointY = PointY;
        ProjPointX(2) = -(A*ProjPointY(0) + B*ProjPointY(1) + D)/C;

        Eigen::Vector3d ProjNewDir1 = (ProjPointX - info.v2);
        Eigen::Vector3d ProjNewDir2 = (ProjPointY - info.v2);
        ProjNewDir1 /= ProjNewDir1.norm();
        ProjNewDir2 /= ProjNewDir2.norm();

        for(int j=0;j<3;++j){
          ProjNewDir1_Arr(i,j) = ProjNewDir1(j);
          ProjNewDir2_Arr(i,j) = ProjNewDir2(j);
        }

        //vectors along princ curv directions in the normal system of coords
        Eigen::Vector3d eigenvec_abs_1;
        Eigen::Vector3d eigenvec_abs_2;
        for(int j=0;j<3;++j)
        {
          eigenvec_abs_1(j) = eigenvec(0,0)*info.e1(j)+eigenvec(1,0)*info.e2(j);
          eigenvec_abs_2(j) = eigenvec(0,1)*info.e1(j)+eigenvec(1,1)*info.e2(j);
        }
        eigenvec_abs_1 /= eigenvec_abs_1.norm();
        eigenvec_abs_2 /= eigenvec_abs_2.norm();

        const Real cosX = eigenvec_abs_1(0)*ProjNewDir1(0) + eigenvec_abs_1(1)*ProjNewDir1(1) + eigenvec_abs_1(2)*ProjNewDir1(2); // scalar product
        const Real sinX = std::sqrt(1.0-std::pow(cosX,2));
        const Real cosY = eigenvec_abs_1(0)*ProjNewDir2(0) + eigenvec_abs_1(1)*ProjNewDir2(1) + eigenvec_abs_1(2)*ProjNewDir2(2);
        const Real sinY = std::sqrt(1.0-std::pow(cosY,2));

        CurvX(i) = eigenval(0)*std::pow(cosX,2) + eigenval(1)*std::pow(sinX,2); //Euler formula
        CurvY(i) = eigenval(0)*std::pow(cosY,2) + eigenval(1)*std::pow(sinY,2);

        if (std::abs(eigenval(1))>std::abs(eigenval(0))){
          Real temp = eigenval(0);
          eigenval(0) = eigenval(1);
          eigenval(1) = temp;
        }
        PrincCurv1(i) = eigenval(0);
        PrincCurv2(i) = eigenval(1);

        const Real gauss_curv = shapeOp.determinant();
        const Real mean_curv = 0.5*shapeOp.trace();

        gauss(i) = gauss_curv;
        mean(i) = mean_curv;
    }

    const auto face2vertices = mesh.getTopology().getFace2Vertices();
    const auto vertices = mesh.getCurrentConfiguration().getVertices();
    WriteVTK writer(vertices, face2vertices);
    writer.addVectorFieldToFaces(ProjNewDir1_Arr, "ProjNewDir1");
    writer.addVectorFieldToFaces(ProjNewDir2_Arr, "ProjNewDir2");
    writer.write("CurvatureDirections.vtp");
}




template<typename tMesh>
void ComputeCurvatures<tMesh>::computePrincGrowthDir(const tMesh & mesh, Eigen::VectorXd & gauss, Eigen::VectorXd & mean, Eigen::VectorXd & PrincCurv1, Eigen::VectorXd & PrincCurv2, Eigen::VectorXd & CurvX, Eigen::VectorXd & CurvY, Eigen::MatrixXd & PrincGrowthDir1, Eigen::MatrixXd & PrincGrowthDir2) const
{
    // get current state quantities
    const int nFaces = mesh.getNumberOfFaces();
    const auto & currentState = mesh.getCurrentConfiguration();

    Eigen::MatrixXd ProjNewDir1_Arr(nFaces,3); //for plotting
    Eigen::MatrixXd ProjNewDir2_Arr(nFaces,3);

    for(int i=0;i<nFaces;++i)
    {
        const ExtendedTriangleInfo & info = currentState.getTriangleInfo(i);

        const Eigen::Matrix2d aform = info.computeFirstFundamentalForm();
        const Eigen::Matrix2d bform = info.computeSecondFundamentalForm();

        const Eigen::Matrix2d shapeOp = aform.inverse() * bform;

        const Eigen::VectorXcd eigenval_comp = shapeOp.eigenvalues();
        Eigen::VectorXd eigenval = eigenval_comp.real();

        const Eigen::EigenSolver<Eigen::Matrix2d> es(shapeOp);
        const Eigen::Matrix2cd eigenvec_comp = es.eigenvectors();
        const Eigen::Matrix2d eigenvec = eigenvec_comp.real();

        //project vectors NewDir1 and NewDir2 onto the plane of our triangle
        //use v2 as point of reference
        const Real distX = (info.face_normal).dot(PrincGrowthDir1.row(i));
        const Real distY = (info.face_normal).dot(PrincGrowthDir2.row(i));
        Eigen::Vector3d PointX = (Eigen::Vector3d() <<  (info.v2(0)+PrincGrowthDir1(i,0)), (info.v2(1)+PrincGrowthDir1(i,1)), (info.v2(2)+PrincGrowthDir1(i,2))).finished();
        Eigen::Vector3d PointY = (Eigen::Vector3d() <<  (info.v2(0)+PrincGrowthDir2(i,0)), (info.v2(1)+PrincGrowthDir2(i,1)), (info.v2(2)+PrincGrowthDir2(i,2))).finished();

        Eigen::Vector3d ProjPointX = PointX - distX*info.face_normal;
        Eigen::Vector3d ProjPointY = PointY - distY*info.face_normal;

        Eigen::Vector3d ProjNewDir1 = (ProjPointX - info.v2);
        Eigen::Vector3d ProjNewDir2 = (ProjPointY - info.v2);
        ProjNewDir1 /= ProjNewDir1.norm();
        ProjNewDir2 /= ProjNewDir2.norm();

        for(int j=0;j<3;++j){
          ProjNewDir1_Arr(i,j) = ProjNewDir1(j);
          ProjNewDir2_Arr(i,j) = ProjNewDir2(j);
        }

        //vectors along princ curv directions in the normal system of coords
        Eigen::Vector3d eigenvec_abs_1;
        Eigen::Vector3d eigenvec_abs_2;
        for(int j=0;j<3;++j)
        {
          eigenvec_abs_1(j) = eigenvec(0,0)*info.e1(j)+eigenvec(1,0)*info.e2(j);
          eigenvec_abs_2(j) = eigenvec(0,1)*info.e1(j)+eigenvec(1,1)*info.e2(j);
        }
        eigenvec_abs_1 /= eigenvec_abs_1.norm();
        eigenvec_abs_2 /= eigenvec_abs_2.norm();

        const Real cosX = eigenvec_abs_1(0)*ProjNewDir1(0) + eigenvec_abs_1(1)*ProjNewDir1(1) + eigenvec_abs_1(2)*ProjNewDir1(2); // scalar product
        const Real sinX = std::sqrt(1.0-std::pow(cosX,2));
        const Real cosY = eigenvec_abs_1(0)*ProjNewDir2(0) + eigenvec_abs_1(1)*ProjNewDir2(1) + eigenvec_abs_1(2)*ProjNewDir2(2);
        const Real sinY = std::sqrt(1.0-std::pow(cosY,2));

        CurvX(i) = eigenval(0)*std::pow(cosX,2) + eigenval(1)*std::pow(sinX,2); //Euler formula
        CurvY(i) = eigenval(0)*std::pow(cosY,2) + eigenval(1)*std::pow(sinY,2);

        if (std::abs(eigenval(1))>std::abs(eigenval(0))){
          Real temp = eigenval(0);
          eigenval(0) = eigenval(1);
          eigenval(1) = temp;
        }
        PrincCurv1(i) = eigenval(0);
        PrincCurv2(i) = eigenval(1);

        const Real gauss_curv = shapeOp.determinant();
        const Real mean_curv = 0.5*shapeOp.trace();

        gauss(i) = gauss_curv;
        mean(i) = mean_curv;
    }

    const auto face2vertices = mesh.getTopology().getFace2Vertices();
    const auto vertices = mesh.getCurrentConfiguration().getVertices();
    WriteVTK writer(vertices, face2vertices);
    writer.addVectorFieldToFaces(ProjNewDir1_Arr, "ProjNewDir1");
    writer.addVectorFieldToFaces(ProjNewDir2_Arr, "ProjNewDir2");
    writer.write("CurvatureDirections.vtp");
}




#include "Mesh.hpp"
template class ComputeCurvatures<Mesh>;
template class ComputeCurvatures<BilayerMesh>;
