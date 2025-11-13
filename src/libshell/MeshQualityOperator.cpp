//
//  MeshQualityOperator.cpp
//  Elasticity
//
//  Created by Wim van Rees on 12/2/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "MeshQualityOperator.hpp"

#include "ExtendedTriangleInfo.hpp"

template<typename tMesh>
Real MeshQualityOperator<tMesh>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::MatrixXd> gradient_vertices, Eigen::Ref<Eigen::VectorXd> , const bool computeGradient) const
{
    // get the mesh features
    const auto & topo = mesh.getTopology();
    const auto & currentState = mesh.getCurrentConfiguration();
    const auto & boundaryConditions = mesh.getBoundaryConditions();

    // get mesh quantities
    const auto vertices = currentState.getVertices();

    const auto vertices_bc = boundaryConditions.getVertexBoundaryConditions();
    const auto face2edges = topo.getFace2Edges();
    const auto face2vertices = topo.getFace2Vertices();

    const Real equilateral_prefac = 4.0*std::sqrt(3.0);

    Real energy = 0.0;

    // compute stretching forces by looping over faces
    for(int i=0;i<face2edges.rows();++i)
    {

        const ExtendedTriangleInfo & info = currentState.getTriangleInfo(i);

        // vertex indices
        const int idx_v0 = face2vertices(i,0);
        const int idx_v1 = face2vertices(i,1);
        const int idx_v2 = face2vertices(i,2);

        // vertex locations
        const Eigen::Vector3d v0_old = vertices.row(idx_v0);
        const Eigen::Vector3d v1_old = vertices.row(idx_v1);
        const Eigen::Vector3d v2_old = vertices.row(idx_v2);

        // perform the transform
        const Eigen::Vector3d v0_new = trafo_func(v0_old);
        const Eigen::Vector3d v1_new = trafo_func(v1_old);
        const Eigen::Vector3d v2_new = trafo_func(v2_old);

        // write down the edges
        const Eigen::Vector3d e0_old = v1_old - v0_old;
        const Eigen::Vector3d e1_old = v2_old - v1_old;
        const Eigen::Vector3d e2_old = v0_old - v2_old;

        // write down the edges
        const Eigen::Vector3d e0 = v1_new - v0_new;
        const Eigen::Vector3d e1 = v2_new - v1_new;
        const Eigen::Vector3d e2 = v0_new - v2_new;

        const Real l_e0_sq = e0.dot(e0); // e0 dot e0
        const Real l_e1_sq = e1.dot(e1); // e1 dot e1
        const Real l_e2_sq = e2.dot(e2); // e2 dot e2
        const Real e0_dot_e1 = e0.dot(e1);
        const Real area = 0.5 * std::sqrt(l_e0_sq * l_e1_sq - e0_dot_e1*e0_dot_e1);

        const Real denum = (l_e0_sq + l_e1_sq + l_e2_sq);
        const Real qualityMetric = equilateral_prefac * area / denum; // distortion from equilateral : 1 if equilateral
        const Real areaMetric = std::pow(qualityMetric - 1, area_exponent);

        energy += area_prefac * areaMetric;// * area;

        if(computeGradient)
        {
            // compute the gradients wrt transform function
            const Real FD_eps = 1e-6;
            Eigen::Matrix3d gradtrafo_v0, gradtrafo_v1, gradtrafo_v2;
            Eigen::Vector3d trafo_eps;
            for(int d=0;d<3;++d)
            {
                trafo_eps.setZero();
                trafo_eps(d) = +FD_eps;
                // vnew = f(v)
                // now the energy is T(vnew) = T(f(v))
                // dT/dvnew = dT/df * df/dv
                // here we need to compute df/dv (matrix)
                const Eigen::Vector3d v0_plus = trafo_func(v0_old + trafo_eps);
                const Eigen::Vector3d v1_plus = trafo_func(v1_old + trafo_eps);
                const Eigen::Vector3d v2_plus = trafo_func(v2_old + trafo_eps);
                const Eigen::Vector3d v0_mins = trafo_func(v0_old - trafo_eps);
                const Eigen::Vector3d v1_mins = trafo_func(v1_old - trafo_eps);
                const Eigen::Vector3d v2_mins = trafo_func(v2_old - trafo_eps);
                for(int dd=0;dd<3;++dd)
                {
                    gradtrafo_v0(d, dd) = 0.5/FD_eps * (v0_plus(dd) - v0_mins(dd)); // df/dv_x
                    gradtrafo_v1(d, dd) = 0.5/FD_eps * (v1_plus(dd) - v1_mins(dd)); // df/dv_y
                    gradtrafo_v2(d, dd) = 0.5/FD_eps * (v2_plus(dd) - v2_mins(dd)); // df/dv_z
                }

                // the matrix contains
                // { {dfx/dvx, dfy/dvx, dfz/dvx} ,
                //   {dfx/dvy, dfy/dvy, dfz/dvy} ,
                //   {dfx/dvz, dfy/dvz, dfz/dvz} }


            }

            // compute the gradients edge lengths squared
            const Eigen::Vector3d gradv0_e0sq = -2.0*e0;
            const Eigen::Vector3d gradv1_e0sq = +2.0*e0;

            const Eigen::Vector3d gradv1_e1sq = -2.0*e1;
            const Eigen::Vector3d gradv2_e1sq = +2.0*e1;

            const Eigen::Vector3d gradv2_e2sq = -2.0*e2;
            const Eigen::Vector3d gradv0_e2sq = +2.0*e2;

            // compute the gradients for the area
            // write area as 0.5 * sqrt ( |e0|^2 * |e1|^2 - (e0.dot(e1))^2 )
            // 0.5*std::sqrt(l_e0_sq*l_e1_sq - std::pow(e0.dot(e1),2))

            const Real grad_area_prefac = 0.125 / area; // d/dx 1/2 sqrt(x) = 1/[ 4 sqrt(x) ] = 1/8 [ 1 / (1/2 sqrt(x) ) ]
            const Eigen::Vector3d gradv0_area = grad_area_prefac * (gradv0_e0sq*l_e1_sq - 2.0*e0_dot_e1*(-e1));
            const Eigen::Vector3d gradv1_area = grad_area_prefac * (gradv1_e0sq*l_e1_sq + gradv1_e1sq*l_e0_sq - 2.0*e0_dot_e1*(e1 - e0));
            const Eigen::Vector3d gradv2_area = grad_area_prefac * (gradv2_e1sq*l_e0_sq - 2.0*e0_dot_e1*( e0));

            const Eigen::Vector3d gradv0_qualityMetric = equilateral_prefac * (gradv0_area / denum  - area / (denum * denum) * (gradv0_e0sq + gradv0_e2sq) );
            const Eigen::Vector3d gradv1_qualityMetric = equilateral_prefac * (gradv1_area / denum  - area / (denum * denum) * (gradv1_e0sq + gradv1_e1sq) );
            const Eigen::Vector3d gradv2_qualityMetric = equilateral_prefac * (gradv2_area / denum  - area / (denum * denum) * (gradv2_e1sq + gradv2_e2sq) );

            const Eigen::Vector3d gradv0_areaMetric = area_exponent * std::pow(qualityMetric - 1, area_exponent - 1) * gradv0_qualityMetric;
            const Eigen::Vector3d gradv1_areaMetric = area_exponent * std::pow(qualityMetric - 1, area_exponent - 1) * gradv1_qualityMetric;
            const Eigen::Vector3d gradv2_areaMetric = area_exponent * std::pow(qualityMetric - 1, area_exponent - 1) * gradv2_qualityMetric;

            // compute the gradient of the energy wrt quality Metric

            const Eigen::Vector3d gradv0_old_areaMetric = gradtrafo_v0 * gradv0_areaMetric;
            const Eigen::Vector3d gradv1_old_areaMetric = gradtrafo_v1 * gradv1_areaMetric;
            const Eigen::Vector3d gradv2_old_areaMetric = gradtrafo_v2 * gradv2_areaMetric;

            for(int j=0;j<3;++j)
            {
                if(not vertices_bc(idx_v0,j)) gradient_vertices(idx_v0,j) += area_prefac * gradv0_old_areaMetric(j);
                if(not vertices_bc(idx_v1,j)) gradient_vertices(idx_v1,j) += area_prefac * gradv1_old_areaMetric(j);
                if(not vertices_bc(idx_v2,j)) gradient_vertices(idx_v2,j) += area_prefac * gradv2_old_areaMetric(j);
            }
        }


        if(not keepNormalToTarget) continue;

        // add normal projection
        const Real l_e0_old_sq = e0_old.dot(e0_old);
        const Real l_e1_old_sq = e1_old.dot(e1_old);
        const Real l_e2_old_sq = e2_old.dot(e2_old);
        const Real e0_old_dot_e1_old = e0_old.dot(e1_old);
        const Real area_old = 0.5 * std::sqrt(l_e0_old_sq * l_e1_old_sq - e0_old_dot_e1_old*e0_old_dot_e1_old);

        const Eigen::Vector3d facepos = (v0_old + v1_old + v2_old)/3.0; // do it wrt old positions
        const Eigen::Vector3d target_normal = target_normal_func(facepos);
        const Eigen::Vector3d face_normal = info.face_normal;
        const Real normalDotProduct = target_normal.dot(face_normal); // 1 if aligned, -1 if not aligned
        const Real normalMetric = 0.5 - 0.5 * std::tanh( normal_k * normalDotProduct * area_old ); // normalDotProduct < 0 --> this is +1, >0 --> this is 0
        energy += normal_prefac * normalMetric; // multiply by area so that the gradient varies smoothly from negative to positive values, going through zero

        if(computeGradient)
        {
            // normal metric gradient
            // compute the gradients edge lengths squared
            const Eigen::Vector3d gradv0_e0sq_old = -2.0*e0_old;
            const Eigen::Vector3d gradv1_e0sq_old = +2.0*e0_old;

            const Eigen::Vector3d gradv1_e1sq_old = -2.0*e1_old;
            const Eigen::Vector3d gradv2_e1sq_old = +2.0*e1_old;

            // compute the gradients for the area
            // write area as 0.5 * sqrt ( |e0|^2 * |e1|^2 - (e0.dot(e1))^2 )
            // 0.5*std::sqrt(l_e0_sq*l_e1_sq - std::pow(e0.dot(e1),2))

            const Real grad_area_old_prefac = 0.125 / area_old; // d/dx 1/2 sqrt(x) = 1/[ 4 sqrt(x) ] = 1/8 [ 1 / (1/2 sqrt(x) ) ]
            const Eigen::Vector3d gradv0_area_old = grad_area_old_prefac * (gradv0_e0sq_old*l_e1_old_sq - 2.0*e0_old_dot_e1_old*(-e1_old));
            const Eigen::Vector3d gradv1_area_old = grad_area_old_prefac * (gradv1_e0sq_old*l_e1_old_sq + gradv1_e1sq_old*l_e0_old_sq - 2.0*e0_old_dot_e1_old*(e1_old - e0_old));
            const Eigen::Vector3d gradv2_area_old = grad_area_old_prefac * (gradv2_e1sq_old*l_e0_old_sq - 2.0*e0_old_dot_e1_old*( e0_old));


            const Real double_face_area = 2.0 * area_old;

            const Real h0 = double_face_area / std::sqrt(l_e0_old_sq);
            const Real h1 = double_face_area / std::sqrt(l_e1_old_sq);
            const Real h2 = double_face_area / std::sqrt(l_e2_old_sq);

            const Eigen::Vector3d e0_normal = (e0_old.cross(face_normal)).normalized();
            const Eigen::Vector3d e1_normal = (e1_old.cross(face_normal)).normalized();
            const Eigen::Vector3d e2_normal = (e2_old.cross(face_normal)).normalized();

            const Eigen::Matrix3d gradv0_n = 1.0/h1 * e1_normal * face_normal.transpose();
            const Eigen::Matrix3d gradv1_n = 1.0/h2 * e2_normal * face_normal.transpose();
            const Eigen::Matrix3d gradv2_n = 1.0/h0 * e0_normal * face_normal.transpose();

            // NOTE: WE NEED TO ADD THE GRADIENT OF THE ENERGY WITH RESPECT TO THE TARGET NORMAL AS WELL (v0/v1/v2 change facecenter --> changes target normal)

            const Eigen::Vector3d grad_normalDotProduct_v0 = target_normal(0) * gradv0_n.row(0) + target_normal(1) * gradv0_n.row(1) + target_normal(2) * gradv0_n.row(2);
            const Eigen::Vector3d grad_normalDotProduct_v1 = target_normal(0) * gradv1_n.row(0) + target_normal(1) * gradv1_n.row(1) + target_normal(2) * gradv1_n.row(2);
            const Eigen::Vector3d grad_normalDotProduct_v2 = target_normal(0) * gradv2_n.row(0) + target_normal(1) * gradv2_n.row(1) + target_normal(2) * gradv2_n.row(2);

            const Eigen::Vector3d grad_normalMetric_v0 = -0.5 * normal_k / std::pow(std::cosh( normal_k * normalDotProduct * area_old) , 2) * (grad_normalDotProduct_v0 * area_old + normalDotProduct * gradv0_area_old);
            const Eigen::Vector3d grad_normalMetric_v1 = -0.5 * normal_k / std::pow(std::cosh( normal_k * normalDotProduct * area_old) , 2) * (grad_normalDotProduct_v1 * area_old + normalDotProduct * gradv1_area_old);
            const Eigen::Vector3d grad_normalMetric_v2 = -0.5 * normal_k / std::pow(std::cosh( normal_k * normalDotProduct * area_old) , 2) * (grad_normalDotProduct_v2 * area_old + normalDotProduct * gradv2_area_old);


            for(int j=0;j<3;++j)
            {
                if(not vertices_bc(idx_v0,j)) gradient_vertices(idx_v0,j) += normal_prefac * grad_normalMetric_v0(j);
                if(not vertices_bc(idx_v1,j)) gradient_vertices(idx_v1,j) += normal_prefac * grad_normalMetric_v1(j);
                if(not vertices_bc(idx_v2,j)) gradient_vertices(idx_v2,j) += normal_prefac * grad_normalMetric_v2(j);
            }
        }
    }

    lastEnergy = energy;
    return lastEnergy;
}

// explicit instantiations
#include "Mesh.hpp"
template class MeshQualityOperator<Mesh>;
template class MeshQualityOperator<BilayerMesh>;
