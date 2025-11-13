//
//  CombinedOperator_Parametric_InverseGrowth_Bilayer.cpp
//  Elasticity
//
//  Created by Wim M. van Rees on 8/15/18.
//  Copyright © 2018 Wim van Rees. All rights reserved.
//

#include "CombinedOperator_Parametric_InverseGrowth_Bilayer.hpp"

#include "MergePerFaceQuantities.hpp"
#include "ComputeCombined_Parametric_InverseGrowth.hpp"


struct MergeHessian_Bilayer
{
    const TopologyData & topo;
    const BoundaryConditionsData & boundaryConditions;
    const std::vector<ExtendedTriangleInfo> & vInfo;
    const tVecMat2d & aforms_bot;
    const tVecMat2d & aforms_top;
    const Eigen::Ref<const Eigen::MatrixXd> gradient_vertices; // the energy gradient per-vertex (all faces contributions summed together)
    const Eigen::Ref<const Eigen::VectorXd> gradient_edges; // the energy gradient per-edge (all faces contributions summed together)
    const Eigen::Ref<const Eigen::MatrixXd> per_face_hessian_bot; // the energy hessian per-face, separated between all individual components (all vertices, edges, opposite vertices, etc)
    const Eigen::Ref<const Eigen::MatrixXd> per_face_hessian_top; // the energy hessian per-face, separated between all individual components (all vertices, edges, opposite vertices, etc)
    const Eigen::Ref<const Eigen::VectorXd> per_vertex_areas;
    const Eigen::Ref<const Eigen::VectorXd> per_edge_areas;

    Eigen::Ref<Eigen::MatrixXd> hessian_faces_abar_bot; // the accumulation of all gradient contributions wrt abar
    Eigen::Ref<Eigen::MatrixXd> hessian_faces_abar_top; // the accumulation of all gradient contributions wrt abar

    MergeHessian_Bilayer(const TopologyData & topo, const BoundaryConditionsData & boundaryConditions, const std::vector<ExtendedTriangleInfo> & vInfo, const tVecMat2d & aforms_bot, const tVecMat2d & aforms_top, const Eigen::Ref<const Eigen::MatrixXd> gv, const Eigen::Ref<const Eigen::VectorXd> ge, const Eigen::Ref<const Eigen::MatrixXd> pfh_bot, const Eigen::Ref<const Eigen::MatrixXd> pfh_top, const Eigen::Ref<const Eigen::VectorXd> pva, const Eigen::Ref<const Eigen::VectorXd> pea, Eigen::Ref<Eigen::MatrixXd> hf_abar_bot, Eigen::Ref<Eigen::MatrixXd> hf_abar_top):
    topo(topo),
    boundaryConditions(boundaryConditions),
    vInfo(vInfo),
    aforms_bot(aforms_bot),
    aforms_top(aforms_top),
    gradient_vertices(gv),
    gradient_edges(ge),
    per_face_hessian_bot(pfh_bot),
    per_face_hessian_top(pfh_top),
    per_vertex_areas(pva),
    per_edge_areas(pea),
    hessian_faces_abar_bot(hf_abar_bot),
    hessian_faces_abar_top(hf_abar_top)
    {}

    MergeHessian_Bilayer(const MergeHessian_Bilayer & c, tbb::split):
    topo(c.topo),
    boundaryConditions(c.boundaryConditions),
    vInfo(c.vInfo),
    aforms_bot(c.aforms_bot),
    aforms_top(c.aforms_top),
    gradient_vertices(c.gradient_vertices),
    gradient_edges(c.gradient_edges),
    per_face_hessian_bot(c.per_face_hessian_bot),
    per_face_hessian_top(c.per_face_hessian_top),
    per_vertex_areas(c.per_vertex_areas),
    per_edge_areas(c.per_edge_areas),
    hessian_faces_abar_bot(c.hessian_faces_abar_bot),
    hessian_faces_abar_top(c.hessian_faces_abar_top)
    {}

    void join(const MergeHessian_Bilayer & )
    {
        //        nothing here : but it has to be here to allow parallel_reduce, which in turn has to be to allow a non-const operator() (we technically are const but we change a const & in the form of const Eigen::Ref<const Eigen::...> ).
    }

    void operator () (const tbb::blocked_range<int>& face_range)
    {
        // back to the per-face approach
        const auto face2edges = topo.getFace2Edges();
        const auto edge2faces = topo.getEdge2Faces();
        const auto face2vertices = topo.getFace2Vertices();

        const int nDims = 21;

        // one or the other for now --> note: grad_h not implemented yet!!
        const Real hY_prefac = 1.0;

        for (int i=face_range.begin(); i != face_range.end(); ++i)
        {
            const ExtendedTriangleInfo & info = vInfo[i];

            int idx_v_other_e0 = -1;
            int idx_v_other_e1 = -1;
            int idx_v_other_e2 = -1;

            if(info.other_faces[0] != nullptr)
            {
                const int other_face_idx = info.other_faces[0]->face_idx;

                const int startidx = (face2edges(other_face_idx,0) == info.idx_e0 ? 0 : (face2edges(other_face_idx,1) == info.idx_e0 ? 1 : 2) );
                const int idx_other_v2 = face2vertices(other_face_idx,(startidx+2)%3);
                idx_v_other_e0 = idx_other_v2;
            }

            if(info.other_faces[1] != nullptr)
            {
                const int other_face_idx = info.other_faces[1]->face_idx;

                const int startidx = (face2edges(other_face_idx,0) == info.idx_e1 ? 0 : (face2edges(other_face_idx,1) == info.idx_e1 ? 1 : 2) );
                const int idx_other_v0 = face2vertices(other_face_idx,(startidx+2)%3);
                idx_v_other_e1 = idx_other_v0;

            }

            if(info.other_faces[2] != nullptr)
            {
                const int other_face_idx = info.other_faces[2]->face_idx;

                const int startidx = (face2edges(other_face_idx,0) == info.idx_e2 ? 0 : (face2edges(other_face_idx,1) == info.idx_e2 ? 1 : 2) );
                const int idx_other_v1 = face2vertices(other_face_idx,(startidx+2)%3);
                idx_v_other_e2 = idx_other_v1;
            }

            // compute the gradient of the area with respect to each of the aforms
            // area = 0.5*sqrt(0.5*(abot + atop)) // midsurface area assuming equal thickness
            /*
             brief derivation:
             A = 0.5*sqrt(aform.det)
             dA/daform = 0.25/sqrt(aform.det) * grad_aform (aform_det)
             = 1/(8 * A) * grad_aform(aform_det)
             = 1/(4 * sqrt(aform_det)) * grad_aform(aform_det)
             = 0.25/det_abar * grad_aform(aform_det)
             */

            const Eigen::Matrix2d aforms_avg = 0.5*(aforms_bot[i] + aforms_top[i]);
            const Real det_abar = std::sqrt(aforms_avg.determinant()); // twice the area
            /*const Real grad_det_bot_11 = 0.5*aforms_bot[i](1,1);
            const Real grad_det_bot_12 = - 0.5*aforms_bot[i](0,1) - 0.5*aforms_bot[i](1,0);
            const Real grad_det_bot_22 = 0.5*aforms_bot[i](0,0);
            const Real grad_det_top_11 = 0.5*aforms_top[i](1,1);
            const Real grad_det_top_12 = - 0.5*aforms_top[i](0,1) - 0.5*aforms_top[i](1,0);
            const Real grad_det_top_22 = 0.5*aforms_top[i](0,0);*/

            const Real grad_det_bot_11 = 0.25*(aforms_bot[i](1,1) + aforms_top[i](1,1));
            const Real grad_det_bot_12 = -0.5*(aforms_bot[i](1,0) + aforms_top[i](1,0));
            const Real grad_det_bot_22 = 0.25*(aforms_bot[i](0,0) + aforms_top[i](0,0));
            const Real grad_det_top_11 = 0.25*(aforms_bot[i](1,1) + aforms_top[i](1,1));
            const Real grad_det_top_12 = -0.5*(aforms_bot[i](1,0) + aforms_top[i](1,0));
            const Real grad_det_top_22 = 0.25*(aforms_bot[i](0,0) + aforms_top[i](0,0));

            const Eigen::Vector3d grad_area_bot = 0.25/det_abar * (Eigen::Vector3d() << grad_det_bot_11, grad_det_bot_12, grad_det_bot_22).finished();
            const Eigen::Vector3d grad_area_top = 0.25/det_abar * (Eigen::Vector3d() << grad_det_top_11, grad_det_top_12, grad_det_top_22).finished();

            // now we can accumulate the different terms : bottom layer
            for(int j=0;j<3;++j) // loop over a11, a12, a22
            {
                for(int d=0;d<3;++d)
                {
                    // v0, v1, v2
                    hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v0, d) * per_face_hessian_bot(i, nDims*j + 3*0 + d) / per_vertex_areas(info.idx_v0) - std::pow(gradient_vertices(info.idx_v0, d) / per_vertex_areas(info.idx_v0),2) * grad_area_bot(j) / 3.0); // onethird area
                    hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v1, d) * per_face_hessian_bot(i, nDims*j + 3*1 + d) / per_vertex_areas(info.idx_v1) - std::pow(gradient_vertices(info.idx_v1, d) / per_vertex_areas(info.idx_v1),2) * grad_area_bot(j) / 3.0);
                    hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v2, d) * per_face_hessian_bot(i, nDims*j + 3*2 + d) / per_vertex_areas(info.idx_v2) - std::pow(gradient_vertices(info.idx_v2, d) / per_vertex_areas(info.idx_v2),2) * grad_area_bot(j) / 3.0);

                    // v_other_e0, v_other_e1, v_other_e2 (those areas are independent of our aform - no second term there)
                    if(idx_v_other_e0 >= 0) hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e0, d) * per_face_hessian_bot(i, nDims*j + 3*3 + d) / per_vertex_areas(idx_v_other_e0));
                    if(idx_v_other_e1 >= 0) hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e1, d) * per_face_hessian_bot(i, nDims*j + 3*4 + d) / per_vertex_areas(idx_v_other_e1));
                    if(idx_v_other_e2 >= 0) hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e2, d) * per_face_hessian_bot(i, nDims*j + 3*5 + d) / per_vertex_areas(idx_v_other_e2));
                }

                // edge e0, e1, e2
                hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e0) * per_face_hessian_bot(i, nDims*j + 3*6 + 0) / per_edge_areas(info.idx_e0) - std::pow(gradient_edges(info.idx_e0) / per_edge_areas(info.idx_e0), 2) * grad_area_bot(j) / 2.0); // onehalf area
                hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e1) * per_face_hessian_bot(i, nDims*j + 3*6 + 1) / per_edge_areas(info.idx_e1) - std::pow(gradient_edges(info.idx_e1) / per_edge_areas(info.idx_e1), 2) * grad_area_bot(j) / 2.0);
                hessian_faces_abar_bot(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e2) * per_face_hessian_bot(i, nDims*j + 3*6 + 2) / per_edge_areas(info.idx_e2) - std::pow(gradient_edges(info.idx_e2) / per_edge_areas(info.idx_e2), 2) * grad_area_bot(j) / 2.0);
            }

            // now we can accumulate the different terms : top layer
            for(int j=0;j<3;++j) // loop over a11, a12, a22
            {
                for(int d=0;d<3;++d)
                {
                    // v0, v1, v2
                    hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v0, d) * per_face_hessian_top(i, nDims*j + 3*0 + d) / per_vertex_areas(info.idx_v0) - std::pow(gradient_vertices(info.idx_v0, d) / per_vertex_areas(info.idx_v0),2) * grad_area_top(j) / 3.0); // onethird area
                    hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v1, d) * per_face_hessian_top(i, nDims*j + 3*1 + d) / per_vertex_areas(info.idx_v1) - std::pow(gradient_vertices(info.idx_v1, d) / per_vertex_areas(info.idx_v1),2) * grad_area_top(j) / 3.0);
                    hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(info.idx_v2, d) * per_face_hessian_top(i, nDims*j + 3*2 + d) / per_vertex_areas(info.idx_v2) - std::pow(gradient_vertices(info.idx_v2, d) / per_vertex_areas(info.idx_v2),2) * grad_area_top(j) / 3.0);

                    // v_other_e0, v_other_e1, v_other_e2 (those areas are independent of our aform - no second term there)
                    if(idx_v_other_e0 >= 0) hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e0, d) * per_face_hessian_top(i, nDims*j + 3*3 + d) / per_vertex_areas(idx_v_other_e0));
                    if(idx_v_other_e1 >= 0) hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e1, d) * per_face_hessian_top(i, nDims*j + 3*4 + d) / per_vertex_areas(idx_v_other_e1));
                    if(idx_v_other_e2 >= 0) hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_vertices(idx_v_other_e2, d) * per_face_hessian_top(i, nDims*j + 3*5 + d) / per_vertex_areas(idx_v_other_e2));
                }

                // edge e0, e1, e2
                hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e0) * per_face_hessian_top(i, nDims*j + 3*6 + 0) / per_edge_areas(info.idx_e0) - std::pow(gradient_edges(info.idx_e0) / per_edge_areas(info.idx_e0), 2) * grad_area_top(j) / 2.0); // onehalf area
                hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e1) * per_face_hessian_top(i, nDims*j + 3*6 + 1) / per_edge_areas(info.idx_e1) - std::pow(gradient_edges(info.idx_e1) / per_edge_areas(info.idx_e1), 2) * grad_area_top(j) / 2.0);
                hessian_faces_abar_top(i,j) += hY_prefac * (2.0 * gradient_edges(info.idx_e2) * per_face_hessian_top(i, nDims*j + 3*6 + 2) / per_edge_areas(info.idx_e2) - std::pow(gradient_edges(info.idx_e2) / per_edge_areas(info.idx_e2), 2) * grad_area_top(j) / 2.0);
            }
        }
    }
};



template<typename tMesh, typename tMaterialType>
Real CombinedOperator_Parametric_InverseGrowth_Bilayer<tMesh, tMaterialType>::computeAll(const tMesh & mesh, Eigen::Ref<Eigen::VectorXd> gradient, const bool computeGradient) const
{
    if(computeGradient)
    {
        if(withGradientThickness) assert(gradient.rows() >= 4*mesh.getNumberOfFaces());
        else if(withGradientYoung) assert(gradient.rows() >= 4*mesh.getNumberOfFaces());
        else assert(gradient.rows() >= 3*mesh.getNumberOfFaces());
    }

    const auto & currentState = mesh.getCurrentConfiguration();
    const auto & restState = mesh.getRestConfiguration();

    const int nFaces = mesh.getNumberOfFaces();
    const int grainSize = 10;

    Eigen::MatrixXd per_face_gradients(nFaces, 21);
    per_face_gradients.setZero();

    Eigen::MatrixXd per_face_hessians_bot, per_face_hessians_top;


    this->profiler.push_start("compute energy / gradient");
    if(not computeGradient)
    {
        // do bottom layer
        ComputeCombined_Parametric_InverseGrowth<tMesh, tMaterialType, bottom, false> compute_tbb_bot(material_properties_bot, currentState, restState, per_face_gradients, per_face_hessians_bot);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces,grainSize), compute_tbb_bot, tbb::auto_partitioner());

        // do top layer
        ComputeCombined_Parametric_InverseGrowth<tMesh, tMaterialType, top, false> compute_tbb_top(material_properties_top, currentState, restState, per_face_gradients, per_face_hessians_top);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces,grainSize), compute_tbb_top, tbb::auto_partitioner());
    }
    else
    {
        int nComponents = 3;
        if(withGradientThickness) nComponents++;
        if(withGradientYoung) nComponents++;
        per_face_hessians_bot.resize(nFaces, nComponents*21);
        per_face_hessians_bot.setZero();

        per_face_hessians_top.resize(nFaces, nComponents*21);
        per_face_hessians_top.setZero();

        ComputeCombined_Parametric_InverseGrowth<tMesh, tMaterialType, bottom, true> compute_tbb_bot(material_properties_bot, currentState, restState,  per_face_gradients, per_face_hessians_bot);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces,grainSize), compute_tbb_bot, tbb::auto_partitioner());

        ComputeCombined_Parametric_InverseGrowth<tMesh, tMaterialType, top, true> compute_tbb_top(material_properties_top, currentState, restState,  per_face_gradients, per_face_hessians_top);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces,grainSize), compute_tbb_top, tbb::auto_partitioner());
    }
    this->profiler.pop_stop();

    // now we merge everything to compute the per_vertex_gradient and per_edge_Gradient
    const int nVertices = mesh.getNumberOfVertices();
    const int nEdges = mesh.getNumberOfEdges();
    Eigen::MatrixXd per_vertex_gradients(nVertices, 3);
    Eigen::VectorXd per_edge_gradients(nEdges);
    per_vertex_gradients.setZero();
    per_edge_gradients.setZero();

    const auto & topo = mesh.getTopology();
    const auto & boundaryConditions = mesh.getBoundaryConditions();
    const tVecMat2d & aforms_bot = mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>();
    const tVecMat2d & aforms_top = mesh.getRestConfiguration().template getFirstFundamentalForms<top>();
    tVecMat2d aforms_avg(nFaces);
    {
        for(int i=0;i<nFaces;++i)
            aforms_avg[i] = 0.5*(aforms_bot[i] + aforms_top[i]);
    }
    this->profiler.push_start("merge per-vertex gradient");
    MergeGradVertices mergevertices(topo, boundaryConditions, per_face_gradients, per_vertex_gradients);
    tbb::parallel_reduce(tbb::blocked_range<int>(0,nVertices,grainSize),mergevertices, tbb::auto_partitioner());
    this->profiler.pop_stop();

    this->profiler.push_start("merge per-edge gradient");
    MergeGradEdges mergeedges(topo, boundaryConditions, per_face_gradients, per_edge_gradients);
    tbb::parallel_reduce(tbb::blocked_range<int>(0,nEdges,grainSize),mergeedges, tbb::auto_partitioner());
    this->profiler.pop_stop();

    // compute the area per-vertex and per-edge
    this->profiler.push_start("multiply and compute per-vertex area");
    Eigen::VectorXd vertex_areas(nVertices);
    Eigen::Map<Eigen::VectorXd> per_vertex_gradients_unrolled(per_vertex_gradients.data(), 3*nVertices);
    MultiplyAndComputeAreas computeAreas_vertices(topo, aforms_avg, per_vertex_gradients_unrolled, vertex_areas, MultiplyAndComputeAreas::RunCase::vertices);
    tbb::parallel_reduce(tbb::blocked_range<int>(0,nVertices,grainSize),computeAreas_vertices, tbb::auto_partitioner());
    this->profiler.pop_stop();

    this->profiler.push_start("multiply and compute per-edge area");
    Eigen::VectorXd edge_areas(nEdges);
    MultiplyAndComputeAreas computeAreas_edges(topo, aforms_avg, per_edge_gradients, edge_areas, MultiplyAndComputeAreas::RunCase::edges);
    tbb::parallel_reduce(tbb::blocked_range<int>(0,nEdges,grainSize),computeAreas_edges, tbb::auto_partitioner());
    this->profiler.pop_stop();

    const Real energy_retval = computeAreas_vertices.energySum + computeAreas_edges.energySum;

//    Real avgThickness = 0.0;
//    if(withGradientThickness)
//    {
//        const std::vector<ExtendedTriangleInfo> & vInfo = mesh.getCurrentConfiguration().getTriangleInfos();
//        for(int i=0;i<nFaces;++i)
//        {
//            const Real thickness = material_properties.getFaceMaterial(i).getThickness();
//            avgThickness += thickness*vInfo[i].double_face_area*0.5;
//        }
//    }
//
//    Real avgYoung = 0.0;
//    if(withGradientYoung)
//    {
//        const std::vector<ExtendedTriangleInfo> & vInfo = mesh.getCurrentConfiguration().getTriangleInfos();
//        for(int i=0;i<nFaces;++i)
//        {
//            const Real Young = material_properties.getFaceMaterial(i).getYoung(); // only isotropic --> is what we have
//            avgYoung += Young*vInfo[i].double_face_area*0.5;
//        }
//    }

    if(computeGradient)
    {
        // map the gradient into abar
        Eigen::Map<Eigen::MatrixXd> gradient_abar_bot(gradient.data(), nFaces, 3);
        Eigen::Map<Eigen::MatrixXd> gradient_abar_top(gradient.data() + 3*nFaces, nFaces, 3);

//        int offset = 3*nFaces;
//        ScalarGradientHelper grad_h(withGradientThickness, avgThickness);
//        ScalarGradientHelper grad_Y(withGradientYoung, avgYoung);
//        if(withGradientThickness)
//        {
//            new (&grad_h.gradient) Eigen::Map<Eigen::VectorXd>(gradient.data() + offset, nFaces);
//            offset += nFaces;
//        }
//        if(withGradientYoung)
//        {
//            new (&grad_Y.gradient) Eigen::Map<Eigen::VectorXd>(gradient.data() + offset, nFaces);
//            offset += nFaces;
//        }


        const std::vector<ExtendedTriangleInfo> & vInfo = mesh.getCurrentConfiguration().getTriangleInfos();

        this->profiler.push_start("merge hessian");
        MergeHessian_Bilayer mergehessian(topo, boundaryConditions, vInfo, aforms_bot, aforms_top, per_vertex_gradients, per_edge_gradients, per_face_hessians_bot, per_face_hessians_top, vertex_areas, edge_areas, gradient_abar_bot, gradient_abar_top);
        tbb::parallel_reduce(tbb::blocked_range<int>(0,nFaces,grainSize),mergehessian, tbb::auto_partitioner());
        this->profiler.pop_stop();
    }

    return energy_retval;
}

// explicit instantiations
#include "Mesh.hpp"

// only bilayer mesh
template class CombinedOperator_Parametric_InverseGrowth_Bilayer<BilayerMesh, Material_Isotropic>;
