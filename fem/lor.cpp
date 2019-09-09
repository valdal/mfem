// Copyright (c) 2019, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "lor.hpp"
#include "fem.hpp"

namespace mfem
{

LORMixedMassMatrix::LORMixedMassMatrix(
   const FiniteElementSpace &fes_ho_,
   const FiniteElementSpace &fes_lor_,
   Coefficient *Q_ho,
   Coefficient *Q_lor)
   : fes_ho(fes_ho_),
     fes_lor(fes_lor_),
     ho2lor(fes_ho, fes_lor)
{
   Mesh *mesh_ho = fes_ho.GetMesh();
   MFEM_VERIFY(mesh_ho->GetNumGeometries(mesh_ho->Dimension()) <= 1,
               "mixed meshes are not supported");

   // If the local mesh is empty, skip all computations
   if (mesh_ho->GetNE() == 0) { return; }

   const FiniteElement *fe_lor = fes_lor.GetFE(0);
   const FiniteElement *fe_ho = fes_ho.GetFE(0);
   ndof_lor = fe_lor->GetDof();
   ndof_ho = fe_ho->GetDof();

   M_mixed.SetSize(ndof_lor*ho2lor.nref, ndof_ho, ho2lor.nel_ho);
   DenseMatrix M_mixed_el(ndof_lor, ndof_ho);

   IntegrationPointTransformation ip_tr;
   IsoparametricTransformation &emb_tr = ip_tr.Transf;

   Vector shape_ho(ndof_ho);
   Vector shape_lor(ndof_lor);

   const Geometry::Type geom = fe_ho->GetGeomType();
   const DenseTensor &pmats = ho2lor.cf_tr.GetPointMatrices(geom);
   emb_tr.SetIdentityTransformation(geom);

   for (int iho=0; iho<ho2lor.nel_ho; ++iho)
   {
      ElementTransformation *el_tr_ho = fes_ho.GetElementTransformation(iho);
      for (int iref=0; iref<ho2lor.nref; ++iref)
      {
         // Assemble the low-order refined mass matrix and invert locally
         int ilor = ho2lor.el_table.GetRow(iho)[iref];
         ElementTransformation *el_tr = fes_lor.GetElementTransformation(ilor);

         // Now assemble the block-row of the mixed mass matrix associated
         // with integrating HO functions against LOR functions on the LOR
         // sub-element.

         // Create the transformation that embeds the fine low-order element
         // within the coarse high-order element in reference space
         emb_tr.GetPointMat() = pmats(iref);
         emb_tr.FinalizeTransformation();

         int order = fe_lor->GetOrder() + 2*fe_ho->GetOrder() + el_tr->OrderW();
         const IntegrationRule *ir = &IntRules.Get(geom, order);
         M_mixed_el = 0.0;
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip_lor = ir->IntPoint(i);
            IntegrationPoint ip_ho;
            ip_tr.Transform(ip_lor, ip_ho);
            fe_lor->CalcShape(ip_lor, shape_lor);
            fe_ho->CalcShape(ip_ho, shape_ho);
            el_tr->SetIntPoint(&ip_lor);
            // For now we use the geometry information from the LOR space
            // which means we won't be mass conservative if the mesh is curved
            double w = el_tr->Weight()*ip_lor.weight;
            if (Q_ho)
            {
               w *= Q_ho->Eval(*el_tr_ho, ip_ho);
            }
            shape_lor *= w;
            AddMultVWt(shape_lor, shape_ho, M_mixed_el);
         }
         M_mixed(iho).CopyMN(M_mixed_el, iref*ndof_lor, 0);
      }
   }
}

DenseMatrix &LORMixedMassMatrix::GetElementMatrix(int iel)
{
   return M_mixed(iel);
}

void LORMixedMassMatrix::Assemble(OperatorHandle &A)
{
   SparseMatrix *Amat
      = new SparseMatrix(fes_lor.GetTrueVSize(), fes_ho.GetTrueVSize());

   int vdim = fes_ho.GetVDim();
   Array<int> vdofs_ho(ndof_ho), vdofs_lor(ndof_lor*ho2lor.nref);
   Array<int> vdofs_lor_local(ndof_lor);
   for (int iho=0; iho<ho2lor.nel_ho; ++iho)
   {
      for (int vd=0; vd<vdim; ++vd)
      {
         for (int iref=0; iref<ho2lor.nref; ++iref)
         {
            int ilor = ho2lor.el_table.GetRow(iho)[iref];
            fes_lor.GetElementDofs(ilor, vdofs_lor_local);
            fes_lor.DofsToVDofs(vd, vdofs_lor_local);
            for (int i=0; i<ndof_lor; ++i)
            {
               vdofs_lor[i + ndof_lor*iref] = vdofs_lor_local[i];
            }
         }
         fes_ho.GetElementDofs(iho, vdofs_ho);
         fes_ho.DofsToVDofs(vd, vdofs_ho);
         Amat->AddSubMatrix(vdofs_lor, vdofs_ho, M_mixed(iho));
      }
   }

   Amat->Finalize();

   // Set the operator handle to the sparse matrix (and assume ownership)
   A.Reset(Amat);
}

BlockDiagonalMassInverse::BlockDiagonalMassInverse(
   const FiniteElementSpace &fes_, Coefficient *Q) : fes(fes_)
{
   const int nel = fes.GetMesh()->GetNE();
   const FiniteElement *fe = fes.GetFE(0);
   const int ndof = fe->GetDof();

   M.SetSize(ndof, ndof, nel);
   M_inv.SetSize(ndof, ndof, nel);
   DenseMatrix M_el(ndof, ndof);
   DenseMatrixInverse M_el_inv(&M_el);

   MassIntegrator *mi = Q ? new MassIntegrator(*Q) : new MassIntegrator;

   for (int iel=0; iel<nel; ++iel)
   {
      ElementTransformation *el_tr = fes.GetElementTransformation(iel);
      mi->AssembleElementMatrix(*fe, *el_tr, M_el);
      M(iel) = M_el;
      M_el_inv.Factor();
      M_el_inv.GetInverseMatrix(M_inv(iel));
   }
}

DenseMatrix &BlockDiagonalMassInverse::GetElementMatrix(int iel)
{
   return M(iel);
}

DenseMatrix &BlockDiagonalMassInverse::GetElementInverse(int iel)
{
   return M_inv(iel);
}

void BlockDiagonalMassInverse::AssembleInverse(OperatorHandle &A)
{
   SparseMatrix *Amat
      = new SparseMatrix(fes.GetTrueVSize(), fes.GetTrueVSize());
   const int nel = fes.GetMesh()->GetNE();
   Array<int> vdofs;
   for (int iel=0; iel<nel; ++iel)
   {
      for (int vd=0; vd<fes.GetVDim(); ++vd)
      {
         fes.GetElementDofs(iel, vdofs);
         fes.DofsToVDofs(vd, vdofs);
         Amat->SetSubMatrix(vdofs, vdofs, M_inv(iel));
      }
   }
   Amat->Finalize();
   A.Reset(Amat);
}

L2ConformingProlongation::L2ConformingProlongation(
   const FiniteElementSpace &fes_ho_,
   const FiniteElementSpace &fes_lor_,
   Coefficient *Q_ho,
   Coefficient *Q_lor)
   : M_mixed(fes_ho_, fes_lor_, Q_ho, Q_lor),
     M_lor_inv(fes_lor_, Q_lor)
{
   M_mixed.Assemble(M_mixed_handle);
   M_lor_inv.AssembleInverse(M_lor_inv_handle);

   SparseMatrix *M_mixed_T = Transpose(*M_mixed_handle.As<SparseMatrix>());
   MmtMlorinvMm = RAP(*M_lor_inv_handle.As<SparseMatrix>(),
                      *M_mixed_T);
   delete M_mixed_T;

   D = new DSmoother(*MmtMlorinvMm);
}

Operator &L2ConformingProlongation::GetRestrictedMassMatrix()
{
   return *MmtMlorinvMm;
}

Operator &L2ConformingProlongation::GetMixedMassMatrix()
{
   return *M_mixed_handle;
}

Solver &L2ConformingProlongation::GetDiagonalPreconditioner()
{
   return *D;
}

L2ProjectionGridTransfer::L2Projection::L2Projection(
   const FiniteElementSpace &fes_ho_,
   const FiniteElementSpace &fes_lor_,
   Coefficient *Q_ho,
   Coefficient *Q_lor)
   : Operator(fes_lor_.GetTrueVSize(), fes_ho_.GetTrueVSize()),
     fes_ho(fes_ho_),
     fes_lor(fes_lor_),
     ho2lor(fes_ho, fes_lor),
     M_mixed(fes_ho, fes_lor, Q_ho, Q_lor),
     M_lor_inv(fes_lor, Q_lor)
{
   Mesh *mesh_ho = fes_ho.GetMesh();
   MFEM_VERIFY(mesh_ho->GetNumGeometries(mesh_ho->Dimension()) <= 1,
               "mixed meshes are not supported");

   // If the local mesh is empty, skip all computations
   if (mesh_ho->GetNE() == 0) { return; }

   const FiniteElement *fe_lor = fes_lor.GetFE(0);
   const FiniteElement *fe_ho = fes_ho.GetFE(0);
   ndof_lor = fe_lor->GetDof();
   ndof_ho = fe_ho->GetDof();

   // R will contain the restriction (L^2 projection operator) defined on
   // each coarse HO element (and corresponding patch of LOR elements)
   R.SetSize(ndof_lor*ho2lor.nref, ndof_ho, ho2lor.nel_ho);
   // P will contain the corresponding prolongation operator
   P.SetSize(ndof_ho, ndof_lor*ho2lor.nref, ho2lor.nel_ho);

   // Pre-allocate work matrices
   DenseMatrix RtMlor(ndof_ho, ndof_lor*ho2lor.nref);
   DenseMatrix RtMlorR(ndof_ho, ndof_ho);
   DenseMatrixInverse RtMlorR_inv(&RtMlorR);
   DenseMatrix Minv_lor(ndof_lor*ho2lor.nref, ndof_lor*ho2lor.nref);

   for (int iho=0; iho<ho2lor.nel_ho; ++iho)
   {
      for (int iref=0; iref<ho2lor.nref; ++iref)
      {
         // Assemble the low-order refined mass matrix and invert locally
         int ilor = ho2lor.el_table.GetRow(iho)[iref];
         Minv_lor.CopyMN(M_lor_inv.GetElementInverse(ilor),
                         iref*ndof_lor, iref*ndof_lor);
      }
      mfem::Mult(Minv_lor, M_mixed.GetElementMatrix(iho), R(iho));
      // Note that R = M_lor^{-1} * M_mixed, and so R^T M_lor = M_mixed^T
      RtMlor.Transpose(M_mixed.GetElementMatrix(iho));
      mfem::Mult(RtMlor, R(iho), RtMlorR);
      RtMlorR_inv.Factor();
      RtMlorR_inv.Mult(RtMlor, P(iho));
   }
}

void L2ProjectionGridTransfer::L2Projection::Mult(
   const Vector &x, Vector &y) const
{
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat(ndof_ho, vdim);
   DenseMatrix yel_mat(ndof_lor*ho2lor.nref, vdim);
   for (int iho=0; iho<fes_ho.GetNE(); ++iho)
   {
      fes_ho.GetElementVDofs(iho, vdofs);
      x.GetSubVector(vdofs, xel_mat.GetData());
      mfem::Mult(R(iho), xel_mat, yel_mat);
      // Place result correctly into the low-order vector
      for (int iref=0; iref<ho2lor.nref; ++iref)
      {
         int ilor = ho2lor.el_table.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            y.SetSubVector(vdofs, &yel_mat(iref*ndof_lor,vd));
         }
      }
   }
}

void L2ProjectionGridTransfer::L2Projection::Prolongate(
   const Vector &x, Vector &y) const
{
   int vdim = fes_ho.GetVDim();
   Array<int> vdofs;
   DenseMatrix xel_mat(ndof_lor*ho2lor.nref, vdim);
   DenseMatrix yel_mat(ndof_ho, vdim);
   for (int iho=0; iho<fes_ho.GetNE(); ++iho)
   {
      // Extract the LOR DOFs
      for (int iref=0; iref<ho2lor.nref; ++iref)
      {
         int ilor = ho2lor.el_table.GetRow(iho)[iref];
         for (int vd=0; vd<vdim; ++vd)
         {
            fes_lor.GetElementDofs(ilor, vdofs);
            fes_lor.DofsToVDofs(vd, vdofs);
            x.GetSubVector(vdofs, &xel_mat(iref*ndof_lor, vd));
         }
      }
      // Locally prolongate
      mfem::Mult(P(iho), xel_mat, yel_mat);
      // Place the result in the HO vector
      fes_ho.GetElementVDofs(iho, vdofs);
      y.SetSubVector(vdofs, yel_mat.GetData());
   }
}

const Operator &L2ProjectionGridTransfer::ForwardOperator()
{
   if (!F) { F = new L2Projection(dom_fes, ran_fes, Q_ho, Q_lor); }
   return *F;
}

const Operator &L2ProjectionGridTransfer::BackwardOperator()
{
   if (!B)
   {
      if (!F) { F = new L2Projection(dom_fes, ran_fes, Q_ho, Q_lor); }
      B = new L2Prolongation(*F);
   }
   return *B;
}

}
