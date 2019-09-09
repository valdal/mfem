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

#ifndef MFEM_LOR
#define MFEM_LOR

#include "../config/config.hpp"
#include "fespace.hpp"

namespace mfem
{

struct HO2LORTransfer
{
   const CoarseFineTransformations &cf_tr;
   int nel_ho, nel_lor, nref;
   Table el_table;
   HO2LORTransfer(const FiniteElementSpace &fes_ho,
                  const FiniteElementSpace &fes_lor)
    : cf_tr(fes_lor.GetMesh()->GetRefinementTransforms())
   {
      nel_ho = fes_ho.GetNE();
      nel_lor = fes_lor.GetNE();
      nref = nel_lor/nel_ho;
      el_table.SetSize(nel_ho, nref);
      for (int ilor=0; ilor<nel_lor; ++ilor)
      {
         int iho = cf_tr.embeddings[ilor].parent;
         el_table.AddConnection(iho, ilor);
      }
      el_table.ShiftUpI();
   }
};

class LORMixedMassMatrix
{
   const FiniteElementSpace &fes_ho, &fes_lor;
   DenseTensor M_mixed;

   int ndof_ho, ndof_lor;
   HO2LORTransfer ho2lor;

public:
   LORMixedMassMatrix(
      const FiniteElementSpace &fes_ho_,
      const FiniteElementSpace &fes_lor_,
      Coefficient *Q_ho,
      Coefficient *Q_lor);
   DenseMatrix &GetElementMatrix(int iel);
   void Assemble(OperatorHandle &A);
};

class BlockDiagonalMassInverse
{
   const FiniteElementSpace &fes;
   DenseTensor M, M_inv;
public:
   BlockDiagonalMassInverse(const FiniteElementSpace &fes_, Coefficient *Q);
   DenseMatrix &GetElementMatrix(int iel);
   DenseMatrix &GetElementInverse(int iel);
   void AssembleInverse(OperatorHandle &A);
};

class L2ConformingProlongation
{
   LORMixedMassMatrix M_mixed;
   BlockDiagonalMassInverse M_lor_inv;
   OperatorHandle M_mixed_handle, M_lor_inv_handle;
   SparseMatrix *MmtMlorinvMm;
   DSmoother *D;
public:
   L2ConformingProlongation(
      const FiniteElementSpace &fes_ho_,
      const FiniteElementSpace &fes_lor_,
      Coefficient *Q_ho=nullptr,
      Coefficient *Q_lor=nullptr);

   Operator &GetRestrictedMassMatrix();
   Operator &GetMixedMassMatrix();
   Solver &GetDiagonalPreconditioner();

   ~L2ConformingProlongation()
   {
      delete MmtMlorinvMm;
      delete D;
   }
};

/** @brief Transfer data between a coarse mesh and an embedded refined mesh
    using L2 projection. */
/** The forward, coarse-to-fine, transfer uses L2 projection. The backward,
    fine-to-coarse, transfer is defined locally (on a coarse element) as
    B = (F^t M_f F)^{-1} F^t M_f, where F is the forward transfer matrix, and
    M_f is the mass matrix on the union of all fine elements comprising the
    coarse element. Note that the backward transfer operator, B, is a left
    inverse of the forward transfer operator, F, i.e. B F = I. Both F and B are
    defined in physical space and, generally, vary between different mesh
    elements.

    This class currently only fully supports L2 finite element spaces and fine
    meshes that are a uniform refinement of the coarse mesh. Generally, the
    coarse and fine FE spaces can have different orders, however, in order for
    the backward operator to be well-defined, the number of the fine dofs (in a
    coarse element) should not be smaller than the number of coarse dofs.

    If used on H1 finite element spaces, the transfer will be performed locally,
    and the value of shared (interface) degrees of freedom will be determined by
    the value of the last transfer to be performed (according to the element
    numbering in the finite element space). As a consequence, the mass
    conservation properties for this operator from the L2 case do not carry over
    to H1 spaces. */
class L2ProjectionGridTransfer : public GridTransfer
{
protected:
   /** Class representing projection operator between a high-order L2 finite
       element space on a coarse mesh, and a low-order L2 finite element space
       on a refined mesh (LOR). We assume that the low-order space, fes_lor,
       lives on a mesh obtained by refining the mesh of the high-order space,
       fes_ho. */
   class L2Projection : public Operator
   {
      const FiniteElementSpace &fes_ho;
      const FiniteElementSpace &fes_lor;

      int ndof_lor, ndof_ho;

      HO2LORTransfer ho2lor;

      LORMixedMassMatrix M_mixed;
      BlockDiagonalMassInverse M_lor_inv;

      DenseTensor R, P;

   public:
      L2Projection(const FiniteElementSpace &fes_ho_,
                   const FiniteElementSpace &fes_lor_,
                   Coefficient *Q_ho=nullptr,
                   Coefficient *Q_lor=nullptr);
      /// Perform the L2 projection onto the LOR space
      virtual void Mult(const Vector &x, Vector &y) const;
      /// Perform the mass conservative left-inverse prolongation operation.
      /// This functionality is also provided as an Operator by L2Prolongation.
      void Prolongate(const Vector &x, Vector &y) const;
      virtual ~L2Projection() { }
   };

   /** Mass-conservative prolongation operator going in the opposite direction
       as L2Projection. This operator is a left inverse to the L2Projection. */
   class L2Prolongation : public Operator
   {
      const L2Projection &l2proj;

   public:
      L2Prolongation(const L2Projection &l2proj_)
      : Operator(l2proj_.Width(), l2proj_.Height()),
        l2proj(l2proj_) { }
      void Mult(const Vector &x, Vector &y) const
      {
         l2proj.Prolongate(x, y);
      }
      virtual ~L2Prolongation() { }
   };

   L2Projection   *F; ///< Forward, coarse-to-fine, operator
   L2Prolongation *B; ///< Backward, fine-to-coarse, operator
   Coefficient *Q_ho, *Q_lor;

public:
   L2ProjectionGridTransfer(FiniteElementSpace &coarse_fes,
                            FiniteElementSpace &fine_fes,
                            Coefficient *Q_ho_=NULL,
                            Coefficient *Q_lor_=NULL)
      : GridTransfer(coarse_fes, fine_fes),
        F(NULL), B(NULL), Q_ho(Q_ho_), Q_lor(Q_lor_)
   { }

   virtual const Operator &ForwardOperator();

   virtual const Operator &BackwardOperator();

   ~L2ProjectionGridTransfer()
   {
      if (F) { delete F; }
      if (B) { delete B; }
   }
};

}

#endif
