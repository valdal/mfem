#include "mfem.hpp"

using namespace mfem;

void visualize(GridFunction &gf, const std::string &caption);
double compute_mass(GridFunction &gf);
double compute_momentum(GridFunction &vel, GridFunction &rho);
double rho_fn(const Vector &x);
void vel_fn(const Vector &xvec, Vector &u);
void compare_mass(GridFunction &rho, GridFunction &rho_lor, bool vis);
void compare_momentum(GridFunction &vel, GridFunction &vel_lor,
                      GridFunction &rho, GridFunction &rho_lor, bool vis);

int Wx = 0, Wy = 0; // window position
int Ww = 350, Wh = 350; // window size
int offx = Ww+5, offy = Wh+25; // window offsets

int main(int argc, char **argv)
{
   // Parse command-line options.
   const char *mesh_file = "../../data/star.mesh";
   int order = 4;
   int ref_levels = 0;
   int lref = -1;
   int lor_order = 0;
   bool vis = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&ref_levels, "-r", "--reflevels", "Mesh refinement.");
   args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
   args.AddOption(&lor_order, "-lo", "--lor-order",
                  "LOR space order (polynomial degree, zero by default).");
   args.AddOption(&vis, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }

   if (lref < 0) { lref = order+1; }

   args.PrintOptions(std::cout);

   // Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // Create the low-order refined mesh
   int basis_lor = BasisType::GaussLobatto; // BasisType::ClosedUniform;
   Mesh mesh_lor(&mesh, lref, basis_lor);

   // Create spaces
   L2_FECollection fec_rho(order, dim);
   L2_FECollection fec_rho_lor(lor_order, dim);

   H1_FECollection fec_vel(order, dim);
   L2_FECollection fec_vel_lor(lor_order, dim);

   FiniteElementSpace fes_rho(&mesh, &fec_rho);
   FiniteElementSpace fes_vel(&mesh, &fec_vel, dim);
   FiniteElementSpace fes_rho_lor(&mesh_lor, &fec_rho_lor);
   FiniteElementSpace fes_vel_lor(&mesh_lor, &fec_vel_lor, dim);

   MFEM_ASSERT(fes_rho.GetTrueVSize() <= fes_rho_lor.GetTrueVSize(),
               "Low order refined space must have at least as many degrees of "
               "freedom as the high order space.");

   L2ProjectionGridTransfer l2_transfer(fes_rho, fes_rho_lor);

   ///////////////
   // HO to LOR //
   ///////////////
   std::cout << "\n\nMapping high-order to LOR using L^2 restriction\n\n";

   // Density
   GridFunction rho(&fes_rho), rho_lor(&fes_rho_lor);
   FunctionCoefficient rho_coeff(rho_fn);
   rho.ProjectCoefficient(rho_coeff);

   const Operator &R_rho = l2_transfer.ForwardOperator();
   const Operator &P_rho = l2_transfer.BackwardOperator();

   R_rho.Mult(rho, rho_lor);
   compare_mass(rho, rho_lor, vis);

   // Velocity
   GridFunction vel(&fes_vel), vel_lor(&fes_vel_lor);
   VectorFunctionCoefficient vel_coeff(dim, vel_fn);
   vel.ProjectCoefficient(vel_coeff);

   GridFunctionCoefficient rho_gf_coeff(&rho);
   GridFunctionCoefficient rho_lor_gf_coeff(&rho_lor);
   L2ProjectionGridTransfer l2_transfer_weighted_1(
      fes_vel, fes_vel_lor, &rho_gf_coeff, &rho_lor_gf_coeff);
   const Operator &R_vel = l2_transfer_weighted_1.ForwardOperator();

   R_vel.Mult(vel, vel_lor);
   compare_momentum(vel, vel_lor, rho, rho_lor, vis);

   ///////////////
   // LOR to HO //
   ///////////////
   std::cout << "\n\nMapping LOR to high-order using L^2 prolongation\n\n";

   Wx = 2.5*offx;
   Wy = 0;

   rho_lor.ProjectCoefficient(rho_coeff);
   P_rho.Mult(rho_lor, rho);
   compare_mass(rho, rho_lor, vis);

   L2ProjectionGridTransfer l2_transfer_weighted_2(
      fes_vel, fes_vel_lor, &rho_gf_coeff, &rho_lor_gf_coeff);
   const Operator &P_vel = l2_transfer_weighted_2.BackwardOperator();

   // Velocity
   std::cout << "\n\nMapping LOR to high-order velocity using density-weighted "
             << "*local* L^2 prolongation\n\n";
   vel_lor.ProjectCoefficient(vel_coeff);
   P_vel.Mult(vel_lor, vel);
   compare_momentum(vel, vel_lor, rho, rho_lor, vis);

   std::cout << "\n\nMapping LOR to high-order velocity using density-weighted "
             << "*global* L^2 prolongation\n\n";
   L2ConformingProlongation l2_conf_prolongation(
      fes_vel, fes_vel_lor, &rho_gf_coeff, &rho_lor_gf_coeff);
   Operator &MmtMlorinvMm = l2_conf_prolongation.GetRestrictedMassMatrix();
   Operator &Mm = l2_conf_prolongation.GetMixedMassMatrix();
   Solver &D = l2_conf_prolongation.GetDiagonalPreconditioner();

   GridFunction vel_rhs(&fes_vel);
   Mm.MultTranspose(vel_lor, vel_rhs);
   vel = 0.0;
   PCG(MmtMlorinvMm, D, vel_rhs, vel, 0, 1000, 1e-28);
   compare_momentum(vel, vel_lor, rho, rho_lor, vis);

   return 0;
}

double rho_fn(const Vector &x)
{
   int problem = 1;
   if (x.Size() == 1)
   {
      return sin(M_PI*2*x(0));
   }
   switch (problem)
   {
      case 1: // smooth field
         return 3.0+x(1)+0.25*cos(2*M_PI*x.Norml2());
      case 2: // cubic function
         return x(1)*x(1)*x(1) + 2*x(0)*x(1) + x(0);
      case 3: // sharp gradient
         return M_PI/2-atan(5*(2*x.Norml2()-1));
      case 4: // basis function
         return (x.Norml2() < 0.1) ? 1 : 0;
      default:
         return 1.0;
   }
}

void vel_fn(const Vector &xvec, Vector &u)
{
   if (xvec.Size() == 2)
   {
      double x = xvec[0];
      double y = xvec[1];
      u[0] = (x + y)*sin(2*M_PI*xvec.Norml2());
      u[1] = (x - y)*sin(2*M_PI*xvec.Norml2());
      return;
   }
   MFEM_ABORT("Only 2 dimensions currently");
}

void visualize(GridFunction &gf, const std::string &caption)
{
   char vishost[] = "localhost";
   int  visport   = 19916;

   Mesh *mesh = gf.FESpace()->GetMesh();

   socketstream sol_sockL2(vishost, visport);
   sol_sockL2.precision(8);
   sol_sockL2 << "solution\n" << *mesh << gf
              << "window_geometry " << Wx << " " << Wy << " " << Ww << " " << Wh
              << "keys jR\n"
              << "plot_caption '" << caption << "'"
              << "window_title '" << caption << "'" << std::flush;
}

double compute_mass(GridFunction &rho)
{
   ConstantCoefficient one(1.0);
   LinearForm lf(rho.FESpace());
   lf.AddDomainIntegrator(new DomainLFIntegrator(one));
   lf.Assemble();
   return lf(rho);
}

double compute_momentum(GridFunction &vel, GridFunction &rho, int vdim)
{
   double integ = 0.0;
   const FiniteElementSpace *fes = vel.FESpace();
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      int intorder = 3*fe->GetOrder() + 1;
      const IntegrationRule *ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      Vector vel_vals, rho_vals;
      vel.GetValues(i, *ir, vel_vals, vdim);
      rho.GetValues(i, *ir, rho_vals);
      ElementTransformation *T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         integ += ip.weight*T->Weight()*vel_vals[j]*rho_vals[j];
      }
   }
   return integ;
}

double compute_momentum(GridFunction &vel, GridFunction &rho)
{
   double integ = 0.0;
   for (int vd=0; vd<vel.FESpace()->GetVDim(); ++vd)
   {
      integ += pow(compute_momentum(vel, rho, vd+1), 2);
   }
   return sqrt(integ);
}

void compare_mass(GridFunction &rho, GridFunction &rho_lor, bool vis)
{
   double mass_ho = compute_mass(rho);
   double mass_lor = compute_mass(rho_lor);
   std::cout << std::setprecision(10);
   std::cout << "Mass (high order): " << mass_ho << '\n';
   std::cout << "Mass (LOR):        " << mass_lor << '\n';
   std::cout << "Relative error:    "
             << std::fabs((mass_ho-mass_lor)/mass_ho)
             << std::endl << std::endl;

   if (vis)
   {
      visualize(rho, "High-order density");
      Wx += offx;
      visualize(rho_lor, "LOR density");
      Wx -= offx;
      Wy += offy;
   }
}

void compare_momentum(GridFunction &vel, GridFunction &vel_lor,
                      GridFunction &rho, GridFunction &rho_lor, bool vis)
{
   double momentum_ho = compute_momentum(vel, rho);
   double momentum_lor = compute_momentum(vel_lor, rho_lor);
   std::cout << "Momentum (high order): " << momentum_ho << '\n';
   std::cout << "Momentum (LOR):        " << momentum_lor << '\n';
   std::cout << "Relative error:        "
               << std::fabs((momentum_ho-momentum_lor)/momentum_ho)
               << std::endl << std::endl;

   if (vis)
   {
      visualize(vel, "High-order velocity");
      Wx += offx;
      visualize(vel_lor, "LOR velocity");
      Wx -= offx;
      Wy += offy;
   }
}
