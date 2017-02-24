// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Michael Rabinovich
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SLIM_H
#define SLIM_H

#include "igl_inline.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace igl
{

// Compute a SLIM map as derived in "Scalable Locally Injective Maps" [Rabinovich et al. 2016].
struct SLIMData
{
  // Input
  Eigen::MatrixXd V; // #V by 3 list of mesh vertex positions
  Eigen::MatrixXi F; // #F by 3/3 list of mesh faces (triangles/tets)
  enum SLIM_ENERGY
  {
    ARAP,
    LOG_ARAP,
    SYMMETRIC_DIRICHLET,
    CONFORMAL,
    EXP_CONFORMAL,
    EXP_SYMMETRIC_DIRICHLET
  };
  SLIM_ENERGY slim_energy;

  // Optional Input
  // soft constraints
  Eigen::VectorXi b;
  Eigen::MatrixXd bc;
  double soft_const_p;

  double exp_factor; // used for exponential energies, ignored otherwise
  bool mesh_improvement_3d; // only supported for 3d

  int slim_reflection_mode;
  bool skipInitialization = false;
  bool validPreInitialization = false;
  double expectedSurfaceAreaOfResultingMap = 0;

  // Output
  Eigen::MatrixXd V_o; // #V by dim list of mesh vertex positions (dim = 2 for parametrization, 3 otherwise)
  Eigen::MatrixXd oldUVs; // #V by dim list of mesh vertex positions (dim = 2 for parametrization, 3 otherwise)

  //weightmap for weighted parameterization
  bool withWeightedParameterization;
  Eigen::VectorXf weightmap;
  Eigen::VectorXf weightPerFaceMap;
  double weightInfluence;

  double energy; // objective value

  int nIterations;

  // INTERNAL
  Eigen::VectorXd M;
  double mesh_area;
  double avg_edge_length;
  int v_num;
  int f_num;
  double proximal_p;

  Eigen::VectorXd WGL_M;
  Eigen::VectorXd rhs;
  Eigen::MatrixXd Ri,Ji;
  Eigen::VectorXd W_11; Eigen::VectorXd W_12; Eigen::VectorXd W_13;
  Eigen::VectorXd W_21; Eigen::VectorXd W_22; Eigen::VectorXd W_23;
  Eigen::VectorXd W_31; Eigen::VectorXd W_32; Eigen::VectorXd W_33;
  Eigen::SparseMatrix<double> Dx,Dy,Dz;
  int f_n,v_n;
  bool first_solve;
  bool has_pre_calc = false;
  int dim;
};

// Compute necessary information to start using SLIM
// Inputs:
//		V           #V by 3 list of mesh vertex positions
//		F           #F by 3/3 list of mesh faces (triangles/tets)
//    b           list of boundary indices into V
//    bc          #b by dim list of boundary conditions
//    soft_p      Soft penalty factor (can be zero)
//    slim_energy Energy to minimize
	void slim_precompute(Eigen::MatrixXd& V,
                                Eigen::MatrixXi& F,
                                Eigen::MatrixXd& V_init,
                                SLIMData& data,
                                SLIMData::SLIM_ENERGY slim_energy,
                                Eigen::VectorXi& b,
                                Eigen::MatrixXd& bc,
                                double soft_p);

// Run iter_num iterations of SLIM
// Outputs:
//    V_o (in SLIMData): #V by dim list of mesh vertex positions
	Eigen::MatrixXd slim_solve(SLIMData& data, int iter_num);

	void adjustSoftConstraintViolationPenalty(SLIMData &data);

} // END NAMESPACE

#endif // SLIM_H