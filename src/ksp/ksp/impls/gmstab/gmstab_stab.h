/*
   Polynomial-stabilization coefficients for KSPGMSTAB. Direct port of
   gmstab_cpp/src/stab_coeffs.cpp / StabCoeffs.m.
*/
#pragma once
#include <petscsys.h>
#include <petscvec.h>

/* Compute the polynomial-step coefficients (length L) and the post-step
   residual norm.

   Inputs:
     r          : array of L+1 Vec columns, [r0, A*r0, A^2*r0, ...].
                  Each Vec must have the same global size N.
                  L is determined by the count (caller passes L+1 vecs).
     L          : polynomial degree (1 or 2 in GMstab).
     beta       : current ||r0||_2
     alpha      : maintaining-the-convergence threshold (default 35° = M_PI*7/36).

   Outputs:
     tau        : caller-allocated PetscScalar array of length L
     beta_new   : updated residual norm.

   Algorithm (matches MATLAB / C++ exactly):
     S = (L+1) x (L+1) Gram matrix S_ij = r_i^T r_j; S(0,0) replaced by beta^2.
     y0 = [1; -S_inner^-1 * S(2:L, 1); 0]
     yL = [0; -S_inner^-1 * S(2:L, L+1); 1]
        (S_inner is the (L-1)x(L-1) middle block; absent when L==1)
     kappa0 = sqrt(y0' * S * y0)
     kappaL = sqrt(yL' * S * yL)
     rho    = (yL' * S * y0) / (kappa0 * kappaL)
     if |rho| > sin(alpha):  delta = -kappa0/kappaL * rho
     else:                   forced-angle branch (see body)
     y0_new = y0 + delta * yL
     tau    = -y0_new(1:L)
     beta_new_sq = y0_new' * S * y0_new
     if beta_new_sq < 0: rebuild S(0,0) from r0^T r0 and retry; clamp to 0.
*/
PETSC_INTERN PetscErrorCode KSPGMSTABStabCoeffs_Private(Vec *r, PetscInt L,
                                                         PetscReal beta_in, PetscReal alpha,
                                                         PetscScalar *tau, PetscReal *beta_new);
