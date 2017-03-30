# IMPORT LIBRARIES

from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import ufl, os

import matplotlib.pyplot as plt

# DIFFERENTIAL OPERATORS

gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

# GEOMETRY

n0   = 25
Ly   = 1.0
Lx   = np.sqrt(2)
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder = None)

# FUNCTION AND VECTOR SPACES

V1 = FunctionSpace(mesh, "CG", 3)
V2 = FunctionSpace(mesh, "BDM", 2)
V3 = FunctionSpace(mesh, "DG", 1)
Z = V1 * V2 * V3

# TEST/TRIAL FUNCTIONS
(psi, u, q) = TrialFunctions(Z)
(phi, v, p) = TestFunctions(Z)

# BOUNDARY CONDITIONS

bc = DirichletBC(Z.sub(0), 0.0, "on_boundary")

# MODEL PARAMETERS

beta   = Constant("1.0")
F      = Constant("1.0")
nu     = Constant("0.0")
r      = Constant("0.1")
tau    = Constant("0.001")
Gwinds = Function(V1).interpolate(Expression("-tau*cos(pi*(x[1]-0.5))", tau=tau))

# ==========================================================================
# SOLVE LINEAR PROBLEM
# ==========================================================================

# CREATE MIXED SOLUTION SPACE
soln0 = Function(Z, name="Linear Solution")

# DEFINE WEAK FORM
Glin =   ( phi*q + inner(grad(phi), grad(psi)) + F*phi*psi )*dx \
       + ( inner(v,u) - inner(v,gradperp(psi)) )*dx \
       + ( beta*p*psi.dx(0) + r*p*q )*dx \
       - ( Gwinds*phi )*dx

# SET-UP ELLIPTIC INVERTER
linear_problem = LinearVariationalProblem(lhs(Glin), rhs(Glin), soln0, bcs=bc)
linear_solver = LinearVariationalSolver(linear_problem, 
    solver_parameters={
            'ksp_type':'preonly',
            'ksp_monitor': True,
            'matnest': False,
            'mat_type': 'aij',
            'pc_type': 'lu'
                            })

# SOLVE LINEAR PROBLEM AND SPLIT SOLUTION
linear_solver.solve()
psi0, u0, q0 = soln0.split()

# PLOT SOLUTION
plot(psi0)
plt.xlabel('Zonal')
plt.ylabel('Meridional')
plt.title('Linear Stommel Solution')
plt.xlim([0,Lx])
plt.ylim([0,Ly])
plt.show()

# ==========================================================================
# SOLVE NONLINEAR PROBLEM
# ==========================================================================

# CREATE MIXED SOLUTION SPACE
soln1 = Function(Z, name="Nonlinear Solution")

# USE LINEAR SOLUTION AS GOOD GUESS AND SPLIT
soln1.assign(soln0)
psi1, u1, q1 = soln1.split()

# DEFINE WEAK FORM
Gnon =   ( phi*q1 + inner(grad(phi), grad(psi1)) + F*phi*psi1 )*dx \
    + ( inner(v,u1) - inner(v,gradperp(psi1)) )*dx \
    + ( beta*p*psi1.dx(0) + r*p*q1 )*dx \
    - ( Gwinds*phi )*dx \
    - ( inner(grad(p),u1)*q1 )*dx

# SET-UP ELLIPTIC INVERTER
nonlinear_problem = NonlinearVariationalProblem(Gnon, soln1, bcs=bc)
nonlinear_solver = NonlinearVariationalSolver(nonlinear_problem, 
                                              solver_parameters={
                                                  'ksp_type':'preonly',
                                                  'ksp_monitor': True,
                                                  'matnest': False,
                                                  'mat_type': 'aij',
                                                  'pc_type':'lu',
                                                  'snes_converged_reason': True,
                                                  'ksp_converged_reason': True
                                              })

# SOLVE NONLINEAR PROBLEM
nonlinear_solver.solve()

# PLOT SOLUTION
plot(psi1)
plt.xlabel('Zonal')
plt.ylabel('Meridional')
plt.title('Nonlinear Stommel Solution')
plt.xlim([0,Lx])
plt.ylim([0,Ly])
plt.show()
