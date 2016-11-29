from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Operators
zcross   = lambda u: as_vector((-u[1],    u[0]))
gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

# Geometry
Lx = 1.0
Ly = 1.0
n0 = 25
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 

# Function and Vector Spaces
Vdg = FiniteElement("BDM", mesh.ufl_cell(), 2)
Vcg = FiniteElement("DG", mesh.ufl_cell(), 1)
Z = FunctionSpace(mesh, MixedElement((Vdg, Vcg)))

# FJP: should remove this soon
Vcg2 = FunctionSpace(mesh,"CG",3)               
eta0 = Function(Vcg2, name='FreeSurface') 

# Trial and Test Functions
(u, eta)   = TrialFunctions(Z)
(v, lmbda) = TestFunctions(Z)

# Solution Space
sol = Function(Z)

# Parameters and Functions
r      = Constant("0.2")
beta   = Constant("1.0")
tau    = Constant("0.001")
F      = Constant("1.0")
fcor   = Function(Vcg2).interpolate(Expression("1.0+beta*x[1]", beta=beta))
Fwinds = Function(Vcg2).interpolate(Expression("(Ly*tau/pi)*sin((pi*(x[1]-Ly/2.0))/Ly)", tau = tau, Ly = Ly))

# Boundary Condtions
# Question 1: Does this impose u \cdot \hat h = 0?
bc = DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4))
#bc  = [DirichletBC(Z.sub(0), Constant((1, 0)), (2,4)),
#       DirichletBC(Z.sub(0), Constant((0, 1)), (1,3))]

# Define the weak form
a = r*inner(v,u)*dx - div(v)*eta*dx + lmbda*div(u)*dx + inner(fcor*v, zcross(u))*dx
L = Fwinds*v[0]*dx

# Set up SW inverter

solve(a == L, sol, bcs=bc)
# Question 2: Why does this solver not work?
"""
sw_problem = LinearVariationalProblem(a, L, sol, bcs=bc)
sw_solver = LinearVariationalSolver(sw_problem,
                                     solver_parameters={
                                         'ksp_type':'cg',
                                         'pc_type':'sor'
                                      })
"""

# Split
u, eta = sol.split()

# Energies
potential_energy = assemble(0.5*eta*eta*dx)
kinetic_energy = assemble(0.5*u*u*dx)

print kinetic_energy, potential_energy

"""
outfile = File("outputsw.pvd")
outfile.write(eta)

#eta_out = Function(Vcg, name='eta').assign(eta)
#outfile.write(eta_out)
"""
