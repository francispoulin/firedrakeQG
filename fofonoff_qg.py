from firedrake import *
import numpy as np

#OPERATORS
gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 20                                            # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 

# Function and Vector Spaces
Vcg  = FunctionSpace(mesh,"CG",3)                    # CG elements for Streamfunction
Vdg  = FunctionSpace(mesh,"DG",1)                    # DG elements for Potential Vorticity (PV)
Vcdg = VectorFunctionSpace(mesh,"DG",2)              # DG elements for velocity

# Boundary Conditions
bc = DirichletBC(Vcg, 0.0, (1, 2, 3, 4))

# Physical parameters and Winds
beta = Constant("1.0")                                 # Beta parameter
F    = Constant("1.0")                                 # Burger number
r    = Constant("0.3")                                 # Bottom drag
tau  = Constant("0.00001")                             # Wind Forcing
Fwinds = Function(Vcg).interpolate(Expression("-tau*cos(pi*(x[1]-0.5))", tau=tau))

# Test and Trial Functions
phi = TestFunction(Vcg)
psi = TrialFunction(Vcg)

# Solution Functions
psi_lin = Function(Vcg, name="Linear Streamfunction")
psi_non = Function(Vcg, name="Nonlinear Streamfunction")

# Define Weak Form
a = -r*inner(grad(phi), grad(psi))*dx - F*phi*psi*dx + beta*phi*psi.dx(0)*dx 
L =  Fwinds*phi*dx

# Set up Elliptic inverter for Linear Problem
linear_problem = LinearVariationalProblem(a, L, psi_lin, bcs=bc)
linear_solver = LinearVariationalSolver(linear_problem,
                                     solver_parameters={
                                         'ksp_type':'preonly',
                                         'pc_type':'lu'
                                      })

# Solve the linear problem
linear_solver.solve()

# Plot Solution
#p = plot(psi_lin)
#p.show()

# Use linear solution as a good guess
psi_non.assign(psi_lin)

# Define Weak Form
F =  -r*inner(grad(phi), grad(psi_non))*dx - F*phi*psi_non*dx + beta*phi*psi_non.dx(0)*dx \
     - inner(grad(phi),gradperp(psi_non))*div(grad(psi_non))*dx \
     - Fwinds*phi*dx

# Set up Elliptic inverter
nonlinear_problem = NonlinearVariationalProblem(F, psi_non, bcs=bc)
nonlinear_solver = NonlinearVariationalSolver(nonlinear_problem,
                                     solver_parameters={
                                         'snes_type': 'newtonls',
                                         'ksp_type':'preonly',
                                         'pc_type':'lu'
                                      })

# solve for streamfunction 
nonlinear_solver.solve()

# Plot Solution
p = plot(psi_non)
p.show()

"""

# Potential Energy

print potential_energy

# Output to a  file
outfile = File("outputQG.pvd")
outfile.write(psi_soln)
"""
