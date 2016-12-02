from firedrake import *
import numpy as np
import ufl
#import matplotlib.pyplot as plt 

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 50                                            # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 


# Function and Vector Spaces
Vcg  = FunctionSpace(mesh,"CG",3)                    # CG elements for Streamfunction
Vdg  = FunctionSpace(mesh,"DG",1)                    # DG elements for Potential Vorticity (PV)
Vcdg = VectorFunctionSpace(mesh,"DG",2)              # DG elements for velocity


# Boundary Conditions
bc = DirichletBC(Vcg, 0.0, (1, 2, 3, 4))


# Physical parameters
alpha =Constant("10.0")								#Penalty parameter
beta = Constant("1.0")                                 # Beta parameter
F    = Constant("1.0")                                 # Burger number
nu    = Constant("0.01")                                 # Lateral viscosity
tau  = Constant("0.001")                               # Wind Forcing
Fwinds = Function(Vcg).interpolate(Expression("-tau*cos(pi*(x[1]-0.5))", tau=tau))


# Define Cell Size Function
def CellSize(mesh):
    mesh.init()
    return 2.0 * ufl.Circumradius(mesh)
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0


# Define Facet Normal
def FacetNormal(mesh):
    mesh.init()
    return ufl.FacetNormal(mesh)
n = FacetNormal(mesh)


# Test Functions
phi, p, v = TestFunction(Vcg), TestFunction(Vdg), TestFunction(Vcdg)

# Trial Functions
psi, q, u = TrialFunction(Vcg), TrialFunction(Vdg), TrialFunction(Vcdg)


# Solution Functions
psi_soln, u_soln = Function(Vcg, name="Streamfunction"), Function(Vcdg)
q0_soln, qn_soln = Function(Vdg), Function(Vcdg)


# Define Winds and Weak Form
a = - nu*inner(div(grad(psi)), div(grad(phi)))*dx \
  + nu*inner(avg(div(grad(psi))), jump(grad(phi), n))*dS + nu*inner(jump(grad(psi), n), avg(div(grad(phi))))*dS \
  - nu*(alpha/h_avg)*inner(jump(grad(psi),n), jump(grad(phi),n))*dS \
  - nu*F*inner(nabla_grad(psi), nabla_grad(phi))*dx + beta*psi.dx(0)*phi*dx
L =  Fwinds*phi*dx


# Set up Elliptic inverter
psi_problem = LinearVariationalProblem(a, L, psi_soln, bcs=bc)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={
                                         'ksp_type':'preonly',
                                         'pc_type':'lu'
                                      })

# solve for streamfunction
psi_solver.solve()


# Plot Solution
p = plot(psi_soln)
p.show()


# Potential Energy
potential_energy = assemble(0.5*psi_soln*psi_soln*dx)
print potential_energy


# Output to a  file
outfile = File("outputQG.pvd")
outfile.write(psi_soln)
