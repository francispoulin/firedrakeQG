from dolfin import *
from mshr import *
import numpy as np

#OPERATORS
zcross = lambda i: as_vector((-i[1],i[0]))

#BASIN & MESH
length = 1.0
width  = 1.0#np.sqrt(2)
resolution = 25
geometry  = Rectangle(Point(0.0, 0.0), Point(width, length))
mesh = generate_mesh(geometry, resolution)

#FUNCTION & VECTOR SPACES
DGv = FiniteElement("BDM", mesh.ufl_cell(), 2)
CG = FiniteElement("DG", mesh.ufl_cell(), 1)
G = FunctionSpace(mesh, MixedElement((DGv, CG)))

#TRIAL/TEST FUNCTIONS
(u, eta) = TrialFunctions(G)
(v, lmbda) = TestFunctions(G)

#SOLUTION SPACES
sol = Function(G)

#VARIABLES
r = Constant("0.1")
beta = Constant("1.0")
tau = Constant("0.001")
F = Constant("0.1")
Ed = Constant("0.0")
fcor = Expression("1.0+beta*x[1]", beta=beta, degree=2)

class Source(Expression):
	def eval(self, values, x):
   		#values[0] = -tau*sin((pi*x[1])/(2*length))
   		values[0] = (length*tau/pi)*sin((pi*(x[1]-length/2.0))/length)
#Fwinds = Source(tau = tau, degree = 2)

Fwinds = Expression("(length*tau/pi)*sin((pi*(x[1]-length/2.0))/length)", tau = tau, length = length, degree=2)

# ==========================================================================

#CGs = FunctionSpace(mesh,"CG",3)
#tmp = project(betay, CGs)
#plot(tmp)
#interactive()

#BOUNDARY CONDITIONS

# Define function G such that G \cdot n = g
class BoundarySource(Expression):
       def __init__(self, mesh, **kwargs):
           self.mesh = mesh
       def eval_cell(self, values, x, ufc_cell):
           cell = Cell(self.mesh, ufc_cell.index)
           n = cell.normal(ufc_cell.local_facet)
           g = 0.0
           values[0] = g*n[0]
           values[1] = g*n[1]
       def value_shape(self):
           return (2,)

GS = BoundarySource(mesh, degree=2)

# Define essential boundary
def boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

BCs = DirichletBC(G.sub(0), GS, boundary)

# ==========================================================================

F = inner(fcor*v, zcross(u))*dx + \
    -div(v)*eta*dx \
    + r*inner(v, u)*dx + lmbda*div(u)*dx - \
    Fwinds*v[0]*dx + F*eta*lmbda*dx
#	Ed*inner(nabla_grad(lmbda), nabla_grad(eta))*dx + \

#inner(v, grad(eta))*dx + \

a, L = lhs(F), rhs(F)
solve(a==L, sol, BCs)

"""
#u, eta = sol.split()

# Assemble system
A = assemble(a)
b = assemble(L)

# Create Krylov solver
solver = PETScKrylovSolver("cg")  #????
solver.set_operator(A)

# Create vector that spans the null space and normalize
null_vec = Vector(sol.vector())
G.dofmap().set(null_vec, 1.0)
null_vec *= 1.0/null_vec.norm("l2")

# Create null space basis object and attach to PETSc matrix
null_space = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(null_space)

# Orthogonalize RHS vector b with respect to the null space (this
# gurantees a solution exists)
null_space.orthogonalize(b);

# Solve
solver.solve(sol.vector(), b)
"""

# Split
u, eta = sol.split()

File("u.pvd") << u
plot(u)
interactive()

File("eta.pvd") << eta
plot(eta)
interactive()
