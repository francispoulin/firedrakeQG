from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc

def topetsc(A): 
    A.force_evaluation()
    return A.petscmat

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 50                                            # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 

# Function and Vector Spaces
Vcg  = FunctionSpace(mesh,"CG",3)                    # CG elements for Streamfunction

# Boundary Conditions
bc = DirichletBC(Vcg, 0.0, (1, 2, 3, 4))

# Physical parameters
beta = Constant("1.0")                               # Beta parameter
F    = Constant("1.0")                               # Burger number

# Trial and Test Functions
phi, psi = TestFunction(Vcg), TrialFunction(Vcg)

# Set up Eigenvalue Problem: QG Basin modes
a =  beta*phi*psi.dx(0)*dx 
m = -inner(grad(psi), grad(phi))*dx - F*psi*phi*dx

# Assemble Weak Form into a PETSc Matrix
# Impose Boundary Conditions on Mass Matrix
A = topetsc(assemble(a))
M = topetsc(assemble(m, bcs=[bc]))

# Build a SLEPc problem 
E = SLEPc.EPS()
E.create(comm=mesh.comm)
E.setOperators(A, M)
E.solve()

nconv = E.getConverged()

print nconv

# Create the results vectors
vr, wr = A.getVecs()
vi, wi = A.getVecs()
evr, evi = [], []
for i in range(nconv):
    k = E.getEigenpair(i, vr, vi)
    evr.append(k.real)
    evi.append(k.imag)

print evr, evi
#p = plot(vr)
#p.show()

