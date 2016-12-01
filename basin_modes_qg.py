from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 25                                            # Spatial resolution
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
petsc_a = assemble(a).M.handle
petsc_m = assemble(m, bcs=bc).M.handle

num_eigenvalues = 5                                 # Number of eigenvalues

# Create solver
es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setProblemType(es.ProblemType.GHEP)              # Generalized eigenvalue problem
es.setWhichEigenpairs(es.Which.SMALLEST_IMAGINARY)  # type of eigenvalues to find
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)                   # Build Operators
es.solve()                                          # solve system

nconv = es.getConverged()
print nconv

# Find eigenvalues an eigenvectors
eigenvalue = []
eigefunction = []
vr, vi = petsc_a.getVecs()
for i in range(num_eigenvalues):
    lam = es.getEigenpair(i, vr, vi)
    print lam
    #eigenvalue[i] = lam.real
    #eigenfunction[i].vector()[:] = vr

#print eigenvalue[0], eigenvalue[1]

""""
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

"""
