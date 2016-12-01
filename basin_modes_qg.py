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

num_eigenvalues = 2                                 # Number of eigenvalues

opts = PETSc.Options()

opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_largest_imaginary", None)
opts.setValue("eps_tol", 1e-10)

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)
es.setFromOptions()
es.solve()

nconv = es.getConverged()
print nconv

eigenmodes_real, eigenmodes_imag = Function(Vcg), Function(Vcg)

# Find eigenvalues an eigenvectors
eigenvaluer, eigenvaluei = [], []
eigefunctionr, eigenfunctioni = [], []

vr, vi = petsc_a.getVecs()
for i in range(nconv):
    lam = es.getEigenpair(i, vr, vi)
    eigenvaluer.append(lam.real)
    eigenvaluei.append(lam.imag)
    #eigenfunction.vector()[:] = vr
    vr_plot = vr
    #p = plot(vr_plot)
    #p.show()
    
eigenmodes_real.vector()[:], eigenmodes_imag.vector()[:] = vr, vi

p = plot(eigenmodes_real)
p.show()
p = plot(eigenmodes_imag)
p.show()

#print eigenvaluer
print eigenvaluei
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
