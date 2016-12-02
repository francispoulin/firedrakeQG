from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc

#OPERATORS
zcross = lambda i: as_vector((-i[1],i[0]))
#gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 25                                            # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 

#FUNCTION & VECTOR SPACES
DG = VectorElement("DG", mesh.ufl_cell(), 1)
CG = FiniteElement("CG", mesh.ufl_cell(), 2)
G = FunctionSpace(mesh, MixedElement((DG, CG)))

#TRIAL/TEST FUNCTIONS
(u, eta) = TrialFunctions(G)
(v, lmbda) = TestFunctions(G)

# EIGENVALUE SOLs
eigenvalues_real = []
eigenvalues_imag = []

# Physical parameters
beta = Constant("1.0")                               # Beta parameter
F    = Constant("1.0")                               # Burger number
Ro   = Constant("1.0")
Fcor = Constant("0.0")
#Fcor = Expression("1.0 + beta*x[1]", beta=beta, degree=2)

# Weak form of SW Model
a = inner(Fcor*v,zcross(u))*dx \
	+ inner(v, grad(eta))*dx \
	- inner(u, grad(lmbda))*dx
m = Ro*inner(v, u)*dx + Ro*F*lmbda*eta*dx

# Assemble Weak Form into a PETSc Matrix
petsc_a = assemble(a).M.handle
petsc_m = assemble(m).M.handle

num_eigenvalues = 5                                 # Number of eigenvalues

opts = PETSc.Options()

opts.setValue("eps_gen_non_hermitian", None)
#opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
#opts.setValue("eps_spectrum", "target imaginary")
#opts.setValue("eps_largest_imaginary", None)
#opts.setValue("eps_tol", 1e-10)
#opts.setValue("spectral_shift", 3.14)

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)
es.setFromOptions()
es.solve()

#nconv = es.getConverged()
#print nconv

"""
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
"""
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
