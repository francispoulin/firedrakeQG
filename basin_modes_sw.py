from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc

#OPERATORS
zcross = lambda i: as_vector((-i[1],i[0]))
#gradperp = lambda i: as_vector((-i.dx(1),i.dx(0)))

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 5                                             # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly,  reorder=None) 

#FUNCTION & VECTOR SPACES
Vdg = VectorFunctionSpace(mesh, "DG", 1)
Vcg = FunctionSpace(mesh, "CG", 2)
Z   = Vdg*Vcg

bc = DirichletBC(Z.sub(0), Constant((0, 0)), (1,2,3,4))

#TRIAL/TEST FUNCTIONS
(u, eta) = TrialFunctions(Z)
(v, lmbda) = TestFunctions(Z)

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
petsc_a = assemble(a, mat_type='aij', bcs=bc).M.handle
petsc_m = assemble(m, mat_type='aij').M.handle

num_eigenvalues = 5                                 # Number of eigenvalues

opts = PETSc.Options()

opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
#opts.setValue("eps_type", "lapack")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_spectrum", "target imaginary")
opts.setValue("eps_target", 3.14)
opts.setValue("eps_tol", 1e-10)
opts.setValue("st_type", "sinvert")

es = SLEPc.EPS().create(comm=COMM_WORLD)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_a, petsc_m)
es.setFromOptions()
es.solve()

nconv = es.getConverged()
print nconv

em_real, em_imag     = Function(Z), Function(Z)
u_real, u_imag       = Function(Vdg), Function(Vdg)
eta_real, eta_imag   = Function(Vcg), Function(Vcg) 
eval_real, eval_imag = [], []

# Find eigenvalues an eigenvectors
vr, vi = petsc_a.getVecs()
em_real, em_imag = Function(Z), Function(Z)
output = File('eigenmodes.pvd')
for i in range(nconv):
    with em_real.dat.vec as vr:
        with em_imag.dat.vec as vi:
            lam = es.getEigenpair(i, vr, vi)
            print lam
            eval_real.append(lam.real)
            eval_imag.append(lam.imag)

            u_real, eta_real = em_real.split()
            u_imag, eta_imag = em_imag.split()
            print u_real.dat.data.min(), u_real.dat.data.max()
            print u_imag.dat.data.min(), u_imag.dat.data.max()
            print eta_real.dat.data.min(), eta_real.dat.data.max()
            print eta_imag.dat.data.min(), eta_imag.dat.data.max()

        output.write(u_real, eta_real, time=i)
#        p = plot(eta_real)
#        p.show()
#        p.savefig('pr%s'%str(i))
#        p = plot(eta_imag)
#        p.show()
#        p.savefig('pi%s'%str(i))



