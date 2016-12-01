from mshr import *
from dolfin import *

# Geometry
Lx   = 1.                                            # Zonal length
Ly   = 1.                                            # Meridonal length
n0   = 25                                            # Spatial resolution
geometry = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
mesh = generate_mesh(geometry, n0)

# Function and Vector Spaces
Vcg  = FunctionSpace(mesh,"CG",3)                    # CG elements for Streamfunction

# Boundary Conditions
def boundary(x, on_boundary):
	return on_boundary
BCs = DirichletBC(Vcg, 0.0, boundary)

# Physical parameters
beta = Constant("1.0")                               # Beta parameter
F    = Constant("1.0")                               # Burger number

# Trial and Test Functions
phi, psi = TestFunction(Vcg), TrialFunction(Vcg)

# Set up Eigenvalue Problem: QG Basin modes
a =  beta*phi*psi.dx(0)*dx 
m = -inner(grad(psi), grad(phi))*dx - F*psi*phi*dx

A, M = PETScMatrix(), PETScMatrix()
assemble(a, tensor = A)
assemble(m, tensor = M)
BCs.apply(M)

# Solve Eigenvalue Problem
eigenvalues_real, eigenvalues_imag = [], []
	
PETScOptions.set("eps_gen_non_hermitian")
PETScOptions.set("st_pc_factor_shift_type", "NONZERO")
PETScOptions.set("eps_type", "krylov")
PETScOptions.set("eps_largest_imaginary")        
PETScOptions.set("eps_tol", 1e-10)

n = 1
eigensolver = SLEPcEigenSolver(A, M)
eigensolver.solve(int(n))
print "Solving Eigenvalue Problem..."

nconv = eigensolver.get_number_converged()
print "Number of converged eigenpairs:", nconv

for i in range(nconv):
    lambda_real, lambda_imag, x_real, x_imag = eigensolver.get_eigenpair(i)
    eigenvalues_real.append(lambda_real)
    eigenvalues_imag.append(lambda_imag)
    		
# List Eigenvalues
print ("Real Eigenfrequencies:"), eigenvalues_real
print ("Imaginary Eigenfrequencies:"), eigenvalues_imag

eigenmodes_real, eigenmodes_imag = Function(Vcg), Function(Vcg)

eigenmodes_real.vector()[:], eigenmodes_imag.vector()[:] = x_real, x_imag

plot(eigenmodes_real)
interactive()
plot(eigenmodes_imag)
interactive()

# Plot Eigenmodes
#eigenmodes_imag.vector()[:] = x_imag
#soln.eigenmodes_imag.assign(eigenmodes_imag)

#eigenmodes_real.vector()[:] = x_real
#soln.eigenmodes_real.assign(eigenmodes_real)


"""
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

# Build a SLEPc problem 
E = SLEPc.EPS()
E.create(comm=mesh.comm)
E.setOperators(A, M)
E.solve()

nconv = E.getConverged()

print nconv

""" 
