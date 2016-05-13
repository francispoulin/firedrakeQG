"""

Insert nice comments

This document was created using:

pandoc -f latex qg_1layer_wave.tex -t rst > qg_1layer_wave.rst

Can make an html version to view locally using:

pandoc qg_1layer_wave.tex -s --mathjax -o qg_1layer_wave.html
"""

from firedrake import *
import numpy as np
import sys

Lx   = 2.*pi                                     # Zonal length
Ly   = 2.*pi                                     # Meridonal length
Lz   = 2.*pi                                     # Meridonal length
n0   = 20                                        # Resolution in horizontal
nz   = 20                                        # Resolution in vertical
dz   = Lz/nz                                     

# Define a 2D mesh for the horiozntal
m = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both", quadrilateral=True, reorder=None)
#FJP: extrud the mesh in the vertical
mesh = ExtrudedMesh(m, nz, layer_height=dz)

Vdg = FunctionSpace(mesh,"DG",1)               # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh,"CG",1)               # CG elements for Streamfunction
#FJP: make the velocity a 2D field 
Vu  = VectorFunctionSpace(mesh,"DG",1, dim=2)  # DG elements for velocity

# Intial Conditions for PV
q0 = Function(Vdg).interpolate(Expression("0.0"))
q0.dat.data[:] += 0.1*np.random.randn(q0.dof_dset.size)

q0.dat.data[:] += 0.1*np.random.randn(q0.dof_dset.size)
Area = Lx*Ly
#print assemble(q0*dx)/Area
qmean = assemble(q0*dx)/Area
q0 -= qmean
#print assemble(q0*dx)/Area

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

psi0 = Function(Vcg)      # Streamfunctions for different time steps
psi1 = Function(Vcg)

# Physical parameters
F    = Constant(1.0)         # Rotational Froude number
beta = Constant(0.0)      # beta plane coefficient
Dt   = 1.0                  # Time step
dt   = Constant(Dt)

# Set up PV inversion
psi = TrialFunction(Vcg)  # Test function
phi = TestFunction(Vcg)   # Trial function

# Build the weak form for the inversion
#FJP: I removed the Froude number since the z-derivative is now in inner and grd
Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q1*phi*dx

# Impose Dirichlet Boundary Conditions on the streamfunction
bc1 = [DirichletBC(Vcg, 0.0, 1),
       DirichletBC(Vcg, 0.0, 2)]

# Define nullspace
#nullspace = VectorSpaceBasis(constant=True)

# Set up Elliptic inverter
psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })

# Make a gradperp operator
gradperp = lambda u: as_vector((-u.dx(1), u.dx(0), 0.0))

# Set up Strong Stability Preserving Runge Kutta 3 (SSPRK3) method

# Mesh-related functions
n = FacetNormal(mesh)

# Set up upwinding type method: ( dot(v, n) + |dot(v, n)| )/2.0
un = 0.5*(dot(gradperp(psi0), n) + abs(dot(gradperp(psi0), n)))

# advection equation
q = TrialFunction(Vdg)
p = TestFunction(Vdg)
a_mass = p*q*dx
a_int  = (dot(grad(p), -gradperp(psi0)*q) + beta*p*psi0.dx(0))*dx
a_flux = (dot(jump(p), un('+')*q('+') - un('-')*q('-')) )*dS
arhs   = a_mass - dt*(a_int + a_flux)

q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1)
q_solver  = LinearVariationalSolver(q_problem, 
                                    solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })


qfile = File("q.pvd")
qfile << q0
psifile = File("psi.pvd")
psifile << psi0
vfile = File("v.pvd")
gradperp_h = lambda u: as_vector((-u.dx(1), u.dx(0)))
v = Function(Vu).project(gradperp_h(psi0))
vfile << v

t = 0.
T = 10.
dumpfreq = 50
tdump = 0

v0 = Function(Vu)

while(t < (T-Dt/2)):

    # Compute the streamfunction for the known value of q0
    q1.assign(q0)
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(1)
    q1.assign(dq1)    
    psi_solver.solve()
    q_solver.solve()

    # Find intermediate solution q^(2)
    q1.assign(0.75*q0 + 0.25*dq1)
    psi_solver.solve()
    q_solver.solve()

    # Find new solution q^(n+1)
    q0.assign(q0/3 + 2*dq1/3)
    
    # Store solutions to xml and pvd
    t +=Dt
    print t

    tdump += 1
    if(tdump==dumpfreq):
        tdump -= dumpfreq
        qfile.write(q0)
        #psifile.write(psi0)
        #v.project(gradperp(psi0))
        #vfile.write(v)
