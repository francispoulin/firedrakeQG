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
n0   = 100                                        # Resolution in horizontal
nz   = 100                                        # Resolution in vertical
dz   = Lz/nz                                     

# Define a 2D mesh for the horiozntal
m = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="x", quadrilateral=True, reorder=None)
mesh = ExtrudedMesh(m, nz, layer_height=dz)

Vdg = FunctionSpace(mesh,"DG",1)               # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh,"CG",1)               # CG elements for Streamfunction
Vu  = VectorFunctionSpace(mesh,"DG",1, dim=2)  # DG elements for velocity

# Intial Conditions for PV
Uj = 1.0
yc = pi
Lj = Ly/10.
q0 = Function(Vdg, name='pv').interpolate(Expression("-2.*Uj/Lj*tanh((x[1]-yc)/Lj)/pow(cosh((x[1]-yc)/Lj),2)",Uj=Uj, Lj=Lj, yc=yc))
V = q0.function_space()
q0.dat.data[:] += 0.01*Uj*np.random.randn(V.dof_dset.size, *V.shape)

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

psi0 = Function(Vcg, name='streamfunction')      # Streamfunctions for different time steps
psi1 = Function(Vcg)

# Physical parameters
F    = Constant(0.0)         # Rotational Froude number
beta = Constant(0.0)      # beta plane coefficient
Dt   = 0.01                  # Time step
dt   = Constant(Dt)

# Set up PV inversion
psi = TrialFunction(Vcg)  # Test function
phi = TestFunction(Vcg)   # Trial function

# Build the weak form for the inversion
Apsi = (inner(grad(psi),grad(phi)))*dx
Lpsi = -q1*phi*dx

# Impose Dirichlet Boundary Conditions on the streamfunction
bc1 = [DirichletBC(Vcg, -Uj*Lj, 1),
       DirichletBC(Vcg,  Uj*Lj, 2)]

# Define nullspace
#nullspace = VectorSpaceBasis(constant=True)

# Set up Elliptic inverter
psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1)
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
a_flux = (dot(jump(p), un('+')*q('+') - un('-')*q('-')) )*(dS_h + dS_v)
arhs   = a_mass - dt*(a_int + a_flux)

q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1)
q_solver  = LinearVariationalSolver(q_problem, 
                                    solver_parameters={
                                        'ksp_type':'cg',
                                        'pc_type':'sor'
                                    })


gradperp_h = lambda u: as_vector((-u.dx(1), u.dx(0)))
v = Function(Vu, name='velocity').project(gradperp_h(psi0))

outfile = File("output.pvd")
outfile.write(q0, psi0, v)

t = 0.
T = 100.
dumpfreq = 100
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

    # calculate energy and enstrophy:
    kinetic_energy = assemble(0.5*dot(gradperp(psi0), gradperp(psi0))*dx)
    potential_energy = assemble(0.5*psi0*psi0*dx)
    total_energy = kinetic_energy + potential_energy
    enstrophy = assemble(0.5*q0*q0*dx)

    print t, kinetic_energy, potential_energy, total_energy, enstrophy
    
    tdump += 1
    if(tdump==dumpfreq):
        tdump -= dumpfreq
        v.project(gradperp_h(psi0))
        outfile.write(q0, psi0, v)
