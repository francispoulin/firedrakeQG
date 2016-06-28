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

Lx   = 20.                                     # Zonal length
Ly   = 20.                                     # Meridonal length
n0   = 100                                     # Spatial resolution
mesh = PeriodicRectangleMesh(n0, n0, Lx, Ly,  direction="both", quadrilateral=True, reorder=None)

p   = 2
Vdg = FunctionSpace(mesh,"DG",p)               # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh,"CG",p)               # CG elements for Streamfunction
Vu  = VectorFunctionSpace(mesh,"DG",p)          # DG elements for velocity

# Physical parameters
U0   = Constant(0.01)
Lv   = Constant(1.0)
F    = Constant(0.0)         # Rotational Froude number
beta = Constant(0.0)      # beta plane coefficient

# Intial Conditions for PV

q_basic = Function(Vdg, name='pvbg').interpolate(Expression("U0*((pow(x[0]-Lx/2.,2)+pow(x[1]-Ly/2.,2)-Lv*Lv)/pow(Lv,3)-0.25*Lv*F)*exp(-(pow(x[0]-Lx/2.,2)+pow(x[1]-Ly/2.,2))/(Lv*Lv))", U0 = U0, Lv = Lv, F = F, Lx=Lx, Ly=Ly))
q0 = Function(Vdg, name='pv').assign(q_basic)
q0.dat.data[:] += 1e-7*np.random.randn(q0.dof_dset.size)
vol = Lx*Ly
qmean= assemble(q0*dx)/vol
q0 -= qmean

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

psi0 = Function(Vcg, name='streamfunction') # Streamfunctions for different time steps
psi1 = Function(Vcg)

# Time Stepping parameters
Dt   = 5.0                  # Time step
dt   = Constant(Dt)

# Set up PV inversion
psi = TrialFunction(Vcg)  # Test function
phi = TestFunction(Vcg)   # Trial function

# Build the weak form for the inversion
#FJP: change F to F**2
Apsi = (inner(grad(psi),grad(phi)) + F*psi*phi)*dx
Lpsi = -q1*phi*dx

#FJP: are these correct?
# Impose Dirichlet Boundary Conditions on the streamfunction
bc1 = [DirichletBC(Vcg, 0.0, 1),
       DirichletBC(Vcg, 0.0, 2)]

# Set up Elliptic inverter
psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0)
psi_solver = LinearVariationalSolver(psi_problem,
                                     solver_parameters={
        'ksp_type':'cg',
        'pc_type':'sor'
        })

# Make a gradperp operator
gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))

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

gradperp_h = lambda u: as_vector((-u.dx(1), u.dx(0)))
v = Function(Vu, name='velocity').project(gradperp_h(psi0))

q_pert = Function(Vdg, name='pv_pert').assign(q0-q_basic)
outfile = File("output.pvd")
outfile.write(q_pert)

t = 0.
T = 100000.
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
    potential_energy = assemble(0.5*F*psi0*psi0)
    total_energy = kinetic_energy + potential_energy
    enstrophy = assemble(0.5*q0*q0*dx)

    print t, kinetic_energy, potential_energy, total_energy, enstrophy
    
    tdump += 1
    if(tdump==dumpfreq): 
        tdump -= dumpfreq
        v.project(gradperp_h(psi0))
        #outfile.write(q0)
        q_pert = Function(Vdg, name='pv_pert').assign(q0-q_basic)
        outfile.write(q_pert)


