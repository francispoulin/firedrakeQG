"""

Insert nice comments

"""

from firedrake import *
import sys

Lx = 4.e6
Ly = 4.e6
n0 = 100                                        # Spatial resolution
mesh = RectangleMesh(n0, n0, Lx, Ly, quadrilateral=True, reorder=None)

Vdg = FunctionSpace(mesh,"DG",1)               # DG elements for Potential Vorticity (PV)
Vcg = FunctionSpace(mesh,"CG",1)               # CG elements for Streamfunction
Vu = VectorFunctionSpace(mesh,"DG",1)          # DG elements for velocity

psi0 = Function(Vcg, name="streamfunction")      # Streamfunctions for different time steps
psi1 = Function(Vcg)

# Physical parameters
f0 = Constant(1.e-4)
H = Constant(500.)
g = Constant(9.81)
beta = Constant(2.e-11)      # beta plane coefficient
F    = Constant(f0*f0/g/H)         # Rotational Froude number
Dt   = 50000.                  # Time step
dt   = Constant(Dt)

# Initial Conditions for PV
x = SpatialCoordinate(mesh)
q0 = Function(Vdg, name="pv").interpolate(Constant(0.0))

dq1 = Function(Vdg)       # PV fields for different time steps
qh  = Function(Vdg)
q1  = Function(Vdg)

# Set up PV inversion
psi = TrialFunction(Vcg)  # Test function
phi = TestFunction(Vcg)   # Trial function

# Build the weak form for the inversion
Apsi = (inner(grad(psi),grad(phi)) + F*psi*phi)*dx
Lpsi = -q1*phi*dx

# Impose Dirichlet Boundary Conditions on the streamfunction
bc1 = [DirichletBC(Vcg, 0., 1),
       DirichletBC(Vcg, 0., 2),
       DirichletBC(Vcg, 0., 3),
       DirichletBC(Vcg, 0., 4)]

# Set up Elliptic inverter
psi_problem = LinearVariationalProblem(Apsi,Lpsi,psi0,bcs=bc1)
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

# Set up upwinding type method (??)
# ( dot(v, n) + |dot(v, n)| )/2.0
un = 0.5*(dot(gradperp(psi0), n) + abs(dot(gradperp(psi0), n)))

Vcg2 = VectorFunctionSpace(mesh, "CG", 2)
rho0 = Constant(1000.)
tau0 = Constant(0.2)
L = Constant(Ly)
tau_expr = as_vector([tau0*sin(pi*(x[1]-0.5*L)/L)/rho0/H, 0.0])
tau = Function(Vcg2, name='tau').interpolate(tau_expr)

# advection equation
q = TrialFunction(Vdg)
p = TestFunction(Vdg)
a_mass = p*q*dx
a_int  = (dot(grad(p), -gradperp(psi0)*q) + beta*p*psi0.dx(0))*dx
a_flux = (dot(jump(p), un('+')*q('+') - un('-')*q('-')) )*dS
a_source = p*(-tau[0].dx(1))*dx
arhs   = a_mass - dt*(a_int + a_flux - a_source)

bc2 = [DirichletBC(Vdg, 0., 3, method='geometric'),
       DirichletBC(Vdg, 0., 4, method='geometric')]

q_problem = LinearVariationalProblem(a_mass, action(arhs,q1), dq1, bcs=bc2)
q_solver  = LinearVariationalSolver(q_problem, 
                                    solver_parameters={
                                        'ksp_type':'cg',
                                        'pc_type':'sor'
                                    })

# diffusion equation
nu = Constant(1.6e5)
mu = Constant(100.*n0/Ly)
def get_flux_form(dS, M):
    fluxes = (-inner(2*avg(outer(q, n)), avg(grad(p)*M))
              - inner(avg(grad(q)*M), 2*avg(outer(p, n)))
              + mu*inner(2*avg(outer(q, n)), 2*avg(outer(p, n)*M)))*dS
    return fluxes

a = p*q*dx + dt*(dot(grad(p), grad(q)*nu)*dx + get_flux_form(dS, nu))
L = p*q*dx
diff_problem = LinearVariationalProblem(a, action(L, q0), q0, bcs=bc2)
diff_solver = LinearVariationalSolver(diff_problem)

outfile = File("outfile.pvd")
v = Function(Vu, name="velocity").project(gradperp(psi0))
courant_number = Function(Vdg, name="Courant number").project(dot(v,v)*Dt*n0*n0/(Lx*Ly))
outfile.write(q0, psi0, v, courant_number)

t = 0.
T = 200.*24.*60.*60.
dumpfreq = 36
tdump = 0

v0 = Function(Vu)

while(t < (T-Dt/2)):

    q1.assign(q0)
    psi_solver.solve()
    q_solver.solve()
    
    q1.assign(dq1)    
    psi_solver.solve()
    q_solver.solve()

    q1.assign(0.75*q0 + 0.25*dq1)
    psi_solver.solve()
    q_solver.solve()

    q0.assign(q0/3 + 2*dq1/3)
    diff_solver.solve()
    
    # Store solutions to xml and pvd
    t +=Dt
    print t

    tdump += 1
    if(tdump==dumpfreq):
        tdump -= dumpfreq
        v.project(gradperp(psi0))
        courant_number.project(dot(v,v)*Dt*n0*n0/(Lx*Ly))
        outfile.write(q0, psi0, v, courant_number)
