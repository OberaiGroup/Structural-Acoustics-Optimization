"""
@author: Harisankar
"""

from fenics import *
from multiphenics import *
import numpy as np
import scipy.optimize as sp
#from scipy.optimize import minimize
from mshr import *



#Iterative solve#############
# Set PETSc solve type (conjugate gradient) and preconditioner
PETScOptions.set("ksp_type", "gmres") # cg, gmres
PETScOptions.set("pc_type", "ilu") # ilu, gamg
PETScOptions.set("pc_factor_levels", 2) # ILU fill_level

# Set the solver tolerance
PETScOptions.set("ksp_rtol", 1.0e-7)
PETScOptions.set("ksp_atol", 1.0e-8)

# Set the maximum solver iterations
PETScOptions.set("ksp_max_it", 1200)

# Allowing non zero initial guess
PETScOptions.set("ksp_initial_guess_nonzero", 1)

# Print PETSc solver configuration
#PETScOptions.set("ksp_view")
#PETScOptions.set("ksp_monitor")

# Create Krylov solver and set operator
solver = PETScKrylovSolver()
solver.set_from_options() # Set PETSc options on the solver



# Helper function to generate subdomain restriction based on a gmsh subdomain id
def generate_subdomain_restriction(mesh, subdomains, subdomain_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] == subdomain_id:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction
    
# Helper function to generate interface restriction based on a pair of gmsh subdomain ids
def generate_interface_restriction(mesh, subdomains, subdomain_ids):
    assert isinstance(subdomain_ids, set)
    assert len(subdomain_ids) is 2
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain ids (except the mesh function corresponding to dimension D, as it is trivially false)
    for f in facets(mesh):
        subdomains_ids_f = set(subdomains[c] for c in cells(f))
        assert len(subdomains_ids_f) in (1, 2)
        if subdomains_ids_f == subdomain_ids:
            restriction[D - 1][f] = True
            for d in range(D - 1):
                for e in entities(f, d):
                    restriction[d][e] = True
    # Return
    return restriction

#### Mesh generation #################

domain = Ellipse(Point(0, 0), 1.25 , 1)
domain.set_subdomain(1, Ellipse(Point(0, 0), 1.25 , 1))
domain.set_subdomain(2, Ellipse(Point(0, 0), 1.15 , 0.9))
domain.set_subdomain(3, Ellipse(Point(0, 0), 1.00 , 0.75))
domain.set_subdomain(4, Circle(Point(0.4, 0), 0.2))

N = 60
mesh = generate_mesh(domain, N)

############### Mesh refinement ########################
refine_domain = CompiledSubDomain("(x[0]-0)*(x[0]-0)/(a*a)+(x[1]-0)*(x[1]-0)/(b*b) >= (1 - 5e-3)", a = 1, b = 0.75)
r_markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)
refine_domain.mark(r_markers, True)
refinedMesh = refine(mesh,r_markers)
mesh = refinedMesh
#refinedMesh = refine(mesh,r_markers)
#mesh = refinedMesh
########################################################

File("output/mesh.pvd") << mesh
N_nodes = mesh.num_vertices()
N_cells = mesh.num_cells()
print("Number of nodes = ",N_nodes)
print("Number of cells = ",N_cells)

#######################################


# Domain definition ##################

celltag = MeshFunction("size_t", mesh, mesh.topology().dim())
celltag.set_all(2)
class shell(SubDomain):
    def inside(self, x, on_boundary):
        a = 1; b=0.75; tol = 1e-3
        return pow(x[0] - 0, 2)/(a*a) + pow(x[1] - 0, 2)/(b*b) >= 1-tol
shell = shell()
for c in cells(mesh):
    if shell.inside(c.midpoint(), True):
        celltag[c.index()] = 1

# Restrictions #
# Generate restriction corresponding to interior subdomain (celltag = 1,2)
solid = generate_subdomain_restriction(mesh, celltag, 1)
fluid = generate_subdomain_restriction(mesh, celltag, 2)        

#
class ctrl(SubDomain):
    def inside(self, x, on_boundary):
        a = 1; b=0.75; tol = 1e-3
        return (pow(x[0] - 0, 2)/(a*a) + pow(x[1] - 0, 2)/(b*b) >= 1-tol)   
ctrl= ctrl()
for c in cells(mesh):
    if ctrl.inside(c.midpoint(), True):
        celltag[c.index()] = 3
#        
class shell(SubDomain):
    def inside(self, x, on_boundary):
        a = 1.15; b=0.9; tol = 1e-3
        return pow(x[0] - 0, 2)/(a*a) + pow(x[1] - 0, 2)/(b*b) >= 1-tol
shell = shell()
for c in cells(mesh):
    if shell.inside(c.midpoint(), True):
        celltag[c.index()] = 1
#        
class roi(SubDomain):
    def inside(self, x, on_boundary):
        r = 0.2;tol = 1e-3
        return pow(x[0] - 0.4, 2) + pow(x[1] - 0, 2) <= (r-tol)*(r-tol) 
roi = roi()
for c in cells(mesh):
    if roi.inside(c.midpoint(), True):
        celltag[c.index()] = 4
#        

#Defining boundaries
facettag = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        a = 1; b=0.75;tol=6e-3
        return (pow(x[0] - 0, 2)/(a*a) + pow(x[1] - 0, 2)/(b*b) >= 1-tol) and (pow(x[0] - 0, 2)/(a*a) + pow(x[1] - 0, 2)/(b*b) <= 1+tol)

class Traction(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[1]<=0.2) and (x[1]>=-0.2) and (x[0]<=0) and on_boundary)
    
facettag.set_all(0)
traction = Traction()
traction.mark(facettag, 1)
interface = Interface()
interface.mark(facettag, 2)



# FUNCTION SPACES ####################################
p_order = 1
# Function space
P1 = VectorElement("Lagrange", mesh.ufl_cell(), p_order)
EL = P1 * P1
V = FunctionSpace(mesh, EL)
M1 = FunctionSpace(mesh, P1)
P2 = FiniteElement("Lagrange", mesh.ufl_cell(), p_order)
EL = P2 * P2
Q = FunctionSpace(mesh, EL)
M2 = FunctionSpace(mesh, P2)

# Block function space
W = BlockFunctionSpace([V, Q], restrict=[solid,fluid])

# TRIAL/TEST FUNCTIONS #
UP = BlockTrialFunction(W)
(u, p) = block_split(UP)
(ur, ui) = (as_vector((u[0], u[1])), as_vector((u[2], u[3])))
(pr, pi) = as_vector((p[0], p[1]))
VQ = BlockTestFunction(W)
(v, q) = block_split(VQ)
(vr, vi) = (as_vector((v[0], v[1])), as_vector((v[2], v[3])))
(qr, qi) = as_vector((q[0], q[1]))



####### Constants  #########################################
#Material properties: Steel for solid and water for fluid
omega = 8000 #forcing frequency
kappa = 2.15e9 #bulk modulus of fluid      
kappar = Constant(kappa)#1.28E5
kappai = Constant(1e-8*omega)#8.9e-4*omega  bulk viscocity *omega = 1.81e-5*3000=5.43E-2
rhof = Constant(997)#1.25 #density of fluid
omega = Constant(omega)#forcing frequency

##Weights for objective function##
beta1 = Constant(1e-3)# weight for the regularization term
beta2 = Constant(1e-5)
alpha = Constant(1) #weight of the minimization functional term
c = Constant(1e-3)
n = 6
eps = (2**n +1)/(2**n)
eps = Constant(eps)

'''# Air as fluid
omega = 7500 #forcing frequency
kappa = 1.28e5 #bulk modulus of fluid      
kappar = Constant(kappa)#1.28E5
kappai = Constant(1e-8*omega)#8.9e-4*omega  bulk viscocity *omega = 1.81e-5*3000=5.43E-2
rhof = Constant(1.25)#1.25 #density of fluid
#'''

rhos = Constant(7800)#density of solid
E = 2E11 #Young's modulus steel
nu = 0.3 # poisson's ratio
lamda = E/(2*(1-nu)) - E/(2*(1+nu))#65.9e9
lamdar = Constant(lamda)# Lame's 1st coefficient
lamdai = Constant(1e-8*omega)

mu0 = E/(2*(1+nu))#76.9e9
mu0 = Constant(mu0)
mur = Constant(1)# Lame's 2nd coefficient normalized
mui = Constant(1e-8*omega/mu0)

mua = 0.1# material A 
mub = 1# materail B   
mua = Constant(mua)# material A 
mub = Constant(mub)# materail B

psi = Constant(1)
psi = Expression(("(1+(x[0]+c1)/c2)"),c1 = 1.25,c2 = 2.5,
                  element = M2.ufl_element())
psi = interpolate(psi,M2)


Pext = Constant(100)# External time harmonic pressure load
L = Constant(1.0)# Scale factor for the length of the domain
Uo = L*Pext/mu0 # Nominal deformation


wt = Expression(("((x[0]-0)*(x[0]-0)/(a1*a1)+(x[1]-0)*(x[1]-0)/(b1*b1) >= (1 - tol)) && \
                  ((x[0]-0)*(x[0]-0)/(a2*a2)+(x[1]-0)*(x[1]-0)/(b2*b2) <= (1 + tol))"), 
                  a1 = 1,b1 = 0.75, a2 = 1.15,b2 = 0.9,tol = 1e-3,
                  element = M2.ufl_element())
wt = interpolate(wt,M2)
mur = interpolate(mur,M2)
#mur.vector()[:] = mur.vector()[:]*wt.vector()[:]
File("output/wt.pvd") << wt



# DEFINE Forcing function #
fur = Expression(("0.0","0.0"), element = M1.ufl_element())
fui = Expression(("0.0","0.0"), element = M1.ufl_element())
o = Expression(("0.0"), element = M2.ufl_element())


##### Non dimensional constants ###################
a1 = mu0/(rhos*(omega**2)*(L**2))
#a2 = lamda/mu0
a2r = lamdar/mu0
a2i = lamdai/mu0
#a3 = fu/(rhos*(omega**2)*Uo)
a3r = fur/(rhos*(omega**2)*Uo)
a3i = fui/(rhos*(omega**2)*Uo)
#a4 = (rhof*(omega**2)*(L**2))/kappa
a4r = (rhof*(omega**2)*(L**2))*kappar/(kappar**2+kappai**2)
a4i = -(rhof*(omega**2)*(L**2))*kappai/(kappar**2+kappai**2)
a5 = Pext/(rhof*(omega**2)*L*Uo)
a6 = Pext*L/(mu0*Uo)
### u, p and mu are non dimensionalised



## Defining integral elements and normal ##################################
dx = Measure("dx", domain = mesh, subdomain_data = celltag)
ds = Measure("ds", domain = mesh, subdomain_data = facettag)
dS = Measure("dS", domain = mesh, subdomain_data = facettag)

#    n = FacetNormal(mesh) # for obtaining interface normal
normal = Expression(("-(b*b)*x[0]/sqrt((b*b*b*b)*x[0]*x[0] + (a*a*a*a)*x[1]*x[1])",
                     "-(a*a)*x[1]/sqrt((b*b*b*b)*x[0]*x[0] + (a*a*a*a)*x[1]*x[1])"),
                    a = 1,b = 0.75, element = M1.ufl_element())

n_outer = Expression(("(b*b)*x[0]/sqrt((b*b*b*b)*x[0]*x[0] + (a*a*a*a)*x[1]*x[1])",
                      "(a*a)*x[1]/sqrt((b*b*b*b)*x[0]*x[0] + (a*a*a*a)*x[1]*x[1])"),
                     a = 1.25,b = 1, element = M1.ufl_element())


# ASSEMBLING GALERKIN EQUATION ####################################
I = Identity(2)
mur = (mua+psi*(mub-mua))
#two = Constant(2)
#one = Constant(1)
#mur = (mua+ eps*((two*psi)**n)*((one+(two*psi)**n)**(-1))*(mub-mua))
sigmarr = 2*mur*0.5*(grad(ur)+grad(ur).T) + mur*a2r*div(ur)*I 
sigmari = 2*mur*0.5*(grad(ui)+grad(ui).T) + mur*a2r*div(ui)*I 
sigmair = 2*mui*0.5*(grad(ur)+grad(ur).T) + a2i*div(ur)*I 
sigmaii = 2*mui*0.5*(grad(ui)+grad(ui).T) + a2i*div(ui)*I 

#Bilinear and linear terms
Bs = a1*(inner(grad(vr), sigmarr-sigmaii) - inner(grad(vi), sigmair+sigmari)) - (dot(vr,ur) - dot(vi,ui))

Csf1 = avg(pr)*dot(avg(vr),-1*normal) - avg(pi)*dot(avg(vi),-1*normal)
Csf2 = avg(pr)*dot(avg(vr),normal) - avg(pi)*dot(avg(vi),normal)

Cfs1 = avg(qr)*dot(avg(ur),-1*normal) - avg(qi)*dot(avg(ui),-1*normal)
Cfs2 = avg(qr)*dot(avg(ur),normal) - avg(qi)*dot(avg(ui),normal)

Bf = (dot(grad(qr),grad(pr)) - dot(grad(qi),grad(pi))) - (a4r*(qr*pr - qi*pi) - a4i*(qi*pr + qr*pi))


#Galerkin equation
a = [[Bs*(dx(1)+dx(3))                                ,                     a1*a6*Csf2*dS(2) ],
     [                    (1/a5)*Cfs2*dS(2)   ,  Bf*(dx(2)+dx(4))                            ]]
f = [-( 1*inner(n_outer,vr) - o*inner(n_outer,vi) )*ds(1) + (inner(a3r,vr) - inner(a3i,vi))*dx(1), o*(qr-qi)*dx(2)] # Pextr = Pext, Pexti = 0



##### Dirichlet BCs #######
pf = Expression((("1","0.0")),element = M1.ufl_element())
bc1 = DirichletBC(W.sub(1), pf, facettag, 3)#magnitude of the external load 
solid_fixed = Expression(("0","0"), element = M1.ufl_element())
bc1 = DirichletBC(W.sub(0).sub(1), solid_fixed, facettag, 1)
bcs = BlockDirichletBC([bc1])


## DIRECT SOLVER ######################
A = block_assemble(a)
F = block_assemble(f)
#bcs.apply(A)
#bcs.apply(F)

U = BlockFunction(W)
U.block_vector()[:] = 0.0 #initial guess
print("Number of degrees of freedom = ",len(U.block_vector()))
block_solve(A, U.block_vector(), F)
u = U.sub(0)
p = U.sub(1)
#

#### ITERATIVE SOLVER #
## SOLVE #
#A = block_assemble(a)
#F = block_assemble(f)
##bcs.apply(A)
##bcs.apply(F)
#
#U = BlockFunction(W)
#U.block_vector()[:] = 0.0 # initial guess
#print("Number of degrees of freedom = ",len(U.block_vector()))
#
## Solving the linear system
#solver.solve(A, U.block_vector(), F)
#U.block_vector().block_function().apply("to subfunctions")
#u = U.sub(0)
#p = U.sub(1)
##
#######################################

#### Storing the primal solution #########
ur0 = interpolate(u.sub(0),M1)
pr0 = interpolate(p.sub(0),M2)  
ui0 = interpolate(u.sub(1),M1)
pi0 = interpolate(p.sub(1),M2)  


#rmesh = refine(mesh)
rmesh = mesh
Qh = FunctionSpace(rmesh, "Lagrange", 1)
P1 = VectorElement("Lagrange", rmesh.ufl_cell(), 1)
Vh = FunctionSpace(rmesh, P1)
Pr = interpolate(pr0,Qh)
Ur = interpolate(ur0,Vh)
Pi = interpolate(pi0,Qh)
Ui = interpolate(ui0,Vh)

mur_init = project(mur,Qh) 

# OUTPUT #
file1 = XDMFFile("output/ur_init.xdmf")
file1.write(Ur,1)
file2 = XDMFFile("output/pr_init.xdmf")
file2.write(Pr,1)
file1 = XDMFFile("output/ui_init.xdmf")
file1.write(Ui,1)
file2 = XDMFFile("output/pi_init.xdmf")
file2.write(Pi,1)

file3 = XDMFFile("output/mu_init.xdmf")
file3.write(mur_init,1)

wt_interface = Expression(("(x[0]-0)*(x[0]-0)/(a*a)+(x[1]-0)*(x[1]-0)/(b*b) >= (1 - tol) && \
                            (x[0]-0)*(x[0]-0)/(a*a)+(x[1]-0)*(x[1]-0)/(b*b) <= (1 + tol)"),
                          a = 1, b = 0.75, tol = 5E-3, element = M2.ufl_element())
Wt_interface = project(wt_interface,Qh)
file3 = XDMFFile("output/wt_interface.xdmf")
file3.write(Wt_interface,1)

File("output/celltag.pvd") << celltag

l = Constant(1)
print("Length of actuator region = ", assemble(l*ds(1)))

one = Constant(1)
 
Rpsi_1 = assemble(beta1*psi*(one-psi)*dx(3))  
Rpsi_2 = assemble(beta2*(dot(grad(psi),grad(psi))+c)**(1/2)*dx(3))
Jpsi = Rpsi_1 + Rpsi_2 + assemble(alpha*(pr0*pr0 + pi0*pi0)*dx(4))

print("Jpsi =", Jpsi)  


#### Solving adjoint equation#############
# TRIAL/TEST FUNCTIONS #
WQ = BlockTrialFunction(W)
(w, q) = block_split(WQ)
(wr, wi) = (as_vector((w[0], w[1])), as_vector((w[2], w[3])))
(qr, qi) = as_vector((q[0], q[1]))
dUdP = BlockTestFunction(W)
(du, dp) = block_split(dUdP)
(dur, dui) = (as_vector((du[0], du[1])), as_vector((du[2], du[3])))
(dpr, dpi) = as_vector((dp[0], dp[1]))


# ASSEMBLE #
I = Identity(2)
sigma_wrr = 2*mur*0.5*(grad(wr)+grad(wr).T) + mur*a2r*div(wr)*I 
sigma_wri = 2*mur*0.5*(grad(wi)+grad(wi).T) + mur*a2r*div(wi)*I 
sigma_wir = 2*mui*0.5*(grad(wr)+grad(wr).T) + a2i*div(wr)*I 
sigma_wii = 2*mui*0.5*(grad(wi)+grad(wi).T) + a2i*div(wi)*I 

#Bilinear and linear terms
Bs = a1*(inner(grad(dur), sigma_wrr-sigma_wii) - inner(grad(dui), sigma_wir+sigma_wri)) - (dot(dur,wr) - dot(dui,wi))

Csf1 = avg(qr)*dot(avg(dur),-1*normal) - avg(qi)*dot(avg(dui),-1*normal)
Csf2 = avg(qr)*dot(avg(dur),normal) - avg(qi)*dot(avg(dui),normal)

Cfs1 = avg(dpr)*dot(avg(wr),-1*normal) - avg(dpi)*dot(avg(wi),-1*normal)
Cfs2 = avg(dpr)*dot(avg(wr),normal) - avg(dpi)*dot(avg(wi),normal)

Bf = (dot(grad(dpr),grad(qr)) - dot(grad(dpi),grad(qi))) - (a4r*(dpr*qr - dpi*qi) - a4i*(dpi*qr + dpr*qi))

two = Constant(2)
#Galerkin equation
a_adj = [[Bs*(dx(1)+dx(3))                                ,                     (1/a5)*Csf2*dS(2) ],
         [                    a1*a6*Cfs2*dS(2)    ,  Bf*(dx(2)+dx(4))                             ]]
f_adj = [o*(inner(dur,a3r) + inner(dui,a3i))*(dx(1)+dx(3)), -two*alpha*(pr0*dpr + pi0*dpi)*dx(4)]
    

##### Dirichlet BCs #######
pf = Expression((("0","0.0")),element = M1.ufl_element())
bc1 = DirichletBC(W.sub(1), pf, facettag, 3)#magnitude of the external load 
solid_fixed = Expression(("0","0"), element = M1.ufl_element())
bc1 = DirichletBC(W.sub(0).sub(1), solid_fixed, facettag, 1)
bcs_adj = BlockDirichletBC([bc1])


## Direct SOLVER ############################
A_adj = block_assemble(a_adj)
F_adj = block_assemble(f_adj)
#bcs_adj.apply(A_adj)
#bcs_adj.apply(F_adj)

U_adj = BlockFunction(W)
block_solve(A_adj, U_adj.block_vector(), F_adj)
w = U_adj.sub(0)
q = U_adj.sub(1)
#

#### ITERATIVE SOLVER #
## SOLVE #
#A_adj = block_assemble(a_adj)
#F_adj = block_assemble(f_adj)
##bcs_adj.apply(A_adj)
##bcs_adj.apply(F_adj)
#
#U_adj = BlockFunction(W)
#U_adj.block_vector()[:] = 0.0 # initial guess
#
## Solving the linear system
#solver.solve(A_adj, U_adj.block_vector(), F_adj)
#U_adj.block_vector().block_function().apply("to subfunctions")
#w = U_adj.sub(0)
#q = U_adj.sub(1)
###########################################

#### Storing the dual solution #########
wr = project(w.sub(0),M1)
qr = project(q.sub(0),M2)  
wi = project(w.sub(1),M1)
qi = project(q.sub(1),M2) 


#### Computing gradient #######
   
DJ = Function(M2)
one = Constant(1)
J = a1*(mua+ psi*(mub-mua))*(inner(grad(ur0),(grad(wr) + grad(wr).T) + a2r*div(wr)*I) - \
                             inner(grad(ui0),(grad(wi) + grad(wi).T) + a2r*div(wi)*I))*dx(3) \
    + beta1*psi*(one-psi)*dx(3)  
#       + beta*((mua+(psi**6)*(mub-mua))-mua)*((mua+(psi**6)*(mub-mua))-muA)*(mub-(mua+(psi**6)*(mub-mua)))*(mub-(mua+(psi**6)*(mub-mua)))*dx(3) \

dpsi = Constant(1)
dpsi = project(dpsi,M2)
J2 = beta2*dot(grad(psi),grad(dpsi))*(dot(grad(psi),grad(psi))+c)**(-1/2)*dx(3) 
DJ2 = derivative(J2,dpsi)
DJ2 = assemble(DJ2)

DJdpsi = derivative(J,psi)
DJdpsi = assemble(DJdpsi)
DJ.vector()[:] = DJdpsi[:]
    
Sensitivity_vector = np.add(DJdpsi,DJ2)
DJ.vector()[:] = Sensitivity_vector[:] 


DJplot = project(DJ,Qh)
file4 = XDMFFile("output/DJdpsi.xdmf")
file4.write(DJplot,1)   




dpsi = Function(M2)
dpsi.vector()[:] = DJ.vector()[:]
dpsi_mag = sqrt(dpsi.vector().inner(dpsi.vector()))
dpsi.vector()[:] = (1/dpsi_mag)*dpsi.vector()[:] # normalizing dmu
#

ps = Function(M2)
ps.vector()[:] = psi.vector()[:]*wt.vector()[:]
h = -(0.00125/8)*sqrt(ps.vector().inner(ps.vector()))
dpsi.vector()[:] = h*dpsi.vector()[:]
psi.vector()[:] = psi.vector()[:] + dpsi.vector()[:]


######## Finding optimized solution ###################
    
# TRIAL/TEST FUNCTIONS #
UP = BlockTrialFunction(W)
(u, p) = block_split(UP)
(ur, ui) = (as_vector((u[0], u[1])), as_vector((u[2], u[3])))
(pr, pi) = as_vector((p[0], p[1]))
VQ = BlockTestFunction(W)
(v, q) = block_split(VQ)
(vr, vi) = (as_vector((v[0], v[1])), as_vector((v[2], v[3])))
(qr, qi) = as_vector((q[0], q[1]))


# ASSEMBLING GALERKIN EQUATION ####################################
I = Identity(2)
mur = (mua+psi*(mub-mua))
#two = Constant(2)
#one = Constant(1)
#mur = (mua+ eps*((two*psi)**n)*((one+(two*psi)**n)**(-1))*(mub-mua))

sigmarr = 2*mur*0.5*(grad(ur)+grad(ur).T) + mur*a2r*div(ur)*I 
sigmari = 2*mur*0.5*(grad(ui)+grad(ui).T) + mur*a2r*div(ui)*I 
sigmair = 2*mui*0.5*(grad(ur)+grad(ur).T) + a2i*div(ur)*I 
sigmaii = 2*mui*0.5*(grad(ui)+grad(ui).T) + a2i*div(ui)*I 

#Bilinear and linear terms
Bs = a1*(inner(grad(vr), sigmarr-sigmaii) - inner(grad(vi), sigmair+sigmari)) - (dot(vr,ur) - dot(vi,ui))

Csf1 = avg(pr)*dot(avg(vr),-1*normal) - avg(pi)*dot(avg(vi),-1*normal)
Csf2 = avg(pr)*dot(avg(vr),normal) - avg(pi)*dot(avg(vi),normal)

Cfs1 = avg(qr)*dot(avg(ur),-1*normal) - avg(qi)*dot(avg(ui),-1*normal)
Cfs2 = avg(qr)*dot(avg(ur),normal) - avg(qi)*dot(avg(ui),normal)

Bf = (dot(grad(qr),grad(pr)) - dot(grad(qi),grad(pi))) - (a4r*(qr*pr - qi*pi) - a4i*(qi*pr + qr*pi))


#Galerkin equation
a = [[Bs*(dx(1)+dx(3))                                ,                     a1*a6*Csf2*dS(2) ],
     [                    (1/a5)*Cfs2*dS(2)   ,  Bf*(dx(2)+dx(4))                            ]]
f = [-( 1*inner(n_outer,vr) - o*inner(n_outer,vi) )*ds(1) + (inner(a3r,vr) - inner(a3i,vi))*dx(1), o*(qr-qi)*dx(2)] # Pextr = Pext, Pexti = 0



## DIRECT SOLVER ######################
A = block_assemble(a)
F = block_assemble(f)
#bcs.apply(A)
#bcs.apply(F)

U = BlockFunction(W)
block_solve(A, U.block_vector(), F)
u = U.sub(0)
p = U.sub(1)
#

#### ITERATIVE SOLVER #
## SOLVE #
#A = block_assemble(a)
#F = block_assemble(f)
##bcs.apply(A)
##bcs.apply(F)
#
#U = BlockFunction(W)
#U.block_vector()[:] = 0.0 # initial guess
#
## Solving the linear system
#solver.solve(A, U.block_vector(), F)
#U.block_vector().block_function().apply("to subfunctions")
#u = U.sub(0)
#p = U.sub(1)
##
#######################################

#### Storing the optimized solution #########
ur0 = interpolate(u.sub(0),M1)
pr0 = interpolate(p.sub(0),M2)  
ui0 = interpolate(u.sub(1),M1)
pi0 = interpolate(p.sub(1),M2)  


Rpsi_1h = assemble(beta1*psi*(one-psi)*dx(3))  
Rpsi_2h = assemble(beta2*(dot(grad(psi),grad(psi))+c)**(1/2)*dx(3))
Jpsi_h = Rpsi_1h + Rpsi_2h + assemble(alpha*(pr0*pr0 + pi0*pi0)*dx(4))
  
print("Jpsi_h =", Jpsi_h)


#### Finding residue ##########
Residue = abs(Jpsi_h - Jpsi - DJ.vector().inner(dpsi.vector()))
print("Residue =",Residue)
print("del_J =",DJ.vector().inner(dpsi.vector()))