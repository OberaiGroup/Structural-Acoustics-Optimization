"""
@author: Harisankar
"""

from fenics import *
from multiphenics import *
import numpy as np
import scipy.optimize as sp
import time
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
beta1 = Constant(1e-0)# weight for the regularization term
beta2 = Constant(1e-7)
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
psi = interpolate(psi,M2)
#psi_temp = np.genfromtxt("psi_array.txt")
#psi.vector()[:] = psi_temp[:]

Pext = Constant(100)# External time harmonic pressure load
L = Constant(1.0)# Scale factor for the length of the domain
Uo = L*Pext/mu0 # Nominal deformation


wt = Expression(("(x[0]-0)*(x[0]-0)/(a1*a1)+(x[1]-0)*(x[1]-0)/(b1*b1) >= (1 - tol)"), 
                  a1 = 1,b1 = 0.75, a2 = 1.15,b2 = 9,tol = 5e-3,
                  element = M2.ufl_element())
#wt = interpolate(wt,M2)
mur = interpolate(mur,M2)
#mur.vector()[:] = mur.vector()[:]*wt.vector()[:]


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

##saving stiffness matrix for further processing
#A_mat = as_backend_type(A).mat()
#from scipy.sparse import csr_matrix
#A_sparray = csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
#from scipy.sparse.linalg import eigs
#vals, vecs = eigs(A_sparray, k=3, M = None, sigma = 0)
#print(vals)
#

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

###############################################3

# Use previous solution as starting guess for itertive solver
Utemp = BlockFunction(W)
Utemp.block_vector()[:] = 0.0 
Uprev = BlockFunction(W)
Uprev.block_vector()[:] = 0.0 



#####  Storing video of optimization history  ######

domain1 = Ellipse(Point(0, 0), 1.25 , 1)
domain2 = Ellipse(Point(0, 0), 1.00 , 0.75)
domain = domain1 - domain2
domain.set_subdomain(1, Ellipse(Point(0, 0), 1.15 , 0.9))

mesh_solid = generate_mesh(domain, N)
mesh_solid = refine(mesh_solid)

Sh = FunctionSpace(mesh_solid, "Lagrange", 1)

fid1 = File("output/ur_video.pvd")
fid2 = File("output/pr_video.pvd")
fid3 = File("output/mur_video.pvd")
u_mag = Constant(0)
u_mag = interpolate(u_mag,M2)

ux = interpolate(ur0.sub(0), M2)
uy = interpolate(ur0.sub(1), M2)
ux = ux.vector()
uy = uy.vector()
u_abs = np.sqrt(np.add(np.multiply(ux,ux),np.multiply(uy,uy)))
u_mag.vector()[:] = u_abs[:]
u_frame = interpolate(u_mag,Sh)
mur_frame = project(mur,Sh)
p_frame = interpolate(pr0,Qh)
u_frame.rename("ur","ur")
mur_frame.rename("mur","mur")
p_frame.rename("pr","pr")
frame = 0
fid1 << u_frame, frame
fid2 << p_frame, frame
fid3 << mur_frame, frame
frame =  frame + 1



   
###  OPTIMIZATION ############################################
## Initializing regularization term
Rpsi_1 = 0
Rpsi_2 = 0
Fpsi = 0
#### Objective Functional to be minimized ################################
def J_eval(psi_control):
    
    global mesh,V,M1,Q,M2,W
    
    # intializing fenics vector mu and converting mu_control(np.array) to mu 
    psi = Constant(0.0)
    psi = interpolate(psi,M2)
    psi.vector()[:] = psi_control[:] 
  
    # TRIAL/TEST FUNCTIONS #
    UP = BlockTrialFunction(W)
    (u, p) = block_split(UP)
    (ur, ui) = (as_vector((u[0], u[1])), as_vector((u[2], u[3])))
    (pr, pi) = as_vector((p[0], p[1]))
    VQ = BlockTestFunction(W)
    (v, q) = block_split(VQ)
    (vr, vi) = (as_vector((v[0], v[1])), as_vector((v[2], v[3])))
    (qr, qi) = as_vector((q[0], q[1]))
    
    # Constants #
    global a1, a2r, a2i, a3r, a3i, a4r, a4i, a5, a6, o, beta1, beta2, alpha, mua, mub, n, eps, c
    
    # Normals
    global normal, n_outer
    
    # Integral elements and BCs
    global dx, ds, dS, bcs
    
        
    # ASSEMBLING GALERKIN EQUATION ####################################
    I = Identity(2)
    mur = (mua+psi*(mub-mua))
#    two = Constant(2)
#    one = Constant(1)
#    mur = (mua+ eps*((two*psi)**n)*((one+(two*psi)**n)**(-1))*(mub-mua))
    
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
    # global Uprev
    #U.block_vector()[:] = Uprev.block_vector()[:] # initial guess

    ## Solving the linear system
    #solver.solve(A, U.block_vector(), F)
    #U.block_vector().block_function().apply("to subfunctions")
    #u = U.sub(0)
    #p = U.sub(1)
    ##
    #######################################
    
    global Utemp     
    Utemp = U
    pr0 = p.sub(0)
    pi0 = p.sub(1)   
    
    #### Computing objective functional #######
        
    global Rpsi_1, Rpsi_2
    one = Constant(1)
 
    Rpsi_1 = assemble(beta1*psi*(one-psi)*dx(3))  
    Rpsi_2 = assemble(beta2*(dot(grad(psi),grad(psi))+c)**(1/2)*dx(3))
    Jpsi = Rpsi_1 + Rpsi_2 + assemble(alpha*(pr0*pr0 + pi0*pi0)*dx(4))
    
    return Jpsi


    
######### GRADIENT DEFINITION ####################################################    
def Gradient_compute(psi_control):
    
    global mesh,V,M1,Q,M2,W
    
    # intializing fenics vector mu and converting mu_control(np.array) to mu 
    psi = Constant(0.0)
    psi = interpolate(psi,M2)
    psi.vector()[:] = psi_control[:] 
  
    # TRIAL/TEST FUNCTIONS #
    UP = BlockTrialFunction(W)
    (u, p) = block_split(UP)
    (ur, ui) = (as_vector((u[0], u[1])), as_vector((u[2], u[3])))
    (pr, pi) = as_vector((p[0], p[1]))
    VQ = BlockTestFunction(W)
    (v, q) = block_split(VQ)
    (vr, vi) = (as_vector((v[0], v[1])), as_vector((v[2], v[3])))
    (qr, qi) = as_vector((q[0], q[1]))
    
    # Constants #
    global a1, a2r, a2i, a3r, a3i, a4r, a4i, a5, a6, o, beta1, beta2, alpha, mua, mub, n, eps, c
    
    # Normals
    global normal, n_outer
    
    # Integral elements and BCs
    global dx, ds, dS, bcs
    
        
    # ASSEMBLING GALERKIN EQUATION ####################################
    I = Identity(2)
    mur = (mua+psi*(mub-mua))
#    two = Constant(2)
#    one = Constant(1)
#    mur = (mua+ eps*((two*psi)**n)*((one+(two*psi)**n)**(-1))*(mub-mua))
    
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
    # global Uprev
    #U.block_vector()[:] = Uprev.block_vector()[:] # initial guess
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
    
    if len(DJdpsi) == len(DJ2):

#        Sensitivity_vector = np.add(DJdpsi,DJ2) 
        
        DJ_vector = np.add(DJdpsi,DJ2)        
        H = psi*dx(3)
        DHdpsi = derivative(H,psi)
        DHdpsi = assemble(DHdpsi)
        DH = DHdpsi[:] + (DHdpsi[:]==0)*1
        Sensitivity_vector = 1e-5*DJ_vector[:]/DH[:]
        
    else:
        print("Error: Length of derivative terms don't match")
        
    return Sensitivity_vector

###########################################################################            


### callback function send to scipy.optimize.minimize for printing values at each interation 
N_J_eval = 1

psi_old = psi.vector()[:]
J_old = 0.0

N_nodes = len(psi.vector())
print("Number of control parameters(psi) = {0:4d}".format(N_nodes))
print("{0:^4s}   {1:^8s}   {2:^8s}   {3:^8s}   {4:^8s}    {5:^8s}    {6:^8s}    {7:^8s}".format( \
      "Iter", "Jpsi", "Fpsi", "TV", "ftol", "del_psi_rms", "rtol", "gtol_rms"))

def callbackF(psi_control):
    
    global mesh,V,M1,Q,M2,W
    
    Jpsi = J_eval(psi_control)
    DJpsi = Gradient_compute(psi_control)
    
    global N_J_eval, J_old, psi_old, N_nodes
    
    ftol = (J_old>0)*(J_old-Jpsi)/max(J_old,Jpsi,1)
    psi_old_rms =  np.sqrt(np.matmul(psi_old,np.transpose(psi_old))/N_nodes)  
    psi_diff = np.add(np.array(psi_control),-1*np.array(psi_old))
    psi_diff_rms = np.sqrt(np.matmul(psi_diff,np.transpose(psi_diff))/N_nodes)
    rtol = psi_diff_rms/psi_old_rms
    gtol = np.sqrt(np.matmul(DJpsi,np.transpose(DJpsi))/N_nodes) #max(np.abs(DJmu)) 
    global Rpsi_2, Rpsi_1, Fpsi
    Fpsi = Jpsi - Rpsi_1 - Rpsi_2
    print("{0:4d}   {1:.3e}   {2:.3e}   {3:.3e}   {4:.3e}   {5:.3e}   {6:.3e}   {7:.3e}".format(N_J_eval, Jpsi, Fpsi, Rpsi_2, ftol, psi_diff_rms, rtol, gtol)) 
    
    J_old = Jpsi
    psi_old[:] = psi_control[:]
    N_J_eval += 1
    
    global Uprev, Utemp
    Uprev.sub(0).vector()[:] = Utemp.sub(0).vector()[:]
    Uprev.sub(1).vector()[:] = Utemp.sub(1).vector()[:]
    
    ### Storing video of optimization history ##########   
    global fid1, fid2, fid3, frame, u_mag, Sh,Qh, mua, mub, n, eps
    ux = interpolate(Uprev.sub(0).sub(0).sub(0), M2)
    uy = interpolate(Uprev.sub(0).sub(0).sub(1), M2)
    ux = ux.vector()
    uy = uy.vector()
    u_abs = np.sqrt(np.add(np.multiply(ux,ux),np.multiply(uy,uy)))
    u_mag.vector()[:] = u_abs[:]
    psi = Constant(0.0)
    psi = interpolate(psi,M2)
    psi.vector().set_local(psi_old)
    
    mur = (mua+psi*(mub-mua))
#    two = Constant(2)
#    one = Constant(1)
#    mur = (mua+ eps*((two*psi)**n)*((one+(two*psi)**n)**(-1))*(mub-mua))
    
    u_frame = interpolate(u_mag,Sh)
    mur_frame = project(mur,Sh)
    p_frame = interpolate(Uprev.sub(1).sub(0),Qh)
    u_frame.rename("ur","ur")
    mur_frame.rename("mur","mur")
    p_frame.rename("pr","pr")
    fid1 << u_frame, frame
    fid2 << p_frame, frame
    fid3 << mur_frame, frame
    frame =  frame + 1
    
###############################################################################
    
    
#### Running optimization algorithm #########
psi0_array = psi.vector()[:]
bnds = [(0, 1.00) for i in psi0_array]

# Computing volume of control region 
vol = assemble(l*dx(3))

t_start = time.time()
print("====================================================================") 
psi_array = sp.minimize(J_eval, psi0_array,                     
                    jac = Gradient_compute, 
                    bounds =  bnds, 
                    tol = None, 
                    callback = callbackF,  
                    options = {'ftol':1e-20,'gtol':1e-8, 'maxiter':80,'disp':None})
# tol is gtol in case of method = BFGS(default for unconstrained problems)  
# default values for BFGS -> 'gtol': 1e-05
# default values for L-BFGS-B -> 'ftol': 2.220446049250313e-09, 'gtol': 1e-05
# tol is both ftol and gtol in case of L-BFGS-B 
# Note: ftol value should be small enough such that the algorithm doesn't terminate at iterate 1
print("====================================================================") 
print("End") 
t_end = time.time()
CPU_time = t_end - t_start 
#Display CPU time
print("Computational time (optimization) =", CPU_time)


print("message: ",psi_array.message)
print("nfev: ",psi_array.nfev)
print("nit: ",psi_array.nit)
#print("njev: ",mu_array.njev)
#print("jac: ",mu_array.jac)
#print("status: ",mu_array.status)
#print("success: ",mu_array.success)

print("Regularization functional value = ",Rpsi_1)
print("Total variation value = ",Rpsi_2)

psi.vector().set_local(psi_array.x)
##Storing psi   
#np.savetxt("psi_array2.txt",psi_array.x)
##

file1 = open("R_sweep.txt","a")
file1.write("\n") 
if Fpsi <= 1e-15:
    Fpsi = J_eval(psi0_array) -Rpsi_1 -Rpsi_2 
  
L = str(Fpsi)
file1.writelines(L) 
file1.close()
################################################################################



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


Qh = FunctionSpace(rmesh, "Lagrange", 1)
P1 = VectorElement("Lagrange", rmesh.ufl_cell(), 1)
Vh = FunctionSpace(rmesh, P1)
Pr = interpolate(pr0,Qh)
Ur = interpolate(ur0,Vh)
Pi = interpolate(pi0,Qh)
Ui = interpolate(ui0,Vh)

mur_opt = project(mur,Qh) 

print("Volume fraction of material B = ", assemble(psi*dx(3))/vol)

# OUTPUT #
file1 = XDMFFile("output/ur_opt.xdmf")
file1.write(Ur,1)
file2 = XDMFFile("output/pr_opt.xdmf")
file2.write(Pr,1)
file1 = XDMFFile("output/ui_opt.xdmf")
file1.write(Ui,1)
file2 = XDMFFile("output/pi_opt.xdmf")
file2.write(Pi,1)

file3 = XDMFFile("output/mu_opt.xdmf")
file3.write(mur_opt,1)
psi_opt = interpolate(psi, Qh)
file3 = XDMFFile("output/psi_opt.xdmf")
file3.write(psi_opt,1)

######## THE END ############

###'''

