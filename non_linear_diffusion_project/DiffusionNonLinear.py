from dolfin import *
import numpy 


def picard(N, divisions,alpha, rho, u_I,p = 1, f=False, plotting = False):
	
	alpha = alpha
	rho = rho
	
	mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
	d = len(divisions)
	mesh = mesh_classes[d-1](*divisions)
	
	V = FunctionSpace(mesh,"CG",p)
	u = TrialFunction(V)
	u_ = Function(V)
	v = TestFunction(V)

	u_I = u_I
	u_1 = project(u_I,V)
	u_k = u_1

	T = 1           		# Playtime
	dt = (1./N)*(1./N)      # Timestep
	t = 0
	
	if f:
		f = Expression('-rho*pow(x[0],3)/3. + rho*pow(x[0],2)/2. + 8*pow(t,3)*pow(x[0],7)/9.- 28*pow(t,3)*pow(x[0],6)/9. + 7*pow(t,3)*pow(x[0],5)/2. - 5*pow(t,3)*pow(x[0],4)/4. + 2*t*x[0] - t',rho=rho,t=t)
		f.t=0
	else:
		f = Constant('0')

	F = u*v*dx + inner(dt/rho*alpha(u_k)*nabla_grad(u),nabla_grad(v))*dx - u_1*v*dx - dt/rho*f*v*dx
	a = lhs(F)
	L = rhs(F)
	E = 0
	while t <= T:
		t += dt
		f.t = t
		solve(a==L, u_)
		if plotting:
			plot(u_,rescale=False,interactive=True) # Press q to proceed to next timestep
		u_1.assign(u_)			  # Updating solution
		u_k.assign(u_)            # Updating solution for the picard iteration which is u_1
	return u_, t, V, dt

def test_d():
	"""The first verification of the FEniCS implementation. Reproduce a constant
	   solution. u(x,t) = constant"""
	def alpha(u):
		return 1
	rho = 1
	N 	= 8
	divisions = [8,8]
	u_I = Expression("1")
	u_, t, V,dt = picard(N, divisions,alpha, rho,u_I)
	
	u_exact = Expression('1')
	u_e = project(u_exact,V)
	diff = max(u_e.vector().array() - u_.vector().array())
	tol = 1.0E-5
	assert diff<tol, 'solution is not constant'

def test_e():
	""" verifying that the solution reproduces an analytical solution. 
    Shows that E/h (E is the discrete L2 norm) remains approximately constant as the mesh in space and
	time is simultaneously refined (i.e., h is reduced)
    """
	rho = 1.0
	u_I = Expression("cos(pi*x[0])")
	def alpha1(u):
		return 1 
	k = []
	h = []
	for i in 10,15,20,30,50:
		u_, t, V,dt = picard(i,[i,i], alpha1, rho,u_I)
		u_exact = Expression('exp(-(pi*pi*t))*cos(pi*x[0])',t=t)
		h.append(dt)
		u_e = project(u_exact,V)
		e = (u_e.vector().array() - u_.vector().array())
		E = numpy.sqrt(numpy.sum(e**2)/u_.vector().array().size)
		k.append(E/dt)
	for i in range(len(k)):
		print "h = %3.5f, E/h = %3.7e" %( h[i], k[i] )
	
def test_f():
	""" Using the method of manufactured solutions to get an indication
    wether the implementation is correct or not.
    Shows that E/h (E is the discrete L2 norm) remains approximately constant as the mesh in space and
	time is simultaneously refined (i.e., h is reduced)"""
	rho = 1.0
	u_I = Constant('0')
	
	def alpha(u):
		return (1+u*u)

	k = []
	h = []
	for i in 10,20,30,50,100:
		u_, t, V,dt = picard(i,[i], alpha, rho,u_I,f= True)
		u_exact = Expression('t*pow(x[0],2)*(0.5-x[0]/3.)',t=t)
		h.append(dt)
		u_e = project(u_exact,V)
		e = (u_e.vector().array() - u_.vector().array())
		E = numpy.sqrt(numpy.sum(e**2)/u_.vector().array().size)
		k.append(E/dt)
	for i in range(len(k)):
		print "h = %3.5f, E/h = %3.7e" %( h[i], k[i] )
def test_g():
	rho = 1
	u_I = Expression("cos(pi*(x[0] * x[1]))")
	def alpha(u):
		return 1
		
	picard(10, [10,10], alpha, rho,u_I, f= False, plotting = True)
test_g()
if '__name__' == '__main__':
	test_d()	
	test_e()
	test_f()
	
 

