#!/usr/bin/env python
"""
2D wave equation solved by finite differences::

  dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,
                        user_action=None, version='scalar',
                        stability_safety_factor=1)

Solve the 2D wave equation u_tt + bu_t = u_xx + u_yy + f(x,y,t) on [0,L_x]X[0,L_y]
with Neumann or dirichtlet boundary conditions
 and initial conditions u(x,y,t=0) = du(x,y,t=0)/dt = 0.

Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.

I, V, f and c are functions: I(x,y), V(x,y), f(x,y,t), c(x,y) = sqrt(q(x,y)) 
V, and f can be specified as None or 0, resulting in V=0 and f=0 
in case of c(x) = constant means constant coefficient
__ c can not be specified by None**

user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.
"""

import numpy as np
import time
import glob, os
import sympy as sym

def solver(I,V,f,c,Lx,Ly,Nx,Ny,dt,T,b,
		   user_action=None, version='scalar'):
		
	x = np.linspace(0,Lx, Nx+1)       #mesh points in x dir
	y = np.linspace(0,Ly, Ny+1)		  #mesh points in y dir
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	
	xv = x[:,np.newaxis]     #for vectorized function evaluations
	yv = y[np.newaxis,:]
	
	if isinstance(c, (float,int)):
		c_max = c
	elif callable(c):
		c_max = np.absolute(max(c(x,y)))
	
	stability_limit = (1/float(c_max))*(1/np.sqrt(1/dx**2 + 1/dy**2))
	if dt <= 0:                # max time step?
		safety_factor = -dt    # use negative dt as safety factor
		dt = safety_factor*stability_limit
	elif dt > stability_limit:
		print 'error: dt=%g exceeds the stability limit %g' % \
				(dt, stability_limit)

	Nt = int(round(T/float(dt)))
	t  = np.linspace(0,Nt*dt, Nt+1)
		
	if I is None or I == 0:
		I = (lambda x,y: 0) if version == 'scalar' else \
			lambda x,y: np.zeros((x.shape[0], y.shape[1]))
	
	if f is None or f == 0:
		f = (lambda x,y,t: 0) if version == 'scalar' else \
			lambda x,y,t: np.zeros((x.shape[0], y.shape[1]))
			
	if V is None or V == 0:
		V = (lambda x,y:0) if version == 'scalar' else \
			lambda x,y: np.zeros((x.shape[0], y.shape[1]))
		

	u	= np.zeros((Nx+3,Ny+3))   # solution array
	u_1 = np.zeros((Nx+3,Ny+3))   # solution at t-dt
	u_2 = np.zeros((Nx+3,Ny+3))   # solution at t-2*dt
	f_a = np.zeros((Nx+1,Ny+1))   # for compiled loops. We do not need ghost points
	V_a = np.zeros((Nx+1,Ny+1))
	
	Ix = range(1,u.shape[0]-1)
	Iy = range(1,u.shape[1]-1)
	It = range(0,t.shape[0])
	
	# Treat c(x) as array
	if isinstance(c, (float,int)):
		c = np.zeros((Nx+3,Ny+3)) + c   # All numbers in c array is now c
	elif callable(c):
		# Call c(x,y) and fill array
		_c = np.zeros((Nx+3,Ny+3))
		for i in Ix:
			for j in Iy:
				_c[i,j] = c(x[i-Ix[0]], y[j-Iy[0]])
		c = _c
		
	q = c**2

	dx2 = (dt/dx)**2; dy2 = (dt/dy)**2
	dt2 = dt**2
	
	import time; t0 = time.clock()
	
	# load initial condition into u_1
	
	if version == 'scalar':
		for i in Ix:
			for j in Iy:
				u_1[i,j] = I(x[i-Ix[0]], y[j-Iy[0]])
	else: # use vectorized version
		u_1[1:-1,1:-1] = I(xv, yv)
		

	if version == 'scalar':
		Ix_ = range(1,u.shape[0])
		Iy_ = range(1,u.shape[1])			
		# Boundary condition U_x(0,y,t) = U_y(x,0,t) = 0	 		
		j = Iy[0]
		for i in Ix_:
			u_1[i-1,j-1] = u_1[i-1,j+1]
			q[i-1,j-1]	 = q[i-1,j+1]    # filling up q using q_x=q_y=0 
		i = Ix[0]
		for j in Iy_:
			u_1[i-1,j-1] = u_1[i+1,j-1]
			q[i-1,j-1]	 = q[i+1,j-1]	 # filling up  q using q_x=q_y=0 
		# Boundary condition U_x(Lx,y,t) = U_y(x,Ly,t) = 0
		j = Iy[-1]
		for i in Ix_:
			u_1[i-1,j+1] = u_1[i-1,j-1]
			q[i-1,j+1]   = q[i-1,j-1]    # filling up q using q_x=q_y=0 
		i = Ix[-1]
		for j in Iy_:
			u_1[i+1,j-1] = u_1[i-1,j-1] 
			q[i+1,j-1]   = q[i-1,j-1]    # filling up q using q_x=q_y=0
		
	else:
		# filling up ghost cells vectorized version
		i = Ix[0]
		u_1[i-1,:] = u_1[i+1,:]
		q[i-1,:]   = q[i+1,:]
		j = Iy[0]
		u_1[:,j-1] = u_1[:,j+1]
		q[:,j-1]   = q[:,j+1]
		i = Ix[-1]
		u_1[i+1,:] = u_1[i-1,:]
		q[i+1,:]   = q[i-1,:]
		j = Iy[-1]
		u_1[:,j+1] = u_1[:,j-1]
		q[:,j+1]   = q[:,j-1]
	

	if user_action is not None:
		user_action(u_1[1:-1,1:-1], x,xv, y,yv, t, 0)
		
	
	# spesial formula for step 1
	n = 0
	if version == 'vectorized':
		f_a[:,:] = f(xv,yv,t[n])
		V_a[:,:] = V(xv,yv)
		u = advance_vectorized(
            u, u_1, u_2, f_a ,q,b,
            dx2, dy2, dt2, V=V_a, step1=True)
	else:
		u = advance_scalar(
			u, u_1, u_2, f, q, b, x, y, t, n,
			dx2, dy2, dt2, V, step1=True)
    						
    
	if user_action is not None:
		user_action(u[1:-1,1:-1],x,xv,y,yv,t,1)  # For plotting, assert_error etc
		
	# Update data structures for next step
	u_2, u_1, u = u_1, u, u_2
	
	for n in It[2:-1]:
		if version == 'vectorized':
			f_a[:,:] = f(xv,yv,t[n])
			u = advance_vectorized(
			u, u_1, u_2, f_a, q, b,
			dx2, dy2, dt2)
		else:
			u = advance_scalar(
			u, u_1, u_2, f, q, b, x, y, t, n,
			dx2, dy2, dt2)


		if user_action is not None:
			if user_action(u[1:-1,1:-1],x,xv,y,yv,t,n):
				break
		u_2, u_1, u = u_1, u, u_2
	return 
	
def advance_scalar(u, u_1, u_2, f, q, b, x, y, t, n, dx2, dy2, dt2, 
				   V=None, step1=False):
	
	dt = np.sqrt(dt2)
	H_V = 2/(2+b*dt)  # help variable to shorten term
	H_V2 = ((b*dt/2) -1)
	Ix = range(1,u.shape[0]-1); Iy = range(1,u.shape[1]-1)
	for i in Ix:
		for j in Iy:
			if step1:
				u[i,j] = 0.5*(2*(u_1[i,j] - H_V2*dt*V(x[i-Ix[0]], y[j-Iy[0]])) + \
						 dx2*(0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
						 0.5*(q[i,j]+q[i-1,j])*(u_1[i,j]-u_1[i-1,j])) + \
						 dy2*(0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
						 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
						 dt2*f(x[i-Ix[0]], y[j-Iy[0]], t[n]))
			
			else:
				u[i,j] = H_V*(2*u_1[i,j] + H_V2*u_2[i,j] + \
						 dx2*(0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
						 0.5*(q[i,j]+q[i-1,j])*(u_1[i,j]-u_1[i-1,j])) + \
						 dy2*(0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
						 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
						 dt2*f(x[i-Ix[0]], y[j-Iy[0]], t[n]))
	
	Ix_ = range(1,u.shape[0])
	Iy_ = range(1,u.shape[1])			
	# Boundary condition U_x(0,y,t) = U_y(x,0,t) = 0	 		
	j = Iy[0]
	for i in Ix_:
		u[i-1,j-1] = u[i-1,j+1] 
	i = Ix[0]
	for j in Iy_:
		u[i-1,j-1] = u[i+1,j-1]
	# Boundary condition U_x(Lx,y,t) = U_y(x,Ly,t) = 0
	j = Iy[-1]
	for i in Ix_:
		u[i-1,j+1] = u[i-1,j-1]
	i = Ix[-1]
	for j in Iy_:
		u[i+1,j-1] = u[i-1,j-1] 
	return u

def advance_vectorized(u,u_1,u_2,f_a,q,b,dx2,dy2,dt2,
					   V=None,step1=False):
	dt = np.sqrt(dt2)
	H_V = 2/(2+b*dt)  # help variable to shorten term
	H_V2 = ((b*dt/2) -1)
	DxqDxu = dx2*(0.5*(q[1:-1,1:-1] + q[2:,1:-1])*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - \
	             0.5*(q[1:-1,1:-1] + q[:-2,1:-1])*(u_1[1:-1,1:-1] - u_1[:-2,1:-1]))
	             
	DyqDyu = dy2*(0.5*(q[1:-1,1:-1] + q[1:-1,2:])*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - \
			     0.5*(q[1:-1,1:-1] + q[1:-1,:-2])*(u_1[1:-1,1:-1] - u_1[1:-1,:-2]))
			     
	if step1:
		u[1:-1,1:-1] = 0.5*(2*(u_1[1:-1,1:-1] - H_V2*dt*V[:,:]) + DxqDxu + DyqDyu + dt2*f_a[:,:])
	else:
		u[1:-1,1:-1] = H_V*(2*u_1[1:-1,1:-1] + H_V2*u_2[1:-1,1:-1] + DxqDxu + DyqDyu + dt2*f_a[:,:])
	

	# filling up ghost cells using boundary conditions
	i = 1
	u[i-1,:] = u[i+1,:]
	j = 1
	u[:,j-1] = u[:,j+1]
	i = u.shape[0]-2
	u[i+1,:] = u[i-1,:]
	j = u.shape[1]-2
	u[:,j+1] = u[:,j-1]
	return u


def test_constant_solution(Lx,Ly,Nx,Ny,dt,T,b,c, u_EXACT):
	x,y,t,b,q = sym.symbols(' x y t b q')
	u_e = u_EXACT
	u_t = sym.diff(u_e,t)
	b_u_t = b*u_t
	u_tt = sym.diff(u_t,t)
	u_x = sym.diff(u_e,x)
	q_u_x = q*u_x
	q_u_xx = sym.diff(q_u_x,x)
	u_y = sym.diff(u_e,y)
	q_u_y = q*u_y
	q_u_yy = sym.diff(q_u_y)
	
	f = u_tt + b_u_t - q_u_xx - q_u_yy
	u_t = sym.lambdify((x,y,t),u_t)
	f_= sym.lambdify((x,y,t),f)
	u = sym.lambdify((x,y,t),u_e)
	I = lambda x,y: u(x,y,0)
	V = lambda x,y: u_t(x,y,0)
	b = 1
	def assert_error(u_num,x,xv,y,yv,t,n):
		u_exact = u(xv,yv,t[n])
		diff= abs(u_exact - u_num).max()
		tol = 1E-12
		msg = 'diff=%g, step %d' % (diff, n)
		assert diff < tol, msg
		
	solver(I,V,f_,c,Lx,Ly,Nx,Ny,dt,T,b,
		  user_action=assert_error, version='vectorized')
	solver(I,V,f_,c,Lx,Ly,Nx,Ny,dt,T,b,
		  user_action=assert_error, version='scalar')

def test_gaussian():
	for filename in glob.glob('tmp_*.png'):
		os.remove(filename)
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import matplotlib.pyplot as plt

	
	
	Lx = 10
	Ly = 10
	I = lambda x,y: np.exp(-((x-0/2.0)/2)**2) 
	V = None
	f = 0
	b = 0.0; T = 20; Nx = 40; Ny = 40; dt = -1
	c = lambda x,y: np.sqrt(9.81*(1 - 0.5*np.exp(-(x-Lx/2)**2 - (y-Ly/2)**2)))   # q = g*H(x,y)= g*(H0 - B(x,y))
	
	def plot(u_num, x, xv, y, yv, t, n):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = x
		Y = y
		X, Y = np.meshgrid(X, Y)
		Z = u_num
		Z2 = -1 + 0.5*np.exp(-(X-Lx/2)**2 - (Y-Ly/2)**2)
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				       linewidth=0, antialiased=False)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='x', offset=-1, cmap=cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='y', offset=11, cmap=cm.coolwarm)
		
		
		ax.set_zlim(-1.01, 1.01)
		ax.set_xlabel('X')
		ax.set_xlim(-1, 10)
		ax.set_ylabel('Y')
		ax.set_ylim(0, 11)
		ax.set_zlabel('Z')
		ax.set_zlim(-100, 100)
		
		surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.hot,
				       linewidth=0, antialiased=False)
		ax.set_zlim(-1.01, 1.01)

		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.savefig('ANIMATION/tmp_%04d.png' % n)
		plt.close()
	solver(I,V,f,c,Lx,Ly,Nx,Ny,dt,T,b,
		  user_action=plot, version='vectorized')

def test_plug():
	""" pulse function for simulating the propagation of a plug wave, 
		where I(x) is constant in some region of the domain and
		zero elsewhere
	
	"""
	for filename in glob.glob('tmp_*.png'):
		os.remove(filename)
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter
	import matplotlib.pyplot as plt
	
	Lx = 10
	Ly = 10
	Ix = lambda x,y: 0 if abs(x-Lx/2.0) > 0.1 else 1
	Iy = lambda x,y: 0 if abs(y-Ly/2.0) > 0.1 else 1
	V = None
	f = 0
	b = 0.0; q = 0; c = 1; T = 20; Nx = 40; Ny = 40; dt = 0.25
	def plot(u_num, x, xv, y, yv, t, n):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X = x
		Y = y
		X, Y = np.meshgrid(X, Y)
		Z = u_num
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				       linewidth=0, antialiased=False)
		ax.set_zlim(-1.01, 1.01)

		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		fig.colorbar(surf, shrink=0.5, aspect=5)

		plt.savefig('ANIMATION/tmp_%04d.png' % n)
		plt.close()
	solver(Iy,V,f,c,Lx,Ly,Nx,Ny,dt,T,b,
		  user_action=plot, version='vectorized')

def test_standing_undamped_waves():
	
	m,n,L,j,w,A,x,y,t,c = sym.symbols('m n L j w A x y t c')
	u_e = A*sym.cos(m*sym.pi*x/L)*sym.cos(n*sym.pi*y/j)*sym.cos(w*t)
	u_t = sym.diff(u_e,t)
	u_tt = sym.diff(u_t,t)
	u_x = sym.diff(u_e,x)
	c_u_x = c*u_x
	c_u_xx = sym.diff(c_u_x,x)
	u_y = sym.diff(u_e,y)
	c_u_y = c*u_y
	c_u_yy = sym.diff(c_u_y,y)
	
	
	f = u_tt - c_u_xx - c_u_yy
	
	m=1;n=1;L=10;j=10;w=1;A=1;c=1;Nx=40;Ny=40;T=20;b=0;dt=0.1
	

		
	
	f_ = lambda x,y,t: np.pi**2*A*c*n**2*np.cos(t*w)*np.cos(np.pi*m*x/L)*np.cos(np.pi*n*y/j)/j**2 - \
					   A*w**2*np.cos(t*w)*np.cos(np.pi*m*x/L)*np.cos(np.pi*n*y/j) + \
					   np.pi**2*A*c*m**2*np.cos(t*w)*np.cos(np.pi*m*x/L)*np.cos(np.pi*n*y/j)/L**2
	u_exact = lambda x,y,t: A*np.cos(m*np.pi*x/L)*np.cos(n*np.pi*y/j)*np.cos(w*t)
	u_t = sym.lambdify((x,y,t), u_t)
	I = lambda x,y : A*np.cos(m*np.pi*x/L)*np.cos(n*np.pi*y/j)
	V = None
	dt_values = [dt*2**(-i) for i in range(5)]

	global counter 
	counter = 0
	E_value = np.zeros(5)

	def compute_error(u_num, x, xv, y, yv, t, n):
		E = np.absolute(u_exact(x,y,t[n])-u_num).max()
		for i in range(len(dt_values)):
			if t[n]-t[n-1] == dt_values[i]:
				if E > E_value[i]:
					E_value[i] = E
		

		
			
	def convergence_rate(E, h):
		m = len(dt_values)
		r = [np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1]) for i in range(1,m, 1)]
		r = [round(r_,2) for r_ in r]
		return r

		
	for _dt in dt_values:
		solver(I,V,f_,c,L,j,Nx,Ny,_dt,T,b,
		  	   user_action=compute_error, version='vectorized')
		
	print 'Error norm values = ', E_value
	print convergence_rate(E_value, dt_values)




if __name__ == "__main__":
	if not os.path.isdir('ANIMATION'):
		os.mkdir('ANIMATION')
	#test_constant_solution(Lx=1,Ly=1,Nx=4,Ny=4,dt=0.25,T=1,b=1,c=4,u_EXACT=5)
	#test_plug()
	#test_standing_undamped_waves()
	test_gaussian()






