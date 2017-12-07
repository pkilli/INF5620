"""
1D wave equation with Dirichlet or Neumann conditions
and variable wave velocity::

 u, x, t, cpu = solver(I, V, f, c, U_0, U_L, L, dt, C, T,
                       user_action=None, version='scalar',
                       stability_safety_factor=1.0)

Solve the wave equation u_tt = (c**2*u_x)_x + f(x,t) on (0,L) with
u=U_0 or du/dn=0 on x=0, and u=u_L or du/dn=0
on x = L. If U_0 or U_L equals None, the du/dn=0 condition
is used, otherwise U_0(t) and/or U_L(t) are used for Dirichlet cond.
Initial conditions: u=I(x), u_t=V(x).

T is the stop time for the simulation.
dt is the desired time step.
C is the Courant number (=max(c)*dt/dx).
stability_safety_factor enters the stability criterion:
C <= stability_safety_factor (<=1).

I, f, U_0, U_L, and c are functions: I(x), f(x,t), U_0(t),
U_L(t), c(x).
U_0 and U_L can also be 0, or None, where None implies
du/dn=0 boundary condition. f and V can also be 0 or None
(equivalent to 0). c can be a number or a function c(x).

user_action is a function of (u, x, t, n) where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""

import sympy as sy
import numpy as np
import time


def solver(I, V, f, c, U_0, U_L, L, dt, C, T, version,stab_factor,user_action=None):
    """Solve u_tt=(c^2*u_x)_x + f on (0,L)x(0,T]."""
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)      # Mesh points in time

    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 101)])
    dx = dt*c_max/(stab_factor*C)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)          # Mesh points in space

    # Treat c(x) as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    # Wrap user-given f, I, V, U_0, U_L if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0


    u   = np.zeros(Nx+1)   # Solution array at new time level
    u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

    import time;  t0 = time.clock()  # CPU time measurement

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    i = Ix[0]
    if U_0 is None:
        # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_1[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                       0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
        0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_0(dt)

    i = Ix[-1]
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_1[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                       0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
        0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        if version == 'scalar':
            for i in Ix[1:-1]:
                u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - \
                        0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
                dt2*f(x[i], t[n])

        elif version == 'vectorized':
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
            C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) -
                0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + \
            dt2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        i = Ix[0]
        if U_0 is None:
            # Set boundary values
            # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
            # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
            ip1 = i+1
            im1 = ip1
            u[i] = - u_2[i] + 2*u_1[i] + \
                   C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                       0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
            dt2*f(x[i], t[n])
        else:
            u[i] = U_0(t[n+1])

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1
            u[i] = - u_2[i] + 2*u_1[i] + \
                   C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
                       0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
            dt2*f(x[i], t[n])
        else:
            u[i] = U_L(t[n+1])

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = t0 - time.clock()
    return cpu_time

def test_convergence_rate(L,w,q,u, u_exact):
    """ finding the convergence rates for several dt's
    testing the scheme against a known solution. Using sympy to find source term
    
    """
    
    x,t,w,L = sy.symbols("x t w L")
    
    #Find source term: f
    #Find u_tt
    u_tt = sy.diff(u,t,t)
    #Find q*u_x, first u_x

    u_x = sy.diff(u,x)
    q_u_x = q*u_x
    q_u_xx = sy.diff(q_u_x,x)
    
    f = u_tt - q_u_xx
    f = sy.lambdify((x,t),f)
    u = sy.lambdify((x,t),u)
    q = sy.lambdify((x),q)
    L=1
    w=1
    c = lambda x : np.sqrt(q(x))
    U_0 = None
    U_L = None

    V = None
    I = lambda x : u(x,0)
    C = 0.89
    dt = 0.1
    T = 2
    stab_factor = 1.0     	
    
    dt_values = [dt*2**(-i) for i in range(5)]
    E_values = []
    
    def plot(u,x,t,n):
    	"""user_action function for solver."""
    	import matplotlib.pyplot as plt
    	plt.plot(x, u, 'r-')
    	plt.draw()
    	time.sleep(2) if t[n] == 0 else time.sleep(0.2)
    
    class Action:
        """Store last solution."""
        def __call__(self, u, x, t, n):
            if n == len(t)-1:
            	self.u = u.copy()
            	self.x = x.copy()
            	self.t = t[n]
                
    action = Action()
    	
    for _dt in dt_values:
    	dx = solver(I,V,f,c,U_0,U_L,L,_dt,C,T,"scalar",stab_factor,user_action=action)
    	u_num = action.u
    	#E = np.sqrt(dx*sum(u_exact(action.x, action.t)-u_num)**2)
    	E = np.absolute(u_exact(action.x, action.t)-u_num).max() #sup norm 
    	E_values.append(E)


    def convergence_rate(E, h):
    	m = len(dt_values)
    	r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1,m, 1)]
    	r = [round(r_,2) for r_ in r]
    	return r
    	
    solver(I,V,f,c,U_0,U_L,L,dt,C,T,"scalar",stab_factor,user_action=plot)
    
    return convergence_rate(E_values, dt_values)

if __name__ == "__main__":
	print "Task a:"
	x,t,w,L = sy.symbols("x t w L")
	L = 1
	w = 1
	u_exact = lambda x,t: np.cos(np.pi*x/float(L))*np.cos(w*t)
	q_a = 1+(x-(L)/2)**4
	u = sy.cos(sy.pi*x/float(L))*sy.cos(w*t)
	r1 = test_convergence_rate(L,w,q_a,u,u_exact)
	print r1
	print "----------"
	print "Task b"
	q_b = 1 + sy.cos(sy.pi*x/L)
	r2 = test_convergence_rate(L,w,q_b,u,u_exact)
	print r2
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
