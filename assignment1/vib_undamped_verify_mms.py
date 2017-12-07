import sympy as sym
import numpy as np
V, t, I, b, w, dt = sym.symbols('V t I b w dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
	"""Return the residual of the discrete eq. with u inserted."""
	
	R = sym.diff(u(t), t,t) - (u(t+dt)-2*u(t)+u(t-dt))/dt**2	
	return sym.simplify(R)

def residual_discrete_eq_step1(u):
	"""Return the residual of the discrete eq. at the first
	step with u inserted."""
	#u_1 = 1/2*(sym.diff(u(t), t,t).subs(t, 0) - w**2*u(0))*dt**2 + u(0) + dt*V    
	u_1 = 1/2.0*(ode_source_term(u).subs(t, 0) - w**2*u(0))*dt**2 + u(0) + dt*V
	R = u(t).subs(t, dt) - u_1
	return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    DtDt = (u(t+dt)-2*u(t)+u(t-dt))/dt**2
    return DtDt
    
def solver():
	"""solves u`` + w**2u = f(t) for t in (0, T], 
		u(0) = I, u`(0) = V."""
	dt=0.1; w=1; V=1; I=1; T=1
	
	dt = float(dt); w = float(w) # avoid integer div.
	Nt = int(round(T/dt))
	u = np.zeros(Nt+1)
	t = np.linspace(0, Nt*dt, Nt+1)
	f = lambda t: w**2*(V*t + I)

	u[0] = I
	u[1] = 0.5*(f(0)-w**2*u[0])*dt**2 + u[0] + dt*V
	
	for n in range(1,Nt):
		u[n+1] = (f(t[n]) - w**2*u[n])*dt**2 + 2*u[n] - u[n-1]
	
	return u, t

def main(u,g):
	"""
	Given some chosen solution u (as a function of t, implemented
	as a Python function), use the method of manufactured solutions
	to compute the source term f, and check if u also solves
	the discrete equations.
	"""
	print '=== Testing exact solution: %s ===' % g
	print "Initial conditions u(0)=%s, u'(0)=%s:" % \
		(u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

	# Method of manufactured solution requires fitting f
	global f  # source term in the ODE
	f = sym.simplify(ode_source_term(u))
	print 'f(t): %s ' % f
	   
	# Residual in discrete equations (should be 0)
	print 'residual step1:', residual_discrete_eq_step1(u)
	print 'residual:', residual_discrete_eq(u)

	#numerical solution


def linear():
		g = V*t + I
		main(lambda t: V*t + I, g)
    
def quadratic():
		g = b*t**2 + V*t + I
		main(lambda t: b*t**2 + V*t + I, g)

if __name__ == '__main__':
	linear()
	quadratic()
	u,t1 = solver()
	print u
	print t1
