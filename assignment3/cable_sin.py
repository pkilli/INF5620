import sympy as sym
from matplotlib.pyplot import plot, hold, legend, savefig, show, title
from numpy import linspace

a = 0
def least_squares(f, psi, Omega, symbolic=True):
    N = len(psi) - 1
    A = sym.ones(N+1, N+1)
    b = sym.ones(N+1, 1)
    x = sym.Symbol('x')
    for i in range(N+1):
        for j in range(i, N+1):
            integrand = psi[i]*psi[j]
            if symbolic:
                I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
            if not symbolic or isinstance(I, sym.Integral):
                # Could not integrate symbolically,
                # fall back on numerical integration
                integrand = sym.lambdify([x], integrand)
                I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
            A[i,j] = A[j,i] = I

        integrand = psi[i]*f
        if symbolic:
            I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
        if not symbolic or isinstance(I, sym.Integral):
            integrand = sym.lambdify([x], integrand)
            I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
        b[i,0] = I
    c = A.LUsolve(b)  # symbolic solve
    # c is a sympy Matrix object, numbers are in c[i,0]
    c = [sym.simplify(c[i,0]) for i in range(c.shape[0])]
    u = sum(c[i]*psi[i] for i in range(len(psi)))
    return u, c


def comparison_plot(f, u, Omega, n, filename=None):
    x = sym.Symbol('x')
    f = sym.lambdify([x], f, modules="numpy")
    u = sym.lambdify([x], u, modules="numpy")
    resolution = 401  # no of points in plot
    xcoor  = linspace(Omega[0], Omega[1], resolution)
    exact  = f(xcoor)
    approx = u(xcoor)
    plot(xcoor, approx)
    hold('on')
    plot(xcoor, exact)
    title('Approximating function with Galerkin method')
    legend(['approximation N = %d' %n , 'exact'])
    filename = ('%s_%03d.png' % (filename,n))
    savefig(filename)
    show()
    
def decrease_in_magnitude(c):
	c_ = 0
	for i in range(len(c)-1):
		if (c[i+1]/c[i] > c_):
			c_ = c[i+1]/c[i]
	print c_
	

def half_domain():
	"""Part a,b,c and of task 2
		approximating function 0.5*x**2 - x with two different basis functions.
		one is the 'modified sine' and the other is for 'all sine'
	"""
	x = sym.symbols('x')
	f = 0.5*x**2 - x
	N_ = [0,1,10]
	omega = [0,1]
	u_1 = 0
	for N in N_:
		psi = [sym.sin((2*i + 1)* sym.pi*x/2) for i in range(N+1)]  # 'modified sine'
		psi2= [sym.sin((i + 1)* sym.pi*x/2) for i in range(N+1)]    # 'all sine'
		u, c = least_squares(f,psi,omega, False)
		u2,c2 = least_squares(f,psi2,omega, False)
		if N is 0: 
			u_1 = u
		if N > 2:
			decrease_in_magnitude(c)
		comparison_plot(f,u,omega,N,'modified_sine_functions')
		comparison_plot(f,u2,omega,N,'all_sine_functions')
	u_1 = sym.lambdify([x], u_1, modules='numpy')
	f = sym.lambdify([x], f, modules="numpy")
	print f(1) - u_1(1)

def entire_domain():
	""" part e of task 2
		appriximating the entire domain [0,2]
		with basis function for "all sine" 
		"""
	omega = [0,2]
	for N in N_:
		psi = [sym.sin((i + 1)* sym.pi*x/2) for i in range(N+1)]
		u, c = least_squares(f,psi,omega,False)
		comparison_plot(f,u,omega,N, 'entire_domain')
half_domain()
if '__name__' == '__main__':
	half_domain()
	entire_domain()
