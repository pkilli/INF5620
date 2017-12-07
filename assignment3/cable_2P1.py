import fe_approx1D
import sympy as sym
import numpy as np

x = sym.symbols('x')
f = 0.5*x**2 -x

phi = fe_approx1D.basis(1)
nodes,elements = fe_approx1D.mesh_uniform(2,1,[0,1],symbolic=False)
A, b = fe_approx1D.assemble(nodes, elements, phi, f, symbolic=False)
c = np.linalg.solve(A,b)
f = sym.lambdify([x], f, modules='numpy')
xf = np.linspace(0,1,10001)
U = np.asarray(c)
xu, u = fe_approx1D.u_glob(U, elements, nodes) 
import scitools.std as plt

u[0] = 0
plt.plot(xu, u, '-', xf, f(xf), '--')
plt.legend(['u', 'f'])
plt.savefig('cable_2P1_modified.png')
xf = np.linspace(xu[1],1,10001)
plt.plot(xu[1:],u[1:],'-',xf,f(xf),'--')
plt.legend(['u','f'])
plt.savefig('cable_2P1_excluded.png')


