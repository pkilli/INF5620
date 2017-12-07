import numpy as np

import matplotlib.pyplot as plt

def simulate(beta,Theta, epsilon, num_periods, 
				time_steps_per_period, plot=True):
	# beta = dimensionless parameter
	# theta = initial angle in degrees
	# epsilon = initial stretch of wire
	# num_periods = simulate for num_periods
	# time_steps_per_period=time step resolution
	# plots = make plots or not
	
	THETA = Theta*np.pi/180
	P = 2*np.pi
	T = int(num_periods * P)
	dt = P/time_steps_per_period
	n = num_periods * time_steps_per_period
	
	t = np.linspace(0, n*dt, n+1)
	X = np.zeros(n+1)
	Y = np.zeros(n+1)
	
	X[0] = (1+epsilon)*np.sin(THETA)
	Y[0] = 1-(1+epsilon)*np.cos(THETA)
	
	L = float(np.sqrt(X[0]**2 + (Y[0]-1)**2))
	
	X[1] = 0.5*((-beta/(1-beta)*(1-beta/L)*X[0])*dt**2 + 2*X[0])
	Y[1] = 0.5*((-beta/(1-beta)*(1-beta/L)*(Y[0]-1) - beta)*dt**2 + 2*Y[0])

	for i in range(1, n):
		L = float(np.sqrt(X[i]**2 + (Y[i]-1)**2))      # Makes new L value every step
		X[i+1] = (-beta/(1.0-beta)*(1.0-beta/L)*X[i])*dt**2 + 2*X[i] - X[i-1]
		Y[i+1] = (-beta/(1.0-beta)*(1.0-beta/L)*(Y[i]-1)-beta)*dt**2 + 2*Y[i] - Y[i-1]
	
	_theta = np.zeros(n+1)
	for j in range(0,n):
		_theta[j] = np.arctan(X[j]/(1.0-Y[j]))
		
	
	if (plot == True):
		#plt.gca().set_aspect('equal')
		plt.plot(X,Y)
		plt.show()
		
		if (Theta < (10)):
			plt.plot(t, _theta)
			plt.plot(t, THETA*np.cos(t))
			plt.show()
	return X, Y, _theta
	
	
def test_1():
	""" test: Theta = 0 and epsilon = 0 gives 0 for every X and Y value"""
	
	x, y, theta = simulate(beta=0.9,Theta=0, epsilon=0, num_periods=6, 
					time_steps_per_period=60, plot = False)
	print "test: Theta = 0 and epsilon = 0 should give 0 for every X and Y value:"
	testOK = True
	for i in range(len(x)):
		if x[i] or y[i] != 0.0:			
			testOK = False
			
	if testOK == True:
		print "Test OK!"
	else:
		print "Test failed" 

def demo(beta, Theta):
	simulate(beta,Theta, epsilon=0, num_periods=3, 
					time_steps_per_period=600, plot = True)
	
if __name__ == '__main__':
	simulate(beta=0.9,Theta=5, epsilon=0, num_periods=6, 
				time_steps_per_period=60, plot = False)
	test_1()
	demo(0.9,5)


