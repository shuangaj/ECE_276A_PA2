import numpy as np  


# Sw -- particle state in world frame
# tau -- time
# v_tau -- v*t
# angular_velocity -- omega
# return predicted particle state with noise
def prediction(Sw,tau,v_tau,angular_velocity):
	# add gaussian noise
	mu = 0 # mean
	var = 0.095 # variance
	N = np.shape(Sw)[1] # number of particles
	# differential drive model
	delta_theta = angular_velocity*tau
	x_w = Sw[0,:]
	y_w = Sw[1,:]
	theta_w = Sw[2,:]
	x_w = x_w + v_tau*np.sin(delta_theta/2)*2/delta_theta*np.cos(theta_w+delta_theta/2) #+ np.array([np.random.normal(mu, var, N)])
	y_w = y_w + v_tau*np.sin(delta_theta/2)*2/delta_theta*np.sin(theta_w+delta_theta/2) #+ np.array([np.random.normal(mu, var, N)])
	theta_w = theta_w + delta_theta #+ np.array([np.random.normal(mu, var, N)])
	Sw[0,:] = x_w # new robot state in world frame
	Sw[1,:] = y_w # new robot state in world frame
	Sw[2,:] = theta_w # new robot state in world frame
	return Sw

	
# def update()


# def pf()


# def resample()







