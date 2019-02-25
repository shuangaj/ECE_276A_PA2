import numpy as np  
import map_utils


def softmax(x):
	# Compute softmax values for each sets of scores in x
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


# Sw 				-- 		particle state in world frame
# tau 				-- 		time
# v_tau 			-- 		v*t
# angular_velocity 	-- 		omega
# return predicted particle state with noise
def prediction(Sw, tau, v_tau, angular_velocity):
	# add gaussian noise
	mu = 0 # mean
	#var = 1e-3 # var
	N = np.shape(Sw)[1] # number of particles
	# differential drive model with noise
	x_w = Sw[0,:]
	y_w = Sw[1,:]
	theta_w = Sw[2,:]
	delta_theta = angular_velocity*tau
	delta_x = v_tau*np.sin(delta_theta/2)*2/delta_theta*np.cos(theta_w+delta_theta/2)
	delta_y = v_tau*np.sin(delta_theta/2)*2/delta_theta*np.sin(theta_w+delta_theta/2)
	x_w = x_w + delta_x + np.array([np.random.normal(mu, abs(np.max(delta_x))/10 , N)])
	y_w = y_w + delta_y + np.array([np.random.normal(mu, abs(np.max(delta_y))/10 , N)])
	theta_w = theta_w + delta_theta + np.array([np.random.normal(mu, abs(delta_theta)/10 , N)])
	Sw[0,:] = x_w # new robot state in world frame
	Sw[1,:] = y_w # new robot state in world frame
	Sw[2,:] = theta_w # new robot state in world framef
	return Sw


# MAP 				--	 	MAP struct
# Sw 				-- 		particle state in world frame
# weight 			-- 		particle weights
# ranges/angles 	-- 		laser scan
# bTl 				-- 		laser frame to body frame transformation
# update the particle weights according to correlation scores
def update(MAP, Sw, weight, ranges, angles, bTl):
	# grid cells representing walls with 1
	map_wall = ((1-1/(1+np.exp(MAP['map']))) > 0.5).astype(np.int)
	x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) # x-positions of each pixel of the map
	y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) # y-positions of each pixel of the map
	# 9 x 9
	x_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # x deviation
	y_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # y deviation

  	# end point of laser beam in phisical units in laser frame
	ex = ranges*np.cos(angles)
	ey = ranges*np.sin(angles)
	# convert to homogenized coordinates in body frame
	s_h = np.ones((4,np.size(ex)))
	s_h[0,:] = ex
	s_h[1,:] = ey
	s_h = np.dot(bTl,s_h)

	numofparticles = np.shape(Sw)[1]
	correlation = np.zeros(numofparticles)
	for i in range(numofparticles):
		xt = Sw[:,i]
		# body to world transform
		x_w = xt[0]
		y_w = xt[1]
		theta_w = xt[2]
		wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w), np.cos(theta_w), 0, y_w],[0,0,1,0],[0,0,0,1]])
		# transformed into world frame
		s_w = np.dot(wTb,s_h)
		ex_w = s_w[0,:]
		ey_w = s_w[1,:]
		Y = np.stack((ex_w,ey_w))
		# calculate correlation
		c = map_utils.mapCorrelation(map_wall, x_im, y_im, Y, x_range, y_range)
		# find best correlation
		correlation[i] = np.max(c)
		# ind = np.unravel_index(np.argmax(c, axis=None), c.shape)
		# Sw[0,:]+=x_range[ind[0]]
		# Sw[1,:]+=x_range[ind[1]]

	# update particle weight
	ph = softmax(correlation)
	weight = weight*ph/np.sum(weight*ph)

	return Sw, weight




# # MAP 				--	 	MAP struct
# # Sw 				-- 		particle state in world frame
# # weight 			-- 		particle weights
# # ranges/angles 	-- 		laser scan
# # bTl 				-- 		laser frame to body frame transformation
# # update the particle weights according to correlation scores
# def update(MAP, Sw, weight, ranges, angles, bTl):
# 	# grid cells representing walls with 1
# 	map_wall = ((1-1/(1+np.exp(MAP['map']))) > 0.9).astype(np.int)
# 	x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) # x-positions of each pixel of the map
# 	y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) # y-positions of each pixel of the map
# 	# 9 x 9
# 	x_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # x deviation
# 	y_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # y deviation
# 	theta_range = np.arange(-4*MAP['res'],5*MAP['res'],MAP['res']) # theta deviation

#   	# end point of laser beam in phisical units in laser frame
# 	ex = ranges*np.cos(angles)
# 	ey = ranges*np.sin(angles)
# 	# convert to homogenized coordinates in body frame
# 	s_h = np.ones((4,np.size(ex)))
# 	s_h[0,:] = ex
# 	s_h[1,:] = ey
# 	s_h = np.dot(bTl,s_h)

# 	numofparticles = np.shape(Sw)[1]
# 	correlation = np.zeros((np.size(theta_range),numofparticles))
# 	for i in range(numofparticles):
# 		xt = Sw[:,i]
# 		# body to world transform
# 		x_w = xt[0]
# 		y_w = xt[1]
# 		for j in range(np.size(theta_range)):
# 			theta_w = xt[2] + theta_range[j]
# 			wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w), np.cos(theta_w), 0, y_w],[0,0,1,0],[0,0,0,1]])
# 			# transformed into world frame
# 			s_w = np.dot(wTb,s_h)
# 			ex_w = s_w[0,:]
# 			ey_w = s_w[1,:]
# 			Y = np.stack((ex_w,ey_w))
# 			# calculate correlation
# 			c = map_utils.mapCorrelation(map_wall, x_im, y_im, Y, x_range, y_range)
# 			# find best correlation
# 			correlation[j,i] = np.max(c)
# 	maxindx = np.argmax(correlation,axis=0)
# 	correlation = correlation[maxindx,np.arange(0,numofparticles,1)]
# 	# Sw[2,:] += theta_range[maxindx]
# 		#ind = np.unravel_index(np.argmax(c, axis=None), c.shape)

# 	# update particle weight
# 	ph = softmax(correlation)
# 	weight = weight*ph/np.sum(weight*ph)

# 	return Sw, weight


# Sw 		--		particle state in world frame
# weight 	--		particle weights
# N 		--		number of particles
# resample the particles to prevent particle depletion
def resample(Sw, weight, N):
	new_Sw = np.zeros((3, N))
	new_weight = np.tile(1/N, N).reshape(1, N)
	j = 0
	c = weight[0,0]
	for k in range(N):
		u = np.random.uniform(0, 1/N) #uniform distribution
		beta = u + k/N #scan each part in the circle
		while beta > c :
			j = j + 1
			c = c + weight[0,j] # increasing the decision section length
		new_Sw[:,k] = Sw[:,j] #if Beta is smaller than many times, put this repeated particles j in new set
		# k=1, k=2, k=3, may all have same particles X[1]
	return new_Sw, new_weight