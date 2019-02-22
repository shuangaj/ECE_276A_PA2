import numpy as np
import map_utils

# MAP -- MAP struct
# xt -- robot pose(best particle)  xt = (x, y, theta)'
# ranges/angles -- laser scan
# bTl -- laser frame to body frame transformation

def mapupdate(MAP, xt, ranges, angles, bTl):
	# body to world transform
	x_w = xt[0]
	y_w = xt[1]
	theta_w = xt[2]
	wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w), np.cos(theta_w), 0, y_w],[0,0,1,0],[0,0,0,1]])

	
	# start point of laser beam in grid units in world frame
	sx = np.ceil((x_w - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
	sy = np.ceil((y_w - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1


	# end point of laser beam in phisical units in laser frame
	ex = ranges*np.cos(angles)
	ey = ranges*np.sin(angles)
	# convert to homogenized coordinates
	s_h = np.ones((4,np.size(ex)))
	s_h[0,:] = ex
	s_h[1,:] = ey
	# transformed into world frame
	s_h = np.dot(wTb,np.dot(bTl,s_h))
	# end point of laser beam in grid units in world frame
	ex = s_h[0,:]
	ey = s_h[1,:]
	# end point of laser beam in grid units in world frame
	ex = np.ceil((ex - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
	ey = np.ceil((ey - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1


	# for each laser scan update the map
	for i in range(np.size(ranges)):
		passed_points = map_utils.bresenham2D(sx, sy, ex[i], ey[i])
		xis = passed_points[0,:].astype(np.int16)
		yis = passed_points[1,:].astype(np.int16)
		indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
		
		# update the log-odds map
		MAP['map'][xis[indGood],yis[indGood]] += np.log(1/4)
		MAP['map'][ex[i],ey[i]] += 2*np.log(4)
	
	# Prevent over-confidence
	MAP['map'] = np.clip(MAP['map'],20*np.log(1/4),20*np.log(4))
	return MAP
	# currentmap = MAP['map']
	# x = Sw[0] # in physical axis
	# y  = Sw[1] # in physical axis
	# xis = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1
	# yis = np.ceil((y - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
	# indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
	# MAP['map'][xis[indGood,yis[indGood]] = 1

 
# MAP = {}
# MAP['res'] = 0.1	# resolution in meters
# MAP['xmin'] = -30   # meters
# MAP['ymin'] = -30
# MAP['xmax'] = 30
# MAP['ymax'] = 30
# MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
# MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
# MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']))  # DATA TYPE: char or int8