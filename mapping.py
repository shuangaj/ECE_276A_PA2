import numpy as np
import numpy.matlib
import map_utils


# MAP 				-- 		MAP struct
# xt 				-- 		robot pose(best particle)  xt = (x, y, theta)'
# ranges/angles 	-- 		laser scan
# bTl 				-- 		laser frame to body frame transformation
# update the log odds map using laser scan
def mapupdate(MAP, xt, ranges, angles, bTl):
	# wTb -- body to world transform
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
	s_h[2,:] = 0.51435
	# transformed into world frame
	s_h = np.dot(wTb,np.dot(bTl,s_h))
	# end point of laser beam in grid units in world frame
	ex = s_h[0,:]
	ey = s_h[1,:]
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
		# MAP['map'][xis,yis] += np.log(1/4)
		if ((ex[i]>1) and (ex[i]<MAP['sizex']) and (ey[i]>1) and (ey[i]<MAP['sizey'])):
			MAP['map'][ex[i],ey[i]] += 2*np.log(4)
	
	# limit the range to prevent over-confidence
	MAP['map'] = np.clip(MAP['map'],10*np.log(1/4),10*np.log(4))
	return MAP


# It / Dt 				-- 		rgb/depth image
# bTi / bTr 				-- 		ir to body / rgb to body
# xt 	-- 		best robot pose to determine wTb
# tmap 				-- 		current texture map
# update the texture map
def texture(It, Dt, bTi, xt, tmap, invK, MAP):
	x_w = xt[0]
	y_w = xt[1]
	theta_w = xt[2]
	wTb = np.array([[np.cos(theta_w),-np.sin(theta_w),0,x_w],[np.sin(theta_w),np.cos(theta_w),0,y_w],[0,0,1,0],[0,0,0,1]])

	v = np.matlib.repmat(np.arange(0, Dt.shape[0], 1), Dt.shape[1], 1).T 
	h = np.matlib.repmat(np.arange(0, Dt.shape[1], 1), Dt.shape[0], 1) 
	v = v.reshape(1,-1)
	h = h.reshape(1,-1) 

	dd = -0.00304*Dt.reshape(1,-1) + 3.31 
	Zo = 1.03/dd
	pixels = np.vstack((h, v, Zo))
	Kpixels = np.dot(invK, pixels) 
	pixel_o = np.vstack(( Kpixels, np.ones((1,Kpixels.shape[1])) ))
	pixel_w = np.dot(np.dot(wTb,bTi), pixel_o) # (4,mn), (4,307200)
	pixel_w /= pixel_w[3]

	ground_index = pixel_w[2,:] < 0.2
	i, j = pixels[0,ground_index], pixels[1,ground_index] 

	rgbi = (i*526.37 + dd[0,ground_index]*(-4.5*1750.46) + 19276.0)/585.051
	rgbj = (j*526.37 + 16662.0)/585.051
	rgbi, rgbj = np.ceil(rgbi).astype(np.uint16)-1, np.ceil(rgbj).astype(np.uint16)-1

	mx = np.ceil((pixel_w[0,ground_index] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
	my = np.ceil((pixel_w[1,ground_index] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
	indGood1 = np.logical_and(np.logical_and(np.logical_and((mx > 1), (my > 1)), (mx < MAP['sizex'])),(my < MAP['sizey']))
	indGood2 = np.logical_and(np.logical_and(np.logical_and((rgbi > 0), (rgbj > 0)), (rgbi < Dt.shape[1])),(rgbj < Dt.shape[0]))
	indGood = np.logical_and(indGood1, indGood2)

	tmap[mx[indGood],my[indGood],:] = It[rgbj[indGood],rgbi[indGood],:]
	return tmap