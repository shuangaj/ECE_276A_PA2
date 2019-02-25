import numpy as np
import map_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.ion()
from scipy.signal import butter, lfilter
from PIL import Image
import mapping
import particle_filter

# load all the relevent data & preprocessing
#-----------------------------------------------------------------------------------------------------------------------
dataset = 20
with np.load("Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts 	4x4956
    encoder_stamps = data["time_stamps"] # encoder time stamps 	4956

with np.load("Hokuyo%d.npz"%dataset) as data:
	lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
	lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
	lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
	lidar_range_min = data["range_min"] # minimum range value [m]
	lidar_range_max = data["range_max"] # maximum range value [m]
	lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)  1081x4962
	lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans											4962

with np.load("Imu%d.npz"%dataset) as data:
	imu_angular_velocity = data["angular_velocity"][2,:] # angular velocity of yaw in rad/sec		12187
	imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements					12187

with np.load("Kinect%d.npz"%dataset) as data:
	disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images	2407
	rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images				2289

# filter the imu data sing a low pass filter with cutoff frequency = 10 Hz
fs = 1000 # sample frequency
cutoff = 10 # cutoff frequency
order = 1
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)
imu_angular_velocity = lfilter(b, a, imu_angular_velocity)

# synchronize time stamp of encoder and imu
# for each encoder time stamp, average the previous imu value
# imu_index = 0
# temp_imu = np.zeros(np.size(encoder_stamps))
# imu_size = np.size(imu_stamps)
# for i in range(1,np.size(encoder_stamps)):
# 	numberofimu = 0
# 	summation = 0
# 	while ((imu_index < imu_size) and (imu_stamps[imu_index] < encoder_stamps[i])):
# 		summation += imu_angular_velocity[imu_index]
# 		numberofimu += 1
# 		imu_index += 1
# 	temp_imu[i] = summation/numberofimu if(numberofimu > 0) else 0
# imu_angular_velocity = temp_imu


# synchronize time stamp of encoder and imu
# for each encoder time stamp, find the nearest two
temp_imu = np.zeros(np.size(encoder_stamps))
for i in range(np.size(encoder_stamps)):
	diff = abs(imu_stamps - encoder_stamps[i])
	index1 = np.argmin(diff)
	diff[index1] = 10000
	index2 = np.argmin(diff)
	temp_imu[i] = (imu_angular_velocity[index1]+imu_angular_velocity[index2])/2
imu_angular_velocity = temp_imu
#-----------------------------------------------------------------------------------------------------------------------

# init MAP
MAP = {}
MAP['res'] = 0.05	# resolution in meters
MAP['xmin'] = -30   # meters
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) 
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32) # log-odds map
texturemap = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8) # texture map
flag = 1  # if true do texture mapping

# transform from laser frame to body frame
bTl = np.array([[1,0,0,0.29833],[0,1,0,0],[0,0,1,0.51435],[0,0,0,1]])

# transformation for texture mapping
K = np.array([[585.05108211, 0, 242.94140713],[0, 585.05108211, 315.83800193],[0, 0 ,1]])
invK = np.linalg.inv(K)
roll = 0
pitch = 0.36
yaw = 0.021
Rx = np.array([ [1, 0, 0],
				[0, np.cos(roll), -np.sin(roll)],
				[0, np.sin(roll),  np.cos(roll)] ])
Ry = np.array([ [np.cos(pitch), 0, np.sin(pitch)],
				[0, 1, 0],
				[-np.sin(pitch), 0, np.cos(pitch)] ])
Rz = np.array([ [np.cos(yaw), -np.sin(yaw), 0],
				[np.sin(yaw),  np.cos(yaw), 0],
				[0, 0, 1] ])
bTi_p = np.array([0.18, 0.005, 0.36]).reshape(3,1)
bTi_R = np.dot(np.dot(Rz,Ry),Rx)
bTi = np.block([ [bTi_R, bTi_p],[np.zeros((1,3)), 1] ])



# initialize the particles in world frame, initially equal to robot state
N = 40 # number of particles
Sw = np.zeros((3,N)) # Sw = (x, y, theta)'
weight = np.array([1/N]*N).reshape(1,N) # equal weights at begining

# initialize map with first scan
angles = np.arange(-135,135.25,0.25)*np.pi/180.0
ranges = lidar_ranges[:,0]
indValid = np.logical_and((ranges < 20),(ranges > 0.5))
ranges = ranges[indValid]
angles = angles[indValid]
maxidx = np.argmax(weight)
xt = Sw[:,maxidx]
MAP = mapping.mapupdate(MAP, xt, ranges, angles, bTl)

# initialize trajectories
trajectory = np.array([[0],[0]])


tau = 1/40 
encoder_idx = 0
lidar_idx = 0
encoder_size = np.size(encoder_stamps)
lidar_size = np.size(lidar_stamps)
num_of_iteration = encoder_size + lidar_size # total number of predictions and update
for i in range(num_of_iteration):
	print('iteration'+str(i+1))
	if (i%1000==0):
		map_result = ((1-1/(1+np.exp(MAP['map']))) < 0.1).astype(np.int)
		wall_result = ((1-1/(1+np.exp(MAP['map']))) > 0.9).astype(np.int)

		# map_result = np.zeros((MAP['sizex'], MAP['sizey']))
		# # end point of laser beam in grid units in world frame
		xis = trajectory[0,:]
		yis = trajectory[1,:]
		# # end point of laser beam in grid units in world frame
		xis = np.ceil((xis - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
		yis = np.ceil((yis - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
		indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
		map_result[xis[indGood],yis[indGood]] = 2
		wall_result[xis[indGood],yis[indGood]] = 2
		texturemap[xis[indGood],yis[indGood],:] = np.array([255,0,0])
		#plt.imshow(map_result,cmap="hot");
		# plt.show()
		# plt.pause(0.001)
		# plt.savefig('map_'+str(i)+'.png')
		plt.imsave('detected_wall/'+str(i)+'.png',wall_result,cmap='hot')
		plt.imsave('detected_map/'+str(i)+'.png',map_result,cmap='hot')
		plt.imsave('texturemap/'+str(i)+'.png',texturemap)
	if ((lidar_idx==lidar_size) or (encoder_stamps[encoder_idx] < lidar_stamps[lidar_idx])):
		v_tau = np.sum(encoder_counts[:,encoder_idx])*0.0022/4 # distance traveled
		angular_velocity = imu_angular_velocity[encoder_idx]
		# prediction
		Sw = particle_filter.prediction(Sw, tau, v_tau, angular_velocity)
		if (encoder_idx < np.size(encoder_stamps)-1):
			encoder_idx += 1
		else:
			lidar_idx += 1
	else:
		# get current valid lidar measurement 
		angles = np.arange(-135,135.25,0.25)*np.pi/180.0
		ranges = lidar_ranges[:,lidar_idx]
		indValid = np.logical_and((ranges < 20),(ranges > 0.5))
		ranges = ranges[indValid]
		angles = angles[indValid]

		# update
		Sw, weight = particle_filter.update(MAP, Sw, weight, ranges, angles, bTl)
		
		# mapping
		# find best particle
		maxidx = np.argmax(weight)
		xt = Sw[:,maxidx]
		# update trajectory
		trajectory = np.hstack((trajectory,xt[0:2].reshape(2,1)))
		# update map
		MAP = mapping.mapupdate(MAP, xt, ranges, angles, bTl)

		# texture mapping
		if (flag):
			diff = abs(disp_stamps - lidar_stamps[lidar_idx])
			index1 = np.argmin(diff)
			diff = abs(rgb_stamps - lidar_stamps[lidar_idx])
			index2 = np.argmin(diff)

			img = Image.open('dataRGBD/Disparity%d/disparity%d_%d.png'%(dataset, dataset, index1+1))
			Dt = np.array(img.getdata(), np.uint16).reshape(img.size[1], img.size[0])
			img = Image.open('dataRGBD/RGB%d/rgb%d_%d.png'%(dataset, dataset, index2+1))
			It = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)

			texturemap = mapping.texture(It, Dt, bTi, xt, texturemap, invK, MAP)

		# resample the particles 
		N_eff = 1/np.dot(weight.reshape(1,N), weight.reshape(N,1))
		if N_eff < 5:
			Sw, weight = particle_filter.resample(Sw, weight, N)

		if (lidar_idx < np.size(lidar_stamps)):
			lidar_idx += 1
		else:
			encoder_idx += 1		



map_result = ((1-1/(1+np.exp(MAP['map']))) < 0.1).astype(np.int)
wall_result = ((1-1/(1+np.exp(MAP['map']))) > 0.9).astype(np.int)

xis = trajectory[0,:]
yis = trajectory[1,:]
# # end point of laser beam in grid units in world frame
xis = np.ceil((xis - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1 
yis = np.ceil((yis - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
map_result[xis[indGood],yis[indGood]] = 2
wall_result[xis[indGood],yis[indGood]] = 2
#plt.imshow(map_result,cmap="hot");
#plt.show()
#plt.savefig('map_final.png')
#plt.imsave('detected_wall/final.png',map_result,cmap='gray')
plt.imsave('detected_map/final.png',map_result,cmap='hot')
plt.imsave('detected_wall/final.png',wall_result,cmap='hot')
plt.imsave('texturemap/final.png',texturemap)

