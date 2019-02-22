import numpy as np
import map_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; #plt.ion()
import mapping
import locolization


# load all the relevent data
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
#-----------------------------------------------------------------------------------------------------------------------


# synchronize time stamp of encoder and imu
# for each encoder time stamp, average the nearst three imu value
imu_index = 0
temp_imu = np.zeros(np.size(encoder_stamps))
imu_size = np.size(imu_stamps)
for i in range(np.size(encoder_stamps)):
	while((imu_index < imu_size-1) and (encoder_stamps[i] > imu_stamps[imu_index])):
		imu_index+=1
	temp_imu[i] = (imu_angular_velocity[imu_index-2]+imu_angular_velocity[imu_index-1]+imu_angular_velocity[imu_index])/3
imu_angular_velocity = temp_imu		#4956
num_of_iteration = np.size(encoder_stamps)+np.size(lidar_stamps) # total number of predictions and update


# init MAP
MAP = {}
MAP['res'] = 0.1	# resolution in meters
MAP['xmin'] = -30   # meters
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) 
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']))  # log-odds map

# initialize the particles in world frame, initially equal to robot state
N = 20 # number of particles
Sw = np.zeros((3,N)) # Sw = (x, y, theta)'
weight = (np.array([1/N]*N).reshape(1, N)) # equal weights at begining

# transform from laser frame to body frame
bTl = np.array([[1,0,0,0.137],[0,1,0,0],[0,0,1,0.51],[0,0,0,1]])

encoder_idx = 0
lidar_idx = 0
encoder_size = np.size(encoder_stamps)
lidar_size = np.size(lidar_stamps)
for i in range(num_of_iteration):
	print('iteration'+str(i+1))
	if ((lidar_idx==lidar_size) or (encoder_stamps[encoder_idx]<lidar_stamps[lidar_idx])):
		tau = encoder_stamps[encoder_idx]-encoder_stamps[encoder_idx-1] if(encoder_idx!=0) else 0.025 # time
		v_tau = (encoder_counts[0][encoder_idx]+encoder_counts[1][encoder_idx]+encoder_counts[2][encoder_idx]+encoder_counts[3][encoder_idx])/4*0.0022 # distance traveled
		angular_velocity = imu_angular_velocity[encoder_idx]
		Sw = locolization.prediction(Sw,tau,v_tau,angular_velocity)
		if (encoder_idx < np.size(encoder_stamps)-1):
			encoder_idx+=1
		else:
			lidar_idx+=1
	else:
		#locolization.update()

		

		# mapping
		# get valid lidar measurement 
		angles = np.arange(-135,135.25,0.25)*np.pi/180.0
		ranges = lidar_ranges[:,lidar_idx]
		indValid = np.logical_and((ranges < 30),(ranges > 0.1))
		ranges = ranges[indValid]
		angles = angles[indValid]

		# find best particle
		maxidx = np.argmax(weight)
		xt = Sw[:,maxidx]

		# update map
		MAP = mapping.mapupdate(MAP, xt, ranges, angles, bTl)
		map_result = ((1-1/(1+np.exp(MAP['map']))) > 0.5).astype(np.int)
		# if (i%100==0):
		# 	plt.imshow(map_result,cmap="hot");
		# 	plt.show()
		# 	plt.pause(0.1)
		if (lidar_idx < np.size(lidar_stamps)):
			lidar_idx+=1
		else:
			encoder_idx+=1		





map_result = ((1-1/(1+np.exp(MAP['map']))) < 0.2).astype(np.int)
plt.imshow(map_result,cmap="hot");
plt.show()





