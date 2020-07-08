#!/usr/bin/env python

import rospy
import torch
import os
import ros_numpy
from ros_numpy import point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

class LidarListener:

	def __init__(self, topic):
		rospy.init_node('lidar_listener', anonymous = True)
		rospy.Subscriber(topic, PointCloud2, self.LidarCallback)
		rospy.spin()

	def network_inference_dummy(self, points):
		"""
		Function simulating a network inference.
		:param points: The input list of points as a numpy array (type float32, size [N,3])
		:return: predictions : The output of the network. Class for each point as a numpy array (type int32, size [N])
		"""    
		
		########
		# Init #
		########    
		
		# Set which gpu is going to be used o nthe machine
		GPU_ID = '1'    
		# Set this GPU as the only one visible by this python script
		os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID  
		# Convert points to a torch tensor
		points = torch.from_numpy(points)    
		# Get the GPU for PyTorch
		if torch.cuda.is_available():
			device = torch.device("cuda:0")
		else:
			device = torch.device("cpu")    
		
		# Convert points to a cuda tensor
		points = points.to(device)    
		#####################
		# Network inference #
		#####################    
		# Instead of network inference, just create dummy classes    
		# Sum all points along the dimension 1. [N, 3] => [N]
		predictions = torch.sum(points, dim=1)    
		# Convert to integers
		predictions = torch.floor(predictions)
		predictions = predictions.type(torch.int32)    
		##########
		# Output #
		##########    
		# Convert from pytorch cuda tensor to a simple numpy array
		predictions = predictions.detach().cpu().numpy()    
		return predictions

	def LidarCallback(self, cloud):
		#rospy.loginfo("received pointcloud " + str(type(cloud)))
		points = pc2.pointcloud2_to_array(cloud)
		points= pc2.get_xyz_points(points)
		# points[:,0]=pc['x']
		# points[:,1]=pc['y']
		# points[:,2]=pc['z']

		rospy.loginfo("points shape: " + str(np.shape(points)))
		
		
		predictions = self.network_inference_dummy(points)
		rospy.loginfo("predictions shape: " + str(np.shape(predictions)))

if __name__ == '__main__':
	L = LidarListener("/velodyne_points")
