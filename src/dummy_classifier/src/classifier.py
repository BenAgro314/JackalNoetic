#!/usr/bin/env python

import rospy
import torch
import os
import ros_numpy
from ros_numpy import point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

class Classifier:

	def __init__(self, in_topic, out_topic):
		self.pub = rospy.Publisher(out_topic, PointCloud2, queue_size=10)
		rospy.init_node('classifier', anonymous = True)
		rospy.Subscriber(in_topic, PointCloud2, self.lidar_callback)
		rospy.spin()

	def network_inference(self, points):
		"""
		Function simulating a network inference.
		:param points: The input list of points as a numpy array (type float32, size [N,3])
		:return: predictions : The output of the network. Class for each point as a numpy array (type int32, size [N])
		"""    
		
		########
		# Init #
		########    
		
		# Set which gpu is going to be used on the machine
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



	def lidar_callback(self, cloud):
		rospy.loginfo("Received Point Cloud")

		# turn pointcloud into numpy record array
		labeled_points = pc2.pointcloud2_to_array(cloud) 
		labeled_points = labeled_points.copy()
	
		
		# turn the record array into an [N,3] sized numpy array of floats 
		xyz_points= pc2.get_xyz_points(labeled_points)
		
		
		# generate prediction, returns [N,1] of classes 
		predictions = self.network_inference(xyz_points)
		
		#add class to labeled points (replacing intensity)
		labeled_points = labeled_points.copy()
		new_dtype = np.dtype({'names':['x','y','z','intensity','ring'], 'formats':['<f4','<f4','<f4','<f4','<u2'],'offsets':[0,4,8,16,20], 'itemsize':32})
		labeled_points.dtype = new_dtype
		labeled_points['class'] = predictions
		
		# generate new pointcloud message with classes 
		msg = pc2.array_to_pointcloud2(labeled_points, rospy.Time.now(), cloud.header.frame_id)
		self.pub.publish(msg)
		
		rospy.loginfo("Sent Pointcloud")
		

if __name__ == '__main__':
	L = Classifier("/velodyne_points", "/classified_points")
