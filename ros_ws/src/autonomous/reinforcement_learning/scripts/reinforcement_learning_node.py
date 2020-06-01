#!/usr/bin/env python

'''
	Controls the publishing of the drive parameters and subscribing to the laser_scan.
	Controls smapling of the lidar readings. Inherited by training_node
'''

import math
import rospy
from sensor_msgs.msg import LaserScan
from drive_msgs.msg import drive_param

import torch

from topics import TOPIC_DRIVE_PARAMETERS, TOPIC_SCAN


class ReinforcementLearningNode():

    def __init__(self, actions, laser_sample_count):
        self.scan_indices = None
        self.laser_sample_count = laser_sample_count
        self.actions = actions
        self.drive_parameters_publisher = rospy.Publisher(
            TOPIC_DRIVE_PARAMETERS, drive_param, queue_size=1)
        rospy.Subscriber(TOPIC_SCAN, LaserScan, self.on_receive_laser_scan) # callback function implemented in training_node.py

    def perform_action(self, action_index):
        if action_index < 0 or action_index >= len(self.actions):
            raise Exception("Invalid action: " + str(action_index))

        angle, velocity = self.actions[action_index]
        message = drive_param()
        message.angle = angle
        message.velocity = velocity
        self.drive_parameters_publisher.publish(message)

    def convert_laser_message_to_tensor(self, message, use_device=True): # Converted to tensor to be able to be used as input to Neural Network
        if self.scan_indices is None:
            self.scan_indices = [int(i * (len(message.ranges) - 1) / (self.laser_sample_count - 1)) for i in range(self.laser_sample_count)] # Taking #laser_sample_count readings at equidistant values from all the aviable readings

        values = [message.ranges[i] for i in self.scan_indices] # storing the sampled lidar reading
        values = [v if not math.isinf(v) else 100 for v in values] # limit infinity values to 100

        return torch.tensor(
            values,
            device=device if use_device else None,
            dtype=torch.float)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
