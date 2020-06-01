#!/usr/bin/env python

'''
	This is the file that launches the Q_learning node for driving the car. 
'''

from reinforcement_learning_node import ReinforcementLearningNode, device
import os
import rospy
from parameters_q_learning import NeuralQEstimator, ACTIONS, LASER_SAMPLE_COUNT
import torch


class QLearningDrivingNode(ReinforcementLearningNode):
    
    def __init__(self):	# Intializes the Neural Network and saves it into policy from parameters in parameters_q_learning
        self.policy = NeuralQEstimator().to(device)

        try:
            self.policy.load()
            self.policy.eval()
        except IOError:
            message = "Model parameters for the neural net not found. You need to train it first."
            rospy.logerr(message)
            rospy.signal_shutdown(message)
            exit(1)

        ReinforcementLearningNode.__init__(self, ACTIONS, LASER_SAMPLE_COUNT)

    def on_receive_laser_scan(self, message):
        if self.policy is None:
            return

        state = self.convert_laser_message_to_tensor(message)

        with torch.no_grad():
            action = self.policy(state).max(0)[1].item()
        self.perform_action(action)


rospy.init_node('q_learning_driving', anonymous=True)
node = QLearningDrivingNode()
rospy.spin()
