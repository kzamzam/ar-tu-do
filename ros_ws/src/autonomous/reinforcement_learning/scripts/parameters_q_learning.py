'''
	Class that contains the neural network and the other hyper parameters
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import rospy

from rospkg import RosPack

# General parameters
ACTIONS = [(-0.6, 0.2), (0.6, 0.2), (0, 0.2)]
ACTION_COUNT = len(ACTIONS)

# Only use some of the LIDAR measurements
# When changing this value, also update laser_sample_count in q_learning.launch
LASER_SAMPLE_COUNT = 128 # was 8 which was a problem since in q_learning.launch it is 128


MODEL_FILENAME = os.path.join(RosPack().get_path(
    "reinforcement_learning"), "q_learning.to")

# Training parameters

# Start by loading previously trained parameters.
# If this is False, training will start from scratch
CONTINUE = False

DISCOUNT_FACTOR = 0.99  # aka gamma

MAX_EPISODE_LENGTH = 1000 #was 500 still needs tuning
# Sample neural net update batch from the replay memory.
# It contains this many transitions.
MEMORY_SIZE = 100000 #was 5000 which is too small, anyway needs tuning later

BATCH_SIZE = 64 #was 128
LEARNING_RATE = 0.0001

# Probability to select a random episode starts at EPS_START
# and reaches EPS_END once EPS_DECAY episodes are completed.
EPS_START = 0.99
EPS_END = 0.01 # was 0.3 but should be much lower
EPS_DECAY = 8000 #was 10000 which caused very slow decay, can be fine tuned later


class NeuralQEstimator(nn.Module): # The neural network which will be used as a policy netowrk to decide best action
    def __init__(self):
        super(NeuralQEstimator, self).__init__()
        self.fc1 = nn.Linear(LASER_SAMPLE_COUNT, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, ACTION_COUNT)

    def forward(self, x): # In torch forward propagation must be implemented manually
        x = F.relu(self.fc1(x)) # RELU activation function for the 2 mid layers
        x = F.relu(self.fc2(x))
        return self.fc3(x) # last layer is not activated since it may be a negative value 

    def load(self):
        self.load_state_dict(torch.load(MODEL_FILENAME))
        rospy.loginfo("Model parameters loaded.")

    def save(self):
        torch.save(self.state_dict(), MODEL_FILENAME)
