#!/usr/bin/env python
# coding: utf-8

'''
The file launched by dueling_deep_q_learning.launch. Launches ros node to train the dueling double deep Q-learning model.
'''

import os
import numpy as np
import torch as T
import torch.nn as nn  # access to neural network layers
import torch.nn.functional as F  # access to activation functions
import torch.optim as optim  # access to optimizers
from collections import deque
import time
import rospy
import math
import random
import sys
from tensorflow.keras.models import load_model
from datetime import datetime

from std_srvs.srv import Empty as EmptyS
from std_srvs.srv import EmptyRequest
from std_msgs.msg import Empty
from reinforcement_learning.msg import EpisodeResult
from gazebo_msgs.msg import ModelStates
import simulation_tools.reset_car as reset_car
from simulation_tools.track import track
from simulation_tools.reset_car import Point
from sensor_msgs.msg import LaserScan
from drive_msgs.msg import drive_param

from topics import TOPIC_CRASH, TOPIC_GAZEBO_MODEL_STATE, TOPIC_EPISODE_RESULT
from topics import TOPIC_DRIVE_PARAMETERS, TOPIC_SCAN, TOPIC_RUN_STEP

# Hyper parameters
mem_size = 10000
gamma = 0.99
epsilon = 0.99
learning_rate = 0.0001
batch_size = 32
actions = [(-0.78, 0.2), (-0.65, 0.2), (-0.3, 0.2), (-0.2, 0.2), (-0.14, 0.2), (-0.09, 0.2), (-0.052, 0.2), (-0.035, 0.2), (-0.017, 0.2),
           (0, 0.2),
           (0.78, 0.2), (0.65, 0.2), (0.3, 0.2), (0.2, 0.2), (0.14, 0.2), (0.09, 0.2), (0.052, 0.2), (0.035, 0.2), (0.017, 0.2)]
n_actions = len(actions)
eps_min = 0.01
eps_start = 0.99
eps_dec = 9000
replace = 100
max_episode_length = 1000
pretrain = 500

Train = True # Set to true when you want to train the NN, and false if you want to load the prev. trained model and evaluate it
Encoded = False  # Set to true when you want to use the encoded version of the lidar
if Encoded:
    laser_sample_count = 10
else:
    laser_sample_count = 180


class ReplayBuffer():  # This class handles how the agent stores the transitions in memory
    def __init__(self, max_size, input_shape):  # input shape is shape of states to store in mem.
        self.mem_size = max_size
        self.mem_cntr = 0  # used to know in which location we are in mem.
        # we make an np array for each thing that is stored in mem.
        self.state_memory = np.zeros((self.mem_size, input_shape),  # input shape to tensor of laser_sample_message?
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  # so that it loops around
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)  # We take min so if the mem. is not filled we don't take zeros
        batch = np.random.choice(max_mem, batch_size, replace=False)  # replace=false so once it selects a memory it doesn't select it again

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingDeepQNetwork(nn.Module):  # The class that handles the dueling deep q network itself
    def __init__(self, lr, n_actions, name, laser_sample_count, chkpt_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(laser_sample_count, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)

        self.V = nn.Linear(32, 1)  # this gives the value of the state which is a scalar
        self.A = nn.Linear(32, n_actions)  # this gives the advantage for each action

        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # stochastic gradient descent, with momentum and adaptive lr
        self.loss = nn.MSELoss()  # Mean squared error as loss function
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')  # selects cpu by default in case gpu isnt compatible
        self.to(self.device)  # send this network to the appropriate device

    def forward(self, state):  # implemenatation of the forward pass in the NN
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        flat3 = F.relu(self.fc3(flat2))
        V = self.V(flat3)
        A = self.A(flat3)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)  # saving the network's state dictionary provided by pytorch

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, laser_sample_count,
                 mem_size, batch_size, actions, eps_min, eps_start, eps_dec,
                 replace, max_episode_length):
        if Encoded:
            chkpt_dir = '/home/zamzam/ar-tu-do/ros_ws/src/autonomous/reinforcement_learning/models/Encoded'
        else:
            chkpt_dir = '/home/zamzam/ar-tu-do/ros_ws/src/autonomous/reinforcement_learning/models/Full_trackOpt'
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.laser_sample_count = laser_sample_count
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        # Added later on
        self.max_episode_length = max_episode_length
        self.actions = actions
        # stuff mostly for bookeeping
        self.episode_count = 0
        self.episode_length = 0
        self.total_step_count = 0
        self.cumulative_reward = 0
        self.done = False
        self.encoder = load_model('/home/zamzam/ar-tu-do/ros_ws/src/autonomous/reinforcement_learning/scripts/weights/encoder_weights.h5')

        self.net_output_debug_string = ""
        self.episode_length_history = deque(maxlen=50)
        self.cumulative_reward_history = deque(maxlen=50)

        self.state = None
        self.action = None
        self.car_position = None
        self.car_orientation = None

        self.drive_forward = None
        self.steps_with_wrong_orientation = 0

        self.episode_start_time_real = time.time()
        self.episode_start_time_sim = rospy.Time.now().to_sec()

        self.scan_indices = None

        reset_car.register_service()

        # Service to pause and unpause gazebo
        self.pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', EmptyS)
        self.unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', EmptyS)

        # Subscribers
        rospy.Subscriber(TOPIC_CRASH, Empty, self.on_crash)
        rospy.Subscriber(TOPIC_GAZEBO_MODEL_STATE, ModelStates, self.on_model_state_callback)
        rospy.Subscriber(TOPIC_SCAN, LaserScan, self.on_receive_laser_scan, queue_size=1)
        # Publishers
        self.episode_result_publisher = rospy.Publisher(
            TOPIC_EPISODE_RESULT, EpisodeResult, queue_size=1)
        self.drive_parameters_publisher = rospy.Publisher(
            TOPIC_DRIVE_PARAMETERS, drive_param, queue_size=1)
        self.run_step_publisher = rospy.Publisher(
            TOPIC_RUN_STEP, Empty, queue_size=1)

        self.memory = ReplayBuffer(mem_size, laser_sample_count)  # extra argument here was deleted
        # The network below will calculate which action to take
        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   laser_sample_count=self.laser_sample_count,
                                   name='ddqn_q_eval',
                                   chkpt_dir=self.chkpt_dir)

        # The network below will evaluate this action to correct and retrain the first network
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                   laser_sample_count=self.laser_sample_count,
                                   name='ddqn_q_next',
                                   chkpt_dir=self.chkpt_dir)
        if not Train:
            self.q_eval.load_checkpoint()
            self.q_next.load_checkpoint()

    def choose_action(self, state):
        if np.random.random() > self.epsilon or not Train:
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)  # the store transition implemented in replay buffer is called

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        #self.epsilon = self.epsilon - self.eps_dec \
        #                if self.epsilon > self.eps_min else self.eps_min
        self.epsilon = self.eps_min + (self.eps_start - self.eps_min) * \
        math.exp(-1. * self.total_step_count / self.eps_dec)

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < pretrain:  # in case we haven't filled enough memory yet we just stop
            return

        self.q_eval.optimizer.zero_grad()  # We must do this as torch accumlates gradients

        self.replace_target_network()  # check if it's time to replace the target network

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        # Now we calculate the Q according to the action the agent actually took using the eval network, for the whole batch
        # Notice the we add the value and advantage function - it's mean across all the actions, because that's what was presented in the paper to avoid the identifiability problem
        q_pred = T.add(V_s,
                        (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        # Now we calculate the q_next which is the Q of the next states using the next (also called target) network
        q_next = T.add(V_s_,
                        (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        # Now we calculate the q_next which is the Q of the next states using the eval network
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)  # action that gives best Q according to eval network

        q_next[dones] = 0.0  # if the next state was terminal then reward is 0
        q_target = rewards + self.gamma*q_next[indices, max_actions]  # this is now our target values calculated from equation from theory of RL, where action is selected by eval nn, but evaluated by next (target) nn

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def get_reward(self):
        '''
        method to return the reward based on location of the agent(car).
        this reward is based on the closer the agent to the center line of the track the higher the reward
        '''
        track_position = track.localize(self.car_position)
        distance = abs(track_position.distance_to_center)  # distance to enter line of the track

        return 1.25 - distance/1.38  #1.38 is the width of the track

    def on_complete_step(self, state, action, reward, state_, done):
        self.store_transition(state, action, reward, state_, done)  # add necessary info to replay memory
        if Train:
            self.learn()  # train the policy NN

    def on_crash(self, _):
        if self.episode_length > 5:
            self.done = True

    def get_episode_summary(self):
        return self.get_episode_summary_helper() + ' ' \
            + ("memory: {0:d} / {1:d}, ".format(self.memory.mem_cntr, mem_size) if self.memory.mem_cntr < mem_size else "") \
            + "ε-greedy: " + str(int(self.epsilon * 100)) + "% random, " \
            + "replays: " + str(self.learn_step_counter) + ", " \
            + "q: [" + self.net_output_debug_string + "], "

    def get_episode_summary_helper(self):
        average_episode_length = sum(
            self.episode_length_history) / len(self.episode_length_history)
        average_cumulative_reward = sum(
            self.cumulative_reward_history) / len(self.cumulative_reward_history)

        result = "Episode " + str(self.episode_count) + ": " \
            + str(self.episode_length).rjust(3) + " steps (" + str(int(average_episode_length)).rjust(3) + " avg), " \
            + "return: " + "{0:.1f}".format(self.cumulative_reward).rjust(5) \
            + " (" + "{0:.1f}".format(average_cumulative_reward).rjust(5) + " avg)"

        episode_duration_real = time.time() - self.episode_start_time_real
        episode_duration_sim = rospy.Time.now().to_sec() - self.episode_start_time_sim
        if episode_duration_real != 0:
            result += ", time: {0:.1f}x".format(episode_duration_sim / episode_duration_real)  # nopep8
        if episode_duration_sim != 0:
            result += ", laser: {0:.1f} Hz".format(float(self.episode_length) / episode_duration_sim)  # nopep8
        return result

    def on_complete_episode(self):
        self.episode_length_history.append(self.episode_length)
        self.cumulative_reward_history.append(self.cumulative_reward)
        self.episode_result_publisher.publish(
            reward=self.cumulative_reward, length=int(
                self.episode_length))

        rospy.loginfo(self.get_episode_summary())
        self.episode_start_time_real = time.time()
        self.episode_start_time_sim = rospy.Time.now().to_sec()

        self.episode_count += 1
        self.episode_length = 0
        self.cumulative_reward = 0

        if self.episode_count % 50 == 0 and Train:  # saves NN parameters ever 50 episodes
            self.save_models()

    def on_receive_laser_scan(self, message):
        '''
        	Callback function when laser message is recieved from lidar.
        '''
        self.pause_physics_client(EmptyRequest())
        if Encoded:
            x = np.asarray(message.ranges)
            x = x.reshape(-1,180,1)
            res = self.encoder.predict(x)
            state_ = res.reshape(10)
            state_ = T.tensor(state_,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu'),
            dtype=T.float)
        else:
            state_ = self.convert_laser_message_to_tensor(message) # convert to tensor of size depending on var. LASER_SAMPLE_COUNT.

        if self.state is not None:
            self.check_car_orientation() # making sure car is running in correct direction before giving any rewards
            reward = self.get_reward()
            if self.done:
                reward = 0  #In case car crashed it recieves 0
            self.cumulative_reward += reward # calc. new cumlative reward for this episode
            self.on_complete_step(self.state, self.action, reward, state_, self.done)

        if self.done or self.episode_length >= self.max_episode_length: # terminal: agent crashes or takes >2 steps in worng dir
            self.drive_forward = True #random.random() > 0.5 # Probability of moving in forward or backward direction when car resets
            reset_car.reset_random(
                max_angle=math.pi / 180 * 20, # max angle deviation from proper heading of 20 degrees
                max_offset_from_center=0.2,
                forward=self.drive_forward)
            self.done = False # resetting the terminal step since car will start in random new position
            self.state = None	# same as above
            if self.episode_length != 0:
                self.on_complete_episode() # this episode is done
            self.unpause_physics_client(EmptyRequest())
        else:  # will be entered every step of the episode if non terminal
            self.state = state_  # set new tensor data from lidar as the current state
            self.action = self.choose_action(state_)
            self.unpause_physics_client(EmptyRequest())
            self.perform_action(self.action)
            self.episode_length += 1
            self.total_step_count += 1


    def check_car_orientation(self):
        '''
        	Method that checks if car is in moving in the right direction
        '''
        if self.car_position is None:	# if car position is not recieved yet from ros through TOPIC_GAZEBO_MODEL_STATE
            return

        track_position = track.localize(self.car_position)	# getting the track position from the poistion in world space
        car_direction = track_position.faces_forward(self.car_orientation)  # saving which direction the car is facing
        if car_direction != self.drive_forward:	# cheking if car moves in same direction as chosen randomly before while resetting car
            self.steps_with_wrong_orientation += 1
        else:
            self.steps_with_wrong_orientation = 0

        if self.steps_with_wrong_orientation > 2:
            self.done = True

    def on_model_state_callback(self, message):
        '''
        	callback function that sets the position and orientation of car based on message recieved from gazebo
        '''
        if len(message.pose) < 2:
            return
        self.car_position = Point(       # Setting car position to named tupple "Point" with data from pose
            message.pose[1].position.x,
            message.pose[1].position.y)
        self.car_orientation = message.pose[1].orientation

    def perform_action(self, action_index):
        if action_index < 0 or action_index >= len(self.actions):
            raise Exception("Invalid action: " + str(action_index))

        angle, velocity = self.actions[action_index]
        message = drive_param()
        message.angle = angle
        message.velocity = velocity
        self.drive_parameters_publisher.publish(message)
        self.run_step_publisher.publish()

    def convert_laser_message_to_tensor(self, message, use_device=True): # Converted to tensor to be able to be used as input to Neural Network
        # Edit this to make sure that each value correspond to a certain angle, may need deeper knowledge in gazebo
        if self.scan_indices is None:
            self.scan_indices = [int(i * (len(message.ranges) - 1) / (self.laser_sample_count - 1)) for i in range(self.laser_sample_count)] # Taking #laser_sample_count readings at equidistant values from all the aviable readings

        values = [message.ranges[i] for i in self.scan_indices] # storing the sampled lidar reading
        values = [v if not math.isinf(v) else 100 for v in values] # limit infinity values to 100

        return T.tensor(
            values,
            device=T.device('cuda:0' if T.cuda.is_available() else 'cpu') if use_device else None,
            dtype=T.float)

rospy.init_node('ddq_learning_training', anonymous=True)
node = Agent(gamma, epsilon, learning_rate, n_actions, laser_sample_count, mem_size, batch_size, actions, eps_min,
             eps_start, eps_dec, replace, max_episode_length)
rospy.spin()  # keeps python from terminating this file as long as the node is active
