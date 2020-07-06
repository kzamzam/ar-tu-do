''' 
Abstract class for all methods that are common between Q-Learning training and Policy Gradient training.
Those methods control what happens when lidar data is recieved, car crashes, car finishes an episode, etc.
Inherits from ReinforecemntLearningNode and inherited by train_q_learning. 
'''
from reinforcement_learning_node import ReinforcementLearningNode, device

from topics import TOPIC_CRASH, TOPIC_GAZEBO_MODEL_STATE, TOPIC_EPISODE_RESULT

import time
import random
import math
from collections import deque
import rospy
from std_msgs.msg import Empty
from reinforcement_learning.msg import EpisodeResult
from gazebo_msgs.msg import ModelStates

from simulation_tools.reset_car import Point
import simulation_tools.reset_car as reset_car
from simulation_tools.track import track

import torch.optim


class TrainingNode(ReinforcementLearningNode):

    def __init__(
            self,
            policy,
            actions,
            laser_sample_count,
            max_episode_length,
            learn_rate):
        ReinforcementLearningNode.__init__(self, actions, laser_sample_count)

        self.policy = policy
        self.max_episode_length = max_episode_length

        self.episode_count = 0
        self.episode_length = 0
        self.total_step_count = 0
        self.cumulative_reward = 0
        self.is_terminal_step = False

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

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learn_rate) # using the Adam optimizer for the policy NN

        reset_car.register_service()

        rospy.Subscriber(TOPIC_CRASH, Empty, self.on_crash) 
        rospy.Subscriber(TOPIC_GAZEBO_MODEL_STATE, ModelStates, self.on_model_state_callback)  # nopep8
        self.episode_result_publisher = rospy.Publisher(
            TOPIC_EPISODE_RESULT, EpisodeResult, queue_size=1)

    def on_crash(self, _):
        if self.episode_length > 5:
            self.is_terminal_step = True

    def get_episode_summary(self):
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

        if self.episode_count % 50 == 0:  # saves NN parameters ever 50 episodes
            self.policy.save()
            rospy.loginfo("Model parameters saved.")

    def on_receive_laser_scan(self, message):
        '''
        	Callback function when laser message is recieved from lidar.
        '''
        new_state = self.convert_laser_message_to_tensor(message) # convert to tensor of size depending on var. LASER_SAMPLE_COUNT.

        if self.state is not None:
            self.check_car_orientation() # making sure car is running in correct direction before giving any rewards
            reward = self.get_reward()
            if self.is_terminal_step:
                reward = -20  # In case car crashed it recieves -5
            self.cumulative_reward += reward # calc. new cumlative reward for this episode
            self.on_complete_step(self.state, self.action, reward, new_state)

        if self.is_terminal_step or self.episode_length >= self.max_episode_length: # terminal: agent crashes or takes >2 steps in worng dir
            self.drive_forward = True #random.random() > 0.5 # Probability of moving in forward or backward direction when car resets
            reset_car.reset_random(
                max_angle=math.pi / 180 * 20, # max angle deviation from proper heading of 20 degrees
                max_offset_from_center=0.2,
                forward=self.drive_forward) 
            self.is_terminal_step = False # resetting the terminal step since car will start in random new position
            self.state = None	# same as above
            if self.episode_length != 0:
                self.on_complete_episode() # this episode is done 
        else:  # will be evaluated every step of the episode
            self.state = new_state  # set new tensor data from lidar as the current state
            self.action = self.select_action(new_state)
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
            self.is_terminal_step = True

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
