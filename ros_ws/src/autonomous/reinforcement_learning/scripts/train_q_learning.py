#!/usr/bin/env python
# coding: utf-8

'''
The file launched by q_learning.launch. Launches ros node to train the deep Q-learning model.
'''

from training_node import TrainingNode, device
import random
import math
from collections import deque
from parameters_q_learning import *

import torch

import simulation_tools.reset_car as reset_car
from simulation_tools.track import track
BATCH_INDICES = torch.arange(0, BATCH_SIZE, device=device, dtype=torch.long) # creating a tensor from 0 to batch_size-1 with step 1


class QLearningTrainingNode(TrainingNode):

    def __init__(self):
        TrainingNode.__init__(
            self,
            NeuralQEstimator().to(device),
            ACTIONS,
            LASER_SAMPLE_COUNT,
            MAX_EPISODE_LENGTH,
            LEARNING_RATE)

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimization_step_count = 0

        if CONTINUE:
            self.policy.load()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:  # if replay memory has less entries than batch_size then we don't train
            return

        if self.optimization_step_count == 0:
            rospy.loginfo("Model optimization started.")

        transitions = random.sample(self.memory, BATCH_SIZE)  # get random transitions taken by agent previously with batch size from replay memory
        states, actions, rewards, next_states, is_terminal = tuple(zip(*transitions))

        states = torch.stack(states)
        actions = torch.tensor(actions, device=device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        next_states = torch.stack(next_states)
        is_terminal = torch.tensor(
            is_terminal, device=device, dtype=torch.bool)

        # This is the part where we get the target q_value(here named as q_updates) of the next_states from the replay memory
        next_state_values = self.policy.forward(next_states).max(1)[0].detach()
        q_updates = rewards + next_state_values * DISCOUNT_FACTOR  # updating the q_value since it's after a transition of the current state using th eupdate rule. It makes sense according to theory that this q_value should be same as q_value of the state before the transition.
        q_updates[is_terminal] = 0.0  # q_value of a terminal state ( crashes ) should be zero

        # This is the part where we retrain the policy NN on the loss between the target q_values and q_values which we already got from prev state
        self.optimizer.zero_grad()  # torch accumulates the gradient of past runs, so we need to make it zero
        net_output = self.policy.forward(states)
        loss = F.smooth_l1_loss(net_output[BATCH_INDICES, actions], q_updates)  # change to MSE, loss between target q_values and q_value of action we took on state
        loss.backward()  # do backward propagation to optimize weights of policy NN
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()  # updates the parameters after back propagation is run
        self.optimization_step_count += 1

    def get_epsilon_greedy_threshold(self):  # implements the epsilon decay
        return EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.total_step_count / EPS_DECAY)

    def select_action(self, state):
        use_epsilon_greedy = self.episode_count % 2 == 0  # use epsilon greedy only every other episode
        if use_epsilon_greedy and random.random() < self.get_epsilon_greedy_threshold():  # take a random action governed by epsilon value
            return random.randrange(ACTION_COUNT)

        with torch.no_grad():  # things work as in training mode, affects dropout and batch normalization, not used here anyway
            output = self.policy(state)  # policy here is the policy NN
            if self.episode_length < 10:
                self.net_output_debug_string = ", ".join(
                    ["{0:.1f}".format(v).rjust(5) for v in output.tolist()])
            return output.max(0)[1].item()  # action index that causes the max q_value

    def get_reward(self):
        '''
        method to return the reward based on location of the agent(car).
        this reward is based on the closer the agent to the center line of the track the higher the reward
        '''
        track_position = track.localize(self.car_position)
        distance = abs(track_position.distance_to_center) # distance to enter line of the track

        if distance < 0.2:
            return 1
        elif distance < 0.4:
            return 0.7
        else:
            return 0.4

    def get_episode_summary(self):
        return TrainingNode.get_episode_summary(self) + ' ' \
            + ("memory: {0:d} / {1:d}, ".format(len(self.memory), MEMORY_SIZE) if len(self.memory) < MEMORY_SIZE else "") \
            + "ε-greedy: " + str(int(self.get_epsilon_greedy_threshold() * 100)) + "% random, " \
            + "replays: " + str(self.optimization_step_count) + ", " \
            + "q: [" + self.net_output_debug_string + "], "

    def on_complete_step(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state, self.is_terminal_step))  # add necessary info to replay memory  
        self.replay() # train the policy NN 


rospy.init_node('q_learning_training', anonymous=True)
node = QLearningTrainingNode()
rospy.spin() # keeps python from terminating this file as long as the node is active
