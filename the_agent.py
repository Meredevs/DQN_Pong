from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from agent_memory import Memory
import numpy as np
import random


class Agent():
    def __init__(self,possible_actions,starting_mem_len,max_mem_len,starting_epsilon,learn_rate, starting_lives = 5, debug = False):
        self.memory = Memory(max_mem_len)
        self.possible_actions = possible_actions
        self.epsilon = starting_epsilon
        self.epsilon_decay = .9/100000
        self.epsilon_min = .05
        self.gamma = .95
        self.learn_rate = learn_rate
        self.model = self._build_model()
        self.model_target = clone_model(self.model)
        self.total_timesteps = 0
        self.lives = starting_lives #this parameter does not apply to pong
        self.starting_mem_len = starting_mem_len
        self.learns = 0


    def _build_model(self):
        model = Sequential()
        model.add(Input((84,84,4)))
        model.add(Conv2D(filters = 32,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = 1,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Flatten())
        model.add(Dense(512,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Dense(len(self.possible_actions), activation = 'linear'))
        optimizer = Adam(self.learn_rate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        print('\nAgent Initialized\n')
        return model

    def get_action(self,state):
        """Explore"""
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions,1)[0]

        """Do Best Acton"""
        a_index = np.argmax(self.model.predict(state))
        return self.possible_actions[a_index]

    def _index_valid(self,index):
        if self.memory.done_flags[index-3] or self.memory.done_flags[index-2] or self.memory.done_flags[index-1] or self.memory.done_flags[index]:
            return False
        else:
            return True

    def learn(self,debug = False):
        """we want the output[a] to be R_(t+1) + Qmax_(t+1)."""
        """So target for taking action 1 should be [output[0], R_(t+1) + Qmax_(t+1), output[2]]"""

        """First we need 32 random valid indicies"""
        states = []
        next_states = []
        actions_taken = []
        next_rewards = []
        next_done_flags = []

        while len(states) < 32:
            index = np.random.randint(4,len(self.memory.frames) - 1)
            if self._index_valid(index):
                state = [self.memory.frames[index-3], self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index]]
                state = np.moveaxis(state,0,2)/255
                next_state = [self.memory.frames[index-2], self.memory.frames[index-1], self.memory.frames[index], self.memory.frames[index+1]]
                next_state = np.moveaxis(next_state,0,2)/255

                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index+1])
                next_done_flags.append(self.memory.done_flags[index+1])

        """Now we get the ouputs from our model, and the target model. We need this for our target in the error function"""
        labels = self.model.predict(np.array(states))
        next_state_values = self.model_target.predict(np.array(next_states))
        
        """Now we define our labels, or what the output should have been
           We want the output[action_taken] to be R_(t+1) + Qmax_(t+1) """
        for i in range(32):
            action = self.possible_actions.index(actions_taken[i])
            labels[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(next_state_values[i])

        """Train our model using the states and outputs generated"""
        self.model.fit(np.array(states),labels,batch_size = 32, epochs = 1, verbose = 0)

        """Decrease epsilon and update how many times our agent has learned"""
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.learns += 1
        
        """Every 10000 learned, copy our model weights to our target model"""
        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')
