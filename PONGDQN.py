''' 

The methods in the code are based on Danielslater at https://github.com/DanielSlater/PyGamePlayer 
and Akshay Srivatsan's work at https://github.com/asrivat1/DeepLearningVideoGames

'''
import os
import random
from collections import deque
from pong_player import PongPlayer
from pygame_player import PyGamePlayer
import tensorflow as tf
import numpy as np
import cv2
from pygame.constants import K_DOWN, K_UP

class DQNAgent(PongPlayer):
    
    ''' 
    This class is used to define the methods of the q-learning agent.  
    It inherits from the PongPlayer class and forwards requests to that praent class.
    The Pongplayer class is also a descendant class of the PyGamePlayer class.
    The PyGamePlayer and PongPlayer classes were created by Daniel Slater.
    https://github.com/DanielSlater/PyGamePlayer
    
    These classes allow for interaction between a learning agent and a game environment 
    without the need to interact directly with the PyGame code.
    
    The PyGamePlayer class is important because it extracts observations about the game state and environment.
    
    '''
    
    # We have to define some parameters for the Q-Learning Algorithim

    resized_state_x, resized_state_y = (84, 84) # Pixel resolution of the grame screen (environment state dimension)

    state_frames = 4 # number of frames to store in one state

    action_size = 3 # How large is the action space of the agent? PONG: up, down, still

    gamma = 0.95  # how much to discount future rewards
                  # weight upcomming actions more heavily than ones in the distant future

    epsilon_initial = 1.0 # Initial probability that the agent will choose random action
                          # Encourages initial exploration and decreases over time to favour exploitation

    epsilon_min = 0.01 # Final probability that the agent will choose random action
    
    learning_rate = 1e-6 # stochastic gradient descent optimizer step size

    explore_steps = 100000 # frames over which to anneal epsilon

    minibatch_size = 32 # minibatch which is a random sample from memory
                        # observations from these batches will be used to train the NN

    memory = 50000 # How many observations do we want to store in memory to be sampled? (memory size)

    observation_steps = 40000 # How many actions the agents takes before q-learning algorithim kicks in 

    savepoints = 10000 # Save neural network weights every _ steps
    
    update_time = 10000 # In simple Q-Learning a state-action table would be updated fequently 
                        # with Q-values.
                        # In this case the table is the neural network and the agent decides 
                        # the next action using the networkm therefore we have to update the
                        # network based on more action-states and observations from memory
            
    def __init__(self, checkpoint_path="PONG_NN_Weights_FINAL", playback_mode=False, verbose_logging=False):

        # set the first action to be - do nothing
        # this is explained further in the get_keys_pressed function
        self.prev_action = np.zeros(self.action_size)
        self.prev_action[1] = 1

        # Make the previous state empty to be filled in later
        self.prev_state = None

        # for output logging
        self.time = 0

        self.epsilon = self.epsilon_initial # self.epsilon will be decayed over time to encourage exploration intitally and exploitation finally

        self.observations = deque() # This deque will store all of the observation information at each step
                                    # previous state, previous action, reward, next state, terminal 
                                    # terminal = True if we have reached a terminal state, meaning the next frame will be a restart
                                    # In the case of Pong this terminal state will always = False

        self.playback_mode = playback_mode # set playback_mode = False to train the Agent
                                           # set playback_mode = True and the Agent will play the game with saved NN weights
                                           # playback_mode = False by default

        self.verbose_logging = verbose_logging # Keep track of the Q-Value output from the neural network

        # Super is a shortcut to access the parent class (classes) without giving a name
        # We force the game to run at 8 frames per second and define the playback mode setting
        # This calls methods from the PyGamePlayer class script
        super(DQNAgent, self).__init__(force_game_fps=8, run_real_time=playback_mode)

        # initialize the Action Q Network
        self.input_layer,self.output_layer,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = DQNAgent.create_network()

        # initialize Target Q Network
        self.input_layerT,self.output_layerT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = DQNAgent.create_network()

        # Used to update the neural network weights and biases
        # see the train method and copy target network method
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        
        # Set action and target placeholders
        # used to train the agent to associate certain actions in specific states lead to specific rewards
        self.action = tf.placeholder("float", [None, self.action_size]) #actioninput
        self.target = tf.placeholder("float", [None]) #yinput

        # Minimize the cost of the neural network while training
        Q_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.target - Q_action))
        self.train_agent = tf.train.RMSPropOptimizer(1e-6, decay=0.99, momentum=0.0, epsilon=1e-10).minimize(self.cost)        

        # Save neural network weights to determine Agent's next action
        self.checkpoint_path = checkpoint_path
        
        # self.session = tf.Session() # Instantiate the tensorflow session 
        self.session = tf.InteractiveSession()
        
        # Every used variable in the network needs to be initialized
        self.session.run(tf.global_variables_initializer())
        
        # Save neural netowrk weights for playback
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path) # output information
        else:
                print("Could not find old network weights") # output information

    def get_keys_pressed(self, screen_array, reward, terminal):
        
        '''
        We use the method to:

        Get and preprocess the frames/images from the game. 
         - This includes, converting the image to grayscale from RGB
         - Resizing the images from 640 x 480 to a more managable size
           for the neural network computation 80 x 80 
         - Also stacking the last 4 frames to provide the agent with more 
           information about the movement of the ball on the Pong screen

        Collect information and store in memory.
         - Store previous observations in memory to be accessed when training
         - This is done so that the agent understands what behavious/actions
           yield a reward of 1 (good) or a reward of -1 (bad)

        Decide when to start training and what actions to for every next state
         - If the length of the observation deque exceeds the set memory size
           then start deleting values at the start of the deque to keep it at
           the set memory size.
         - We se tthe number of observation steps above, so once the length of the 
           observation deque exceeds that length the observation phase is over and 
           the model begins to train
         - The next state is updated to be the previous state and we use the previous action 
           to determine the next action using the choose_next_action and key_presses_from_action
           methdods
    
        '''
        
        # preprocess image data to be fed into neural network to grayscale
        # resize image to 84x84 (7056 pixels)
        screen_resized_grayscaled = cv2.cvtColor(cv2.resize(screen_array, (self.resized_state_x, self.resized_state_y)),cv2.COLOR_BGR2GRAY)
        
        # set the pixels to all be 0 or 1 
        threshold, screen_resized_binary = cv2.threshold(screen_resized_grayscaled, 1, 255, cv2.THRESH_BINARY)
    
        # first frame must be handled differently
        # we specify an empty previous state to start
        if self.prev_state is None:
            # the previous state will contain the image data from the previous 4 frames as set above by self.state_frames 
            self.prev_state = np.stack(tuple(screen_resized_binary for _ in range(self.state_frames)), axis=2)
            return DQNAgent.key_presses_from_action(self.prev_action)
        
        # *****
        screen_resized_binary = np.reshape(screen_resized_binary, (self.resized_state_x, self.resized_state_y, 1))
        next_state = np.append(self.prev_state[:, :, 1:], screen_resized_binary, axis=2)
        
        if not self.playback_mode:
            # store the transition in previous_observations
            self.observations.append((self.prev_state, self.prev_action, reward, next_state, terminal))
            # if the length of the observations deque grows to be larger than the set memory size
            # then start letting go of the oldest observations and xontinue to add the newst ones
            if len(self.observations) > self.memory:
                self.observations.popleft()

            # If the length of the obersavtion deque exceeds the set amount of observation steps
            # then it is time to begin the training process/ Q-value optimization
            # and the time counter starts counting
            if len(self.observations) > self.observation_steps:
                self.train()
                self.time += 1
                
        # update the state
        self.prev_state = next_state
        
        # Use the choose_next_action method to update the next action
        self.prev_action = self.choose_next_action()

        if not self.playback_mode:
            # gradually reduce the probability of a random action
            
            # this is controlled by the number of explore steps
            
            # the number of observation steps controls for how long the agent 
            # chooses random actions while observing the environment
            
            if self.epsilon > self.epsilon_min and len(self.observations) > self.observation_steps:
            # so if epsilon is 1 the agent will always choose a random 
            # action from the action set
            # at the same time, if the observation deque becomes longer 
            # than the number of set observation steps 
            #then we reduce the probability of random action using:
                self.epsilon -= (self.epsilon_initial - self.epsilon_min) / self.explore_steps

            print("Time: %s epsilon: %s reward %s" % (self.time, self.epsilon, reward))
                  
        # prev_action is = choose_next_action    
        # we get the Key Press returned from this method
        # UP, DOWN, LEFT OR RIGHT
        return DQNAgent.key_presses_from_action(self.prev_action)                                      
                  
    def choose_next_action(self):    

        '''
        The agent uses this to decide what next action to take.  This is controlled by epsilon.  
        A random number is selcted between 0 and 1.  If it is less than epsilon at the time, 
        then a random action is chosen.  If it is greater than epsilon then we use the 
        neural network to predict the next best action bast on the state.
        '''       
        # the action input to the game is an array of 3 numbers
        # [1,0,0] input results in no action by the agent
        # [0,1,0] input results in one key press for UP
        # [0,0,1] input results in one key press for DOWN
                  
        # so the new action for the agent to take, determined by the neural network
        # is going to be an array [x,x,x]
                  
        # so set the array to be an array of zeroes, to be adjusted 
        # when the new action index is determined by the neural network
        next_action = np.zeros(self.action_size)
        action_index = 0
        
        # if epsilon < a random number, choose random action
        if (not self.playback_mode) and (random.random() <= self.epsilon):
            # choose an action randomly
            action_index = random.randrange(self.action_size)
            
        else:
            # choose an action useing neural network prediction
            Q_values = self.output_layer.eval(feed_dict= {self.input_layer:[self.prev_state]})[0]

            # everytime, the neural network makes an action prediction print the q-values          
            if self.verbose_logging:
                print("Action Q-Values are %s" % Q_values)
            action_index = np.argmax(Q_values)
            
        # use the outout array of the neural network (Q-values) 
        # to set the correct action input in the
        next_action[action_index] = 1
        return next_action               
 
    # Used to update the network weights for exploitation of Q-value function
    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)
        
    def train(self):

        '''
        This method is used to train the agent to associate state-action pairs with positive or negative rewards.
        We take a random sample from meomory (a mini batch), calculate the possible actions the agent could take 
        in the next state, and then calculate the expected reward using the Bellman Equation.
        Those expcted rewards are then used to train the agent to associate certain actions with better rewards

        '''
        # sample a mini_batch to train on
        mini_batch = random.sample(self.observations, self.minibatch_size)
        
        # the self.observations deque holds arrays of the observations of the game screen 
        # prev_state, last_action, reward, next_state, terminal
        # [   0            1          2         3         4     ]
        # the mini_batch randomly samples the memory to get  
        prev_states = [d[0] for d in mini_batch]
        actions = [d[1] for d in mini_batch]
        rewards = [d[2] for d in mini_batch]
        next_states = [d[3] for d in mini_batch]
        
        agents_expected_reward = []
        # this gives us the agents expected reward for each action the agent might take
        
        # this loop does the following:
        # first, we feed the next possible states into the neural network and a set of possible actions are returned
        # these are Q_values or possible actions 
        possible_actions = self.output_layerT.eval(feed_dict={self.input_layerT: next_states})
   
        # Second, we determine if the respective state is the last or terminal state 
        # (determined by the get_key_presses method) in the PyGamePlayer parent class
        # If it is then the expected reward is simply from the same observation array 
        # as the respective possible action
        
        for m in range(len(mini_batch)):
            if mini_batch[m][4]:
                agents_expected_reward.append(rewards[m])
                
            # If the state we are in is not a terminal state then we calculate the 
            # reward at current time step + discounted future reward
            # This is the Q-Value function
            # Instead of update a table of Q-values with state-action pairs 
            # it will be used to train the Neural Network
            else:
                agents_expected_reward.append(rewards[m] + self.gamma * np.max(possible_actions[m]))
        
        # train the agent to associate these actions in these states lead to this reward
        # self.action and self.target were set as tensorflow placeholders above
        # this is the equilevent of updating the Q-Value table in no-Deep Q-Learning
        self.train_agent.run(feed_dict={self.target : agents_expected_reward, self.action : actions, self.input_layer : prev_states})
        
        # save neural network checkpoints
        if self.time % self.savepoints == 0:
            self.saver.save(self.session, self.checkpoint_path + '/network', global_step=self.time)
            
        # update neural netowrk weights and biases for more accurate predictions
        if self.time % self.update_time == 0:
            self.copyTargetQNetwork()
            
    
    def create_network():
        
        ''' 
        Now we can develop the neural network we will use to determine what action the agent should take given a set of 4 states
        This neural network was written using Tensorflow, by https://github.com/songrotek/DQN-Atari-Tensorflow.git.
        
        I am removing the max-pooling layers, as Akshay Srivatsan noted that the layer may have 
        discarded useful information about the environment state (image), that the agent might find informative.
        
        There are hidden convolutional layers.
        The input is of size 84,84,4 (the state as a stack of 4 images)
        The output is the shape of the total number of possible actions
        
        '''
        # Weight and bias variables
        W_conv1 = DQNAgent.weight_variable([8,8,4,32])
        b_conv1 = DQNAgent.bias_variable([32])

        W_conv2 = DQNAgent.weight_variable([4,4,32,64])
        b_conv2 = DQNAgent.bias_variable([64])

        W_conv3 = DQNAgent.weight_variable([3,3,64,64])
        b_conv3 = DQNAgent.bias_variable([64])

        W_fc1 = DQNAgent.weight_variable([3136,512])
        b_fc1 = DQNAgent.bias_variable([512])

        W_fc2 = DQNAgent.weight_variable([512,DQNAgent.action_size])
        b_fc2 = DQNAgent.bias_variable([DQNAgent.action_size])

        # input layer tensor
        input_layer = tf.placeholder("float", [None, DQNAgent.resized_state_x, DQNAgent.resized_state_y, DQNAgent.state_frames])

        # hidden layers
        h_conv1 = tf.nn.relu(DQNAgent.conv2d(input_layer,W_conv1,4) + b_conv1)
        h_conv2 = tf.nn.relu(DQNAgent.conv2d(h_conv1,W_conv2,2) + b_conv2)
        h_conv3 = tf.nn.relu(DQNAgent.conv2d(h_conv2,W_conv3,1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
    
        # Q Value layer
        output_layer = tf.matmul(h_fc1,W_fc2) + b_fc2
        
        # Return the input layer and putput layer to be fed into the network
        return input_layer,output_layer,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    # define the shape and tensor for the weight variables
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    # define the shape and tensor for the weight variables
    def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
    
    # define the input_layer, weight variable, and tensor for the convolutional layers
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

                    
    def key_presses_from_action(action_set):
        
        '''
        This function translates the action_set to actual key presses in PyGame 

        '''
        if action_set[0] == 1:
            return [K_DOWN]
        elif action_set[1] == 1:
            return []
        elif action_set[2] == 1:
            return [K_UP]
        raise Exception("Unexpected action")

        
if __name__ == '__main__':
    player = DQNAgent()
    
    # call the start method in the parent classes
    player.start()
    
    # importing pong will start the game playing  
    import pong