# Deep Reinforcement Learning with PONG

This project was greatly based off of Akshay Srivatsan’s project on Deep Learning in Video Games.  The pygame_player and pong_player python scripts were also created by Daniel Slater.  Both of their projects are linked below:

Akshay Srivatsan : https://github.com/asrivat1/DeepLearningVideoGames
Daniel Slater: https://github.com/DanielSlater/PyGamePlayer


## Introduction to Reinforcement Learning

Reinforcement learning is a form of machine learning in which the ‘learner’ must discover which actions yield the most reward by trying them [1].  The process is meant to develop optimal patterns in the behavior of the model by providing feedback on the model’s decisions.  The feedback prompts the learner to make better decisions in the future [2].  Essentially, what I have done is found a way to make the learner interact with the game, and extract the necessary feedback required to train a convolutional neural network to predict the best action for the learner to take in a specific state.

The important elements of a reinforcement system include: a policy, a reward signal, and an action-value (Q-value) function.  A policy defines or controls the way the agent behaves in a certain state.  The reward signal defines the goal, and the model’s objective is to maximize the reward signal over time based on its decisions.  The value function is a measure of the overall expected reward assuming the agent is in a specific state and performs a specific action for an episode while adhering to the policy.  

Q-Learning is a reinforcement learning algorithm which uses a table to store Q-values of all possible state and action pairs in order to teach the agent what the best action to take is in each state. The Bellman Equation is used to update the table:

Q^* (s,a)=r(s,a)+γmax⁡Q^* (s^',a^')

It defines the relationship between a state-action pair to the future state-actions pairs.  Q^* (s,a) was defined above as the Q-value function, which is the measure of overall expected reward given a current state, along with an action over an episode, following a policy.  ‘r’ is the reward, s is the state, a is the action, and s’ and a’ are the future states and actions.  Gamma () is the discount factor and controls the importance of future rewards vs immediate ones [3].  A larger gamma parameter will result in the agent prioritizing maximizing future rewards more than immediate ones.  In my specific case this is important because it may take the player agent in PONG many actions to yield a positive reward.  

In the case of Deep Q-Learning the table is replaced with a neural network in order to deal with large amounts of environmental data.  PONG’s original game environment consists of 640x480 pixels.  In order to decrease the computation time and power necessary I will resize the image to 84x84 (7056) pixels. 
Every line in my code is commented out to explain what each parameter, function, class, method is doing, but before we get to the code there are some other important terms to define and explain.

## Exploration vs Exploitation

One of the reasons reinforcement learning is a great machine learning technique is that the system you are controlling or training on provides the data for the agent in the form of live and immediate feedback.  The agent creates its own experience [3].  This is significant because we have to make sure that the agent is able to explore, ideally, every single possible action in every single possible state.  If it can do this then it will be able to learn the outcome of every action in every state and find the optimal path to maximize the overall reward.  The more the agent can explore, the more prepared it is to choose the optimal action in the future.  However, there is a balance to be achieved here.  The agent must explore as much as it can, but eventually it needs to start using the information it learned in the past to repeat the most rewarding action path it found.  Essentially, we allow the agent to explore by taking random actions in as many states as possible, store the observations and feedback from those state-action pairs, and exploit what it learned to decide what the most rewarding sequence of actions is. 
In order to achieve the balance between exploration and exploitation we can use an epsilon-greedy policy.  The parameter epsilon is set to a number between 0 and 1, and when the next action is being selected a random number between 0 and 1 is generated.  If that random number is greater than epsilon, then the agent uses the trained neural network to make the next decision.  If the random number is lower than epsilon, then a random action is selected to encourage exploration.  The epsilon parameter is decayed over a set number of exploration steps so that there is a long period of mostly exploration and a gradual transition into mostly or only exploitation.  

## Monte Carlo Methods

I will not go too deep into Monte Carlo methods, but for this reinforcement learning case it is necessary.  The agent needs to take samples of its memory in minibatches in order to record expected rewards from certain state-action pairs.  Monte Carlo methods are used often in reinforcement learning algorithms to obtain expected values.  They are methods which use repeated random sampling in order to achieve a result [3].  We make sure that each sample we take is random, so the entire scope of agent memory is covered.

## Object Oriented Programming

During my research into deep reinforcement learning with python I found that most projects used this pattern of programming where the agent is a collection of collaborating objects.  The agent is a class containing various methods and functions I would call behaviours.  These set behaviours within the agent class were used to preprocess data, determine the next action, create the neural network, train the neural network, and translate action arrays into actual key-presses within the game environment.  The agent class also inherits from a couple other parent classes.  Specifically, the PyGamePlayer and PongPlayer classes. These classes were written by Daniel Slater, who’s GitHub project on Deep Q-Learning I drew valuable information from.  They allow us to interact with the PyGame environment without having to touch the game mode.  The classes help make implementing a learning agent simpler.
	

[1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.

[2] https://github.com/asrivat1/DeepLearningVideoGames

[3] Zychlinski, S., & Zychlinski, S. (2019, February 23). The Complete Reinforcement Learning Dictionary. Retrieved from https://towardsdatascience.com/the-complete-reinforcement-learning-dictionary-e16230b7d24e
