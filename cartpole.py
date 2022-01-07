import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

"""
The environment’s step function returns exactly what we need. In fact, step returns four values. These are:

    observation (object): an environment-specific object representing your observation of the environment. 
                        For example, pixel data from a camera, joint angles and joint velocities of a robot, 
                        or the board state in a board game.
    reward (float): amount of reward achieved by the previous action. The scale varies between environments,
                     but the goal is always to increase your total reward.
    done (boolean): whether it’s time to reset the environment again. 
                    Most (but not all) tasks are divided up into well-defined episodes, 
                    and done being True indicates the episode has terminated. 
                    (For example, perhaps the pole tipped too far, or you lost your last life.)
    info (dict): diagnostic information useful for debugging. 
                    It can sometimes be useful for learning 
                    (for example, it might contain the raw probabilities behind the environment’s last state change). 
                    However, official evaluations of your agent are not allowed to use this for learning.

"""

"""
Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Action:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
"""


#Global variables

GAMMA = 0.05
LEARNING_RATE = 0.01
EXPLORATION_MIN = 0.01

BATCH_SIZE = 15



class agent_DQN():
    def __init__(self, action_space, observation_space, exploration_rate, exploration_decay):
        self.action_space = action_space
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.memory = []

        #neural network (MLP) to predict next best actions
        self.q_net = self.build_dqn_model(observation_space=observation_space,
                                           action_space=action_space, learning_rate = LEARNING_RATE)
        self.target_q_net = self.build_dqn_model(observation_space=observation_space,
                                                  action_space=action_space, learning_rate = LEARNING_RATE)

        pass

    @staticmethod
    def build_dqn_model(observation_space, action_space, learning_rate):
        """
        Builds a neural network for the agent
        :param observation_space: observation specification
        :param action_space: action specification
        :param learning_rate: learning rate
        return: neural network as a model
        """
        q_net = Sequential()
        q_net.add(Dense(128, input_dim=observation_space, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(action_space, activation='linear',
                        kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                    loss='mse')
        q_net.summary()
        return q_net

        
        

    def remember(self, state, action, reward, next_state, done):
        """
        Allows to put the relevant information of each step in the memory
        return: None
        """
        self.memory.append((state, action, reward, next_state, done))
        pass

    def take_action(self, state):
        """
        Takes an action by either trusting the neural network
        or by exploring randomly
        return: Action as an int
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        else:
            q_values = self.q_net.predict(state)
            return np.argmax(q_values[0])

    def update_target_network(self):
        """
        Updates the target Q network with the parameters
        from the currently trained Q network.
        return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())
    
    def experience_replay(self):
        """
        Allows to learn from previous experiences 
        return: None
        """
        if len(self.memory) < BATCH_SIZE:
            pass
        else:
            batch = random.sample(self.memory, BATCH_SIZE)
            network_resetter = 0
            for state, action, reward, next_state, done in batch:
                network_resetter += 1
                q_update = reward
                if not done:
                    q_update += GAMMA*np.argmax(self.target_q_net.predict(next_state)[0])
                q_values = self.q_net.predict(state)
                q_values[0][action] = q_update
                self.q_net.fit(state, q_values, verbose= 0)
                if network_resetter%5 == 0:
                    self.update_target_network()
            self.exploration_rate = max(self.exploration_rate*self.exploration_decay, EXPLORATION_MIN)

    

def play_cartpole():
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = agent_DQN(action_space, observation_space, exploration_rate = 1, exploration_decay = 0.995)
    nb_run = 0
    while nb_run < 150:
        nb_run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        done = 0
        while not done:
            step += 1
            print(step)
            # env.render()
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [1, observation_space])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print('\nRun: ' + str(nb_run) + '; \nexploration: ' + str(agent.exploration_rate) + '; \nscore: ' + str(step) )
                break
            agent.experience_replay()


if __name__ == "__main__":
    play_cartpole()




