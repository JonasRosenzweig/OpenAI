
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input
import random
import matplotlib.pyplot as plt
from collections import deque
import time
from keras.activations import relu, linear
from keras.optimizers import Adam
from keras.losses import mean_squared_error
import gym


class DQNAgent():
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=50000)
        self.action_space_dim = 4
        self.observation_space_dim = 8
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 64
        self.test_episode_rewards = []
        self.test_average_rewards = []
        self.training_episode_rewards = []
        self.training_average_rewards = []
        self.replay_counter = 0
        self.training_episodes = 2000
        self.training_frame_count = 0
        
        self.model = self.initialize_model()
        
        
    def initialize_model(self):
            inputs = Input(shape=(8,))
            dense = Dense(512, activation=relu)
            x = dense(inputs)
            x = Dense(256, activation = relu)(x)
            outputs = layers.Dense(4, activation = linear)(x)
            model = keras.Model(inputs = inputs, outputs = outputs, name = "1Dense")
            model.compile(loss = mean_squared_error, optimizer = Adam(lr=0.001)) 
            model.summary()
            return model
            
    def get_action(self, state):
            if np.random.rand() < self.epsilon:
                return random.randrange(self.action_space_dim)
            predicted_actions = self.model.predict(state)
            return np.argmax(predicted_actions[0])
        
    def add_to_memory(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
        
    def sample_from_memory(self):
            sample = random.sample(self.memory, self.batch_size)
            return sample
        
    def extract_from_sample(self, sample):
            states = np.array([i[0] for i in sample])
            actions = np.array([i[1] for i in sample])
            rewards = np.array([i[2] for i in sample])
            next_states = np.array([i[3] for i in sample])
            done_list = np.array([i[4] for i in sample])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            return np.squeeze(states), actions, rewards, next_states, done_list
       
    def learn_from_memory(self):
            if len(self.memory) < self.batch_size or self.replay_counter != 0:
                return
            if np.mean(self.training_episode_rewards[-10:]) > 100:
                return
            sample = self.sample_from_memory()
            states, actions, rewards, next_states, done_list = self.extract_from_sample(sample)
            targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
            target_vec = self.model.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])
            target_vec[[indexes], [actions]] = targets
            self.model.fit(states, target_vec, epochs=1, verbose=0)
                   
    def train(self):
            start_time = time.time()
            for episode in range(self.training_episodes):
                steps = 1000
                state = self.env.reset()
                episode_reward = 0
                state = np.reshape(state, [1, 8])
                episode_frame_count = 0
                
                for step in range(steps):
                    
                    #self.env.render()
                    exploit_action = self.get_action(state)
                    next_state, reward, done, info = self.env.step(exploit_action)
                    next_state = np.reshape(next_state, [1, 8])
                    episode_reward += reward
                    self.training_frame_count += 1
                    episode_frame_count += 1
                    self.add_to_memory(state, exploit_action, reward, next_state, done)
                    state = next_state
                    self.update_counter()
                    self.learn_from_memory()
                    if done: 
                        break
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                average_reward = np.mean(self.training_episode_rewards[-100:])
                if average_reward > 200:
                    break
                if average_reward < -400 and episode > 100:
                    break
                if average_reward < -300 and episode > 200:
                    break
                if average_reward < -200 and episode > 300:
                    break
                train_time_minutes = (time.time() - start_time)/60
                if train_time_minutes > 30:
                    break
                self.training_episode_rewards.append(episode_reward)
                self.training_average_rewards.append(average_reward)
                print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}
    Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}\t|| Epsilon: {:.2f}
    Frames this episode: {}\t\t|| Total Frames trained: {}\n"""
                    .format(episode, episode_reward, reward, average_reward, self.epsilon, episode_frame_count, self.training_frame_count))
            self.env.close()
            
    def save(self, name):
            self.model.save(name)
            
    def update_counter(self):
            self.replay_counter += 1
            step_size = 5
            self.replay_counter = self.replay_counter % step_size     
            
    def test_trained_model(self, trained_model):
            for episode in range(self.testing_episodes):
                steps = self.frames
                trained_state = self.env.reset()
                episode_reward = 0
                observation_space_dim = self.env.observation_space.shape[0]
                trained_state = np.reshape(trained_state, [1, observation_space_dim])
                for step in range(steps):
                
                    # self.env.render()
                    trained_action = np.argmax(trained_model.predict(trained_state)[0])
                    next_state, reward, done, info = self.env.step(trained_action)
                    next_state = np.reshape(next_state, [1, observation_space_dim])
                    trained_state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                average_reward_trained = np.mean(self.trained_rewards[-100:])
                self.test_episode_rewards.append(episode_reward)
                self.test_average_rewards.append(average_reward_trained)
                
                
                print("""Episode: {}\t\t\t|| Episode Reward: {:.2f}\
    Last Frame Reward: {:.2f}\t|| Average Reward: {:.2f}"""
                  .format(episode, episode_reward, reward, average_reward_trained))
                    
            self.env.close() 
                       
                        
        
        
if __name__ == '__main__':
    
    env = gym.make('LunarLander-v2')
    env.seed(21)
    np.random.seed(21)
    
    model = DQNAgent(env)
    model.train()
    
    
                
        
        
        
