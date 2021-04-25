import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import sys
import gym
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
 
class PG_Agent:
    def __init__(self):
        self.model = self.build_model()
        self.states, self.actions, self.rewards = [], [], []
 
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
        return model
 
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * 0.99 + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
 
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
 
    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        advantages = np.zeros((episode_length, 2))
        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i]
        self.model.fit(np.vstack(self.states), advantages, verbose=0)
        self.states, self.actions, self.rewards = [], [], []
 
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = PG_Agent()
    for e in range(1000):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 4])
        while not done:
            # env.render() if e > 600 else None
            policy = agent.model.predict(state).flatten()
            action = np.random.choice(2, 1, p=policy)[0]
            next_state, reward, done, info = env.step(action)
            agent.append_sample(state, action, reward)
            state = np.reshape(next_state, [1, 4])
            score += reward
            if done:
                agent.train_model()
                # print("episode:", e, "  score:", score)
        # if e % 50 == 0:
        #     agent.model.save_weights("./cartpole_pg.h5")
        if e % 100 == 0:
            for e in range(20):
                done = False
                score = 0
                state = env.reset()
                state = np.reshape(state, [1, 4])        
                while not done:
                    env.render() if e > 600 else None
                    policy = agent.model.predict(state).flatten()
                    action = np.argmax(policy)
                    next_state, reward, done, info = env.step(action)
                    state = np.reshape(next_state, [1, 4])
                    score += reward
                    env.render()
                print("episode:", e, "  score:", score)