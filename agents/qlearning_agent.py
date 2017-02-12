import gym
from gym import wrappers
import numpy as np
import qlearning as ql
import collections

def CartPoleFeaturExtractor(state, action):
    apos, avel, xpos, xvel = state
    apos = round(apos, 3)
    avel = round(avel, 2)
    xpos = round(xpos, 3)
    xvel = round(xvel, 2)
    encoding = (xpos, xvel, apos, avel, action)

    featureVector = []
    indicate = 1
    featureVector.append((encoding, indicate))
    featureVector.append((("apos", apos, action), indicate))
    featureVector.append((("avel", avel, action), indicate))
    featureVector.append((("xpos", xpos, action), indicate))
    featureVector.append((("xvel", xvel, action), indicate))
    
    return featureVector


class QlearningAgent(object):
    def __init__(self, action_space, fe):
        self.action_space = action_space
        self.qlearn = ql.QLearningAlgorithm(action_space, discount=0.9, featureExtractor=fe, explorationProb=0.02)

    def act(self, observation, reward, done):
        state = tuple(observation)
        return self.qlearn.getAction(state)

    def learn(self, oldObs, action, observation, reward, done):
        state = tuple(oldObs)
        newState = tuple(observation)
        self.qlearn.incorporateFeedback(state, action, reward, newState)


env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, 'C:\\Users\\Tyler\\Downloads\\tmp\\cart-pole')
qa = QlearningAgent(env.action_space, CartPoleFeaturExtractor)

observation = env.reset()
reward = 0
done = False

record = collections.deque(100*[0], 100)
for i_episode in range(10000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        oldObs = observation

        action = qa.act(observation, reward, done)
        observation, reward, done, info = env.step(action)
        true_reward = -1 if done is True and t < 199 else 0
        action = qa.learn(oldObs, action, observation, true_reward, done)

        if done:
            record.appendleft(t+1)
            print "Episode finished after {} timesteps".format(t+1)
            if i_episode % 10 is 0:
                print "\tRolling Ave: ", float(sum(record)) / max(len(record), 1)
            break