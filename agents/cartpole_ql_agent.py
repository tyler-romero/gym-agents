import gym
from gym import wrappers
import qlearning as qla
import collections

def CartPoleFeaturExtractor(state, action):
    #Round so that features can generalize between states slightly
    state = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, state))

    featureVector = []
    featureVector.append((state, 1))
    featureVector.append((("f0", state[0], action), 1))
    featureVector.append((("f1", state[1], action), 1))
    featureVector.append((("f2", state[2], action), 1))
    featureVector.append((("f3", state[3], action), 1))
    
    return featureVector

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, 'C:\\Users\\Tyler\\Downloads\\tmp\\cart-pole')

ql = qla.QLearningAlgorithm(env.action_space, discount=0.9, featureExtractor=CartPoleFeaturExtractor, explorationProb=0.02)
record = collections.deque(100*[0], 100)

observation = env.reset()
reward = 0
done = False
for i_episode in range(1000):
    observation = env.reset()
    for t in range(200):
        env.render()

        # Your Code Here #
        action = ql.getAction(tuple(observation))
        oldObs = observation
        # End Your Code #

        observation, reward, done, info = env.step(action)

         # Your Code Here #
        true_reward = -1 if done is True and t < 199 else 0 # Negative reward if agent fails, zero reward else
        ql.incorporateFeedback(tuple(oldObs), action, true_reward, tuple(observation))
        # End Your Code #

        if done:    #Print relevant info and start new episode
            record.appendleft(t+1)
            print "Episode finished after {} timesteps".format(t+1)
            if i_episode % 10 is 0:
                print "\tRolling Ave: ", float(sum(record)) / max(len(record), 1)
            break