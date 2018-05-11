import pickle
import json
from time import sleep

import numpy as np
import roboschool
import gym
import matplotlib.pyplot as plt

from es_learner import ESLearner


config = json.load(open('config/humanoid.json'))
env = gym.make(config['env_name'])


learner = ESLearner(input_dims=config['input_size'],
                    output_dims=config['output_size'],
                    hidden_size1=config['hidden_units_1'],
                    hidden_size2=config['hidden_units_2'],
                    std=config['initialization_std'],
                    output_lower_bound=config['lower_bound'],
                    output_upper_bound=config['upper_bound'],
                    sigma=config['sigma'],
                    alpha=config['alpha'],
                    pop=config['N'],
                    env=env,
                    discrete=config['discrete']
                    )

#params = pickle.load(open('params/TEST', 'rb'))
#params = pickle.load(open('params/Humanoid', 'rb'))
#learner.load_params(params)

reward_list = []

gen = 0
while True:

    gen += 1
    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))
    if gen % 1000 == 0:
        pickle.dump(params, open('params/{}_{}'.format(config['env_name'], gen), 'wb'))

plt.plot(reward_list)
plt.show()
# plt.savefig('cheetah.png')

for i in range(1):
    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([config['input_size'], 1])
    while not done:
        #env.render()
        action = learner.model(x, learner.params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([config['input_size'], 1])
        episode_reward += reward
    print(episode_reward)
print('Mean action: {}'.format(np.mean(np.abs(action))))