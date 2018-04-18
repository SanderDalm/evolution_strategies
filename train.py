import pickle
import json

import numpy as np
import gym
import matplotlib.pyplot as plt

from es_learner import EsLearner


config = json.load(open('config/humanoid.json', 'rb'))
env = gym.make(config['env_name'])

learner = EsLearner(input_dims=config['input_size'],
                    output_dims=config['output_size'],
                    hidden_size=config['hidden_units'],
                    output_lower_bound=config['lower_bound'],
                    output_upper_bound=config['upper_bound'],
                    sigma=config['sigma'],
                    alpha=config['alpha'],
                    pop=config['N'],
                    env=env,
                    discrete=config['discrete']
                    )

reward_list = []
for gen in range(config['num_generations']):

    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))
    if gen % 100 == 0 or gen == config['num_generations']-1:
        pickle.dump(params, open('params/params_{}_{}'.format(config['env_name'], gen), 'wb'))

plt.plot(reward_list)

for i in range(10):
    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([INPUT_SIZE, 1])
    while not done:
        env.render()
        action = learner.model(x, params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([INPUT_SIZE, 1])
        episode_reward += reward
    print(reward)
env.close()