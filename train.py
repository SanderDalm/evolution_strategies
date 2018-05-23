import pickle
import json
from time import sleep

import numpy as np
import roboschool
import gym
import matplotlib.pyplot as plt
from OpenGL import GLU

from es_learner import ESLearner


config = json.load(open('config/cheetah.json'))
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
                    discrete=config['discrete'],
                    optimizer=config['optimizer']
                    )

#params = pickle.load(open('params/thuis_21mei', 'rb'))
#learner.load_params(params)

reward_list = []
gen = 0
while True:

    gen += 1
    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))

    if gen % 10 == 0:
        avg_rewards = [np.mean(reward_list[index - 100:index]) if index >= 100 else 0 for index, _ in
                       enumerate(reward_list)]
        avg_rewards = avg_rewards[100:]
        plt.plot(avg_rewards)
        plt.savefig('progress/{}_progress_{}.png'.format(config['env_name'], gen))
        plt.clf()

    if gen % 10 == 0:
        pickle.dump(params, open('params/{}_{}'.format(config['env_name'], gen), 'wb'))

#pickle.dump(params, open('thuis_21mei', 'wb'))
plt.plot(reward_list)
plt.show()