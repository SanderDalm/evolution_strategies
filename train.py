import pickle
import json

import numpy as np
import gym
import matplotlib.pyplot as plt

from es_learner import ESLearner


config = json.load(open('config/cartpole.json', 'rb'))
env = gym.make(config['env_name'])

learner = ESLearner(input_dims=config['input_size'],
                    output_dims=config['output_size'],
                    hidden_size=config['hidden_units'],
                    std=config['initialization_std'],
                    output_lower_bound=config['lower_bound'],
                    output_upper_bound=config['upper_bound'],
                    sigma=config['sigma'],
                    alpha=config['alpha'],
                    pop=config['N'],
                    env=env,
                    discrete=config['discrete'],
                    use_VBN=config['VBN']
                    )

#params, VBN = pickle.load(open('params/Humanoid-v1_200', 'rb'))
#learner.load_params(params, VBN)

reward_list = []
for gen in range(config['num_generations']):

    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))
    if gen % int(config['num_generations']/10) == 0 or gen == config['num_generations']-1:
        pickle.dump((params, learner.VBN), open('params/{}_{}'.format(config['env_name'], gen), 'wb'))


plt.plot(reward_list)
plt.show()
#plt.savefig('500_gens_adam.png')

for i in range(10):
    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([config['input_size'], 1])
    while not done:
        env.render()
        action = learner.model_VBN(x, params)
        #action, _, _ = learner.model_no_VBN(x, params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([config['input_size'], 1])
        episode_reward += reward
    print(episode_reward)
env.close()