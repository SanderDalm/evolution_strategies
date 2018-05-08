import pickle
import json

import numpy as np
import gym
import matplotlib.pyplot as plt

from es_learner import ESLearner


config = json.load(open('config/humanoid.json', 'rb'))
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

#learner.collect_batch_norm_statistics()

params = pickle.load(open('params/Humanoid-v1_5700', 'rb'))
learner.load_params(params)

reward_list = []
num_gens = config['num_generations']
for gen in range(num_gens):

    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))
    if gen % 100 == 0 or gen == num_gens-1:
        #pickle.dump((params, learner.VBN), open('params/{}_{}'.format(config['env_name'], gen), 'wb'))
        pickle.dump(params, open('params/{}_{}'.format(config['env_name'], gen), 'wb'))

learner.optimizers['w1'].rms

plt.plot(reward_list)
plt.show()
#plt.savefig('lr.01.png')
#plt.savefig('lr.001.png')


for i in range(15):
    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([config['input_size'], 1])
    while not done:
        env.render()
        #action = learner.model_VBN(x, learner.params)
        action, _, _ = learner.model_no_VBN(x, learner.params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([config['input_size'], 1])
        episode_reward += reward
    print(episode_reward)
env.close()
