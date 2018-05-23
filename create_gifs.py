import imageio
import json
import pickle
from OpenGL import GLU
import matplotlib.pyplot as plt

import numpy as np
import gym
import roboschool

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

params = pickle.load(open('params/CartPole-v1_40', 'rb'))
#params = pickle.load(open('params/RoboschoolHumanoid-v1_61000', 'rb'))
learner.load_params(params)

def render_episode(mode=None, filename=None):

    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([config['input_size'], 1])
    frames = []

    while not done:

        if mode == 'create_gif':
            frame = env.render(mode='rgb_array')
            frames.append(frame)
        if mode == 'render':
            frame = env.render()
            frames.append(frame)

        action = learner.model(x, learner.params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([config['input_size'], 1])
        episode_reward += reward

    if mode == 'create_gif':
        frames = [frame for index, frame in enumerate(frames) if index % 5 == 0]
        imageio.mimsave(filename, frames)

    #plt.imshow(frames[0])
    #plt.savefig('init1.png')
    print(episode_reward)
    return episode_reward

score = render_episode('create_gif', 'cartpole_40.gif')

# scores = []
# for i in range(100):
#     score = render_episode()
#     print(i, score)
#     scores.append(score)
# print('Mean score:')
# print(np.mean(scores))