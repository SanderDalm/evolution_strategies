import imageio
import json
import pickle
from OpenGL import GLU

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

params = pickle.load(open('params/RoboschoolHumanoid-v1_25000', 'rb'))
learner.load_params(params)


def render_episode(create_gif=True):

    episode_reward = 0
    done = False
    observation = env.reset()
    x = observation.reshape([config['input_size'], 1])
    frames = []
    while not done:
        if create_gif:
            frame = env.render(mode='rgb_array')
        else:
            frame = env.render()
        frames.append(frame)
        action = learner.model(x, learner.params)
        observation, reward, done, info = env.step(action)
        x = observation.reshape([config['input_size'], 1])
        episode_reward += reward
    print(episode_reward)
    print('Mean action: {}'.format(np.mean(np.abs(action))))

    if create_gif:
        frames = [frame for index, frame in enumerate(frames) if index % 3 == 0]
        imageio.mimsave('episode.gif', frames)

render_episode(create_gif=True)