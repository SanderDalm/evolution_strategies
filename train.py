from EsLearner import EsLearner


import gym
import matplotlib.pyplot as plt
import pickle
env = gym.make("HalfCheetah-v1") #Humanoid-v1

INPUT_SIZE = 17 #376 cheetah
OUTPUT_SIZE = 6#17  cheetah
HIDDEN_UNITS = 256
LOWER_BOUND = -.4
UPPER_BOUND = .4
STD = .02
LR = 1

num_generations = 100
N = 100

learner = EsLearner(input_dims=INPUT_SIZE,
                    output_dims=OUTPUT_SIZE,
                    hidden_size=HIDDEN_UNITS,
                    output_lower_bound=-1,
                    output_upper_bound=1,
                    sigma=STD,
                    alpha=LR,
                    pop=N,
                    env=env,
                    discrete=False
                    )

reward_list = []
for gen in range(num_generations):

    rewards, params = learner.run_generation()
    reward_list.append(rewards)
    print('Mean reward after {} generations: {}'.format(gen, np.mean(rewards)))
    print('Mean W1: {}'.format(np.mean(np.abs(params['w1']))))
    if gen % 100 == 0 or gen == num_generations-1:
        pickle.dump(params, open('params/params_{}'.format(gen), 'wb'))

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