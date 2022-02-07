import gym
import numpy as np
import time
import matplotlib.pyplot as plt

# useful gym methods
env = gym.make('FrozenLake-v1')  # Making the FrozenLake environment
print(env.observation_space.n)   # get number of states
print(env.action_space.n)   # get number of actions
env.reset()  # reset environment to default state
action = env.action_space.sample()  # get a random action
new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
print(new_state, reward, done, info)  # prints the information the action variable has
env.render()   # render the GUI for the environment

# Solving the example
STATES = env.observation_space.n
ACTIONS = env.action_space.n

# Creating the Q-table
Q = np.zeros((STATES, ACTIONS))  # creates a matrix with all 0 values

# Defining constants
EPISODES = 2000  # how many times to run the environment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of environment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96  # discount factor
epsilon = 0.9  # start with a 90% chance of picking a random action
RENDER = True  # if we want to see training set to true if not set to false
rewards = []

for episode in range(EPISODES):

    state = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        # code to pick action
        if np.random.uniform(0, 1) < epsilon:  # we will check if a randomly selected value is less than epsilon.
            action = env.action_space.sample()  # take random action
        else:
            action = np.argmax(Q[state, :])  # until here

        next_state, reward, done, _ = env.step(action)

        # formula for updating Q-values
        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                    reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")  # and now we can see our Q values

# Plotting the training process and seeing the agents improvement
def get_average(values):
    return sum(values)/len(values)


avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i+100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()
