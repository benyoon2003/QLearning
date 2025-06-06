import time
import pickle
import numpy as np
from vis_gym import *
import time

gui_flag = False # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

'''

Complete the function below to do the following:

	1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
	   configuration and taking actions until a terminal state is reached.
	2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
	3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
	4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

	Some important notes:
		
		- The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
		
		- To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
		  For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
		  will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

		- Your Q-table should be a dictionary with the following format:

				- Each key is a number representing the state (hashed using the provided hash() function), and each value should be an np.array
				  of length equal to the number of actions (initialized to all zeros).

				- This will allow you to look up Q(s,a) as Q_table[state][action], as well as directly use efficient numpy operators
				  when considering all actions from a given state, such as np.argmax(Q_table[state]) within your Bellman equation updates.

				- The autograder also assumes this format, so please ensure you format your code accordingly.
  
		  Please do not change this representation of the Q-table.
		
		- The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

		- Don't forget to reset the environment to the initial configuration after each episode by calling:
		  obs, reward, done, info = env.reset()

		- The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

		- The value of epsilon is initialized to 1. You are free to choose the decay rate.
		  No default value is specified for the decay rate, experiment with different values to find what works.

		- To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
		  Example usage below. This function should be called after every action.
		  if gui_flag:
		      refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {}
	Q_table_updates = {} # Keeps track of the number of updates to a state's action

	'''
	YOUR CODE HERE

	'''

	# Runs the number of episodes to find optimal policy
	for _ in range(num_episodes):
		if (_%1000 == 0):
			print(_/1000)
		
		obs = env.reset()[0]
		done = False

		# Each episode must run until done
		while not done: 
			guard_in_cell = obs['guard_in_cell']
			state = hash(obs)

			# Fill the state key's values with zeros for both the Q table and update table
			if Q_table.get(state) is None:
				Q_table[state] = np.zeros(len(env.actions))
				Q_table_updates[state] = np.zeros(len(env.actions))

			available_actions = Q_table[state]

			# Need to limit available actions depending on if the current cell has a guard
			start = 0
			end = 6
			if guard_in_cell:
				start = 4
				available_actions = available_actions[start:end]
			else:
				end = 4
				available_actions = available_actions[start:end]

			# Epsilon Greedy policy
			chosen_action = start + np.argmax(available_actions)
			if (random.random() <= epsilon):
				chosen_action = start + np.random.choice(len(available_actions))

			obs, reward, done, info = env.step(chosen_action)
			guard_in_next_cell = obs['guard_in_cell']
			next_state = hash(obs)

			# Fill state keys values with zeros if necessary
			if Q_table.get(next_state) is None:
				Q_table[next_state] = np.zeros(len(env.actions))
				Q_table_updates[next_state] = np.zeros(len(env.actions))

			available_actions_next_state = Q_table[next_state]

			# Need to limit available actions depending on if the current cell has a guard
			start = 0
			end = 6
			if guard_in_next_cell:
				start = 4
				available_actions_next_state = available_actions_next_state[start:end]
			else:
				end = 4
				available_actions_next_state = available_actions_next_state[start:end]

			prev_Q_value = Q_table[state][chosen_action]
			nu = 1 / (1 + Q_table_updates[state][chosen_action])
			max_Q_value_next_state = np.max(available_actions_next_state)

			# Updates Q table and update table
			Q_table[state][chosen_action] = ((1 - nu) * prev_Q_value) + (nu * (reward + (gamma * max_Q_value_next_state)))
			Q_table_updates[state][chosen_action] = Q_table_updates[state][chosen_action] + 1
		
		# Update epsilon with decay rate and prevent it from getting to 0
		if epsilon > 0.1:
			epsilon = epsilon * decay_rate

	return Q_table

decay_rate = 0.99
Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

Q_table = np.load('Q_table.pickle', allow_pickle=True)
for _ in range(10000):
	start = time.time()
	obs, reward, done, info = env.reset()
	total_reward = 0
	while not done:
		end = time.time()
		state = hash(obs)
		action = np.argmax(Q_table[state])
		obs, reward, done, info = env.step(action)
		total_reward += reward
		if (end - start > 5):
			print(obs)
			print(info)
			if gui_flag:
				refresh(obs, reward, done, info)  # Update the game screen [GUI only]

	print("Total reward:", total_reward)

env.close() # Close the environment


