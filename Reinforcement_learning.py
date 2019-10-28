import gym
import numpy as np
import random
import matplotlib.pyplot as plt

gym.envs.register(

    id='MountainCarMyEasyVersion-v0',

    entry_point='gym.envs.classic_control:MountainCarEnv',

    max_episode_steps=100000,  # MountainCar-v0 uses 200

)

env = gym.make("MountainCarMyEasyVersion-v0")

GAMMA = 0.9

n_position_state = 50
position_state_array = np.linspace(-1.2, 0.6, num=n_position_state)

n_velocity_state = 50
velocity_state_array = np.linspace(-0.07, 0.07, num=n_velocity_state)

# Row = position, Column = velocity
# policy_matrix = np.random.randint(low=0, high=3, size=(n_position_state, n_velocity_state))

# action = policy_matrix[observation[0], observation[1]]
possible_actions = [0, 1, 2]

# Rows = state, Column = actions
q_table = np.random.rand(n_position_state, n_velocity_state, len(possible_actions))

times_action_executed = np.zeros((n_position_state, n_velocity_state, len(possible_actions)))

policy = np.zeros((n_position_state, n_velocity_state), dtype=int)
v_table = np.zeros((n_position_state, n_velocity_state), dtype=int)


def get_position_velocity_of_state(observation):
    """
        Get the position and velocity discretizated from the observation vector
        :param observation: two positions array, where position 0 has the position and position 1 the velocity
        :return: state in the q_table
        """
    # Discretize the continuous value of the observations, action and position
    # Position 0 = position, Position 1 = velocity
    obs = (np.digitize(observation[0], position_state_array) - 1,
           np.digitize(observation[1], velocity_state_array) - 1)
    position = obs[0]
    velocity = obs[1]
    return position, velocity


def update_policy_and_v_table(pos, vel):
    policy[pos, vel] = np.argmax(q_table[pos, vel])
    v_table[pos, vel] = max(q_table[pos, vel])


def backpropagate(history):
    """
    Backpropagate the q_value through the states executed. The history array has in every position the history saved
    in a tuple: [pos, vel, action, reward]
    :return:
    """
    for pos, hist in reversed(list(enumerate(history))):
        if pos == len(history) - 1:
            next_state_pos = hist[0]
            next_state_vel = hist[1]
            continue

        current_pos = hist[0]
        current_vel = hist[1]
        action = hist[2]
        reward = hist[3]
        current_state_q_value = q_table[current_pos, current_vel, action]

        learning_rate = times_action_executed[current_pos, current_vel][action] ** -0.9

        new_q_value = current_state_q_value + learning_rate * \
                      (reward + GAMMA * max(q_table[next_state_pos, next_state_vel]) - current_state_q_value)

        q_table[current_pos, current_vel, action] = new_q_value

        update_policy_and_v_table(current_pos, current_vel)

        next_state_pos = current_pos
        next_state_vel = current_vel


def q_learning(times_repeat):
    # In the beginning, this rate must be at its highest value, because we don’t know anything about
    # the values in Q-table.
    # Therefore we set it to 1 so that it is only exploration and we choose a random state

    for i in range(times_repeat):
        calculate_until_finish(False, i)

    epsilon = calculate_until_finish(True, times_repeat)

    return epsilon


def calculate_until_finish(draw, n_times):
    current_pos, current_vel, done, epsilon, history, times_action_executed, timesteps = restart_environment()

    n_max_timesteps = 1000

    while not done and timesteps < n_max_timesteps:

        timesteps += 1
        if draw:
            # env.render(file_path='./mountain_car.gif', mode='gif')
            env.render()
            action = policy[current_pos, current_vel]
        else:
            if random.random() > epsilon:  # Exploitation, choose the best action
                action = policy[current_pos, current_vel]
            else:  # Exploration, we choose a random action
                action = random.choice(possible_actions)

        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # if reward != -1:
        #     print("------------------------------------")
        # print("DONE:", reward)
        # break

        # else:
        # print("NOT DONE:", reward)
        # print("Timestep", timesteps, "execute action", action, "epsilon", epsilon, "reward", reward)

        new_pos, new_vel = get_position_velocity_of_state(observation)

        # if done:
        #     reward = 100

        current_q_value = q_table[current_pos, current_vel, action]

        # Update the counters table
        times_action_executed[current_pos, current_vel][action] += 1

        # λ = n^−α
        learning_rate = times_action_executed[current_pos, current_vel][action] ** -0.9

        # Update the q values table
        v = current_q_value + learning_rate * (reward + GAMMA * max(q_table[new_pos, new_vel]) - current_q_value)

        new_state_q_value = max(q_table[new_pos, new_vel])

        if done:
            new_state_q_value = 0

        new_q_value = current_q_value + learning_rate * \
                      (reward + GAMMA * new_state_q_value - current_q_value)

        q_table[current_pos, current_vel, action] = new_q_value

        history.append([current_pos, current_vel, action, reward])

        update_policy_and_v_table(current_pos, current_vel)
        # Add the value to the table for the convergence plot
        # q_table_all_values[current_pos, current_vel][action].append(value)

        current_pos, current_vel = new_pos, new_vel

        # epsilon = 1 / timesteps ** 0.01
        epsilon -= 1 / n_max_timesteps

        # print("epsilon", epsilon)

    # if draw:
    # else:

    if n_times % 10 == 0:
        print("Iteration", n_times, "done in", timesteps, "timesteps")
        fig, ax = plt.subplots()
        im = ax.pcolormesh(velocity_state_array, position_state_array, v_table[:, :], vmin=0, vmax=-8)
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Position")
        ax.set_title("Iteration " + str(n_times))
        fig.colorbar(im)
        plt.savefig("img/" + str(n_times) + ".png")

    # plt.show()
    # else:

    backpropagate(history)

    return epsilon


def restart_environment():
    current_pos, current_vel = get_position_velocity_of_state(env.reset())
    timesteps = 0
    epsilon = 1
    done = False
    history = []
    # times_action_executed = np.zeros((n_position_state, n_velocity_state, len(possible_actions)))
    return current_pos, current_vel, done, epsilon, history, times_action_executed, timesteps


q_learning(1000)
# plt.show()
