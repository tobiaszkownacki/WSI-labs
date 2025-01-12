import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

LEARNING_RATE = 0.25
DISCOUNT_FACTOR = 0.9


class QLearnFrozenLake:
    DELTA_EPSILON = 0.00001
    MIN_EPSILON = 0.001

    def __init__(
        self,
        num_of_episodes: int,
        num_of_steps: int,
        learning_rate: float,
        discount_factor: float,
        is_slippery: bool = False,
    ):
        self.env = gym.make(
            "FrozenLake-v1", desc=None, map_name="8x8", is_slippery=is_slippery
        )
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.q_table = np.zeros((self.state_size, self.action_size))
        self.num_of_episodes = num_of_episodes
        self.num_of_steps = num_of_steps
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def train(
        self,
        reward_function: Callable[[int, int, bool], int],
        rewards: list[int] = None,
    ) -> None:
        if rewards is None:
            rewards = np.zeros(self.num_of_episodes)

        epsilon: float = 1.0

        qtable = np.zeros((self.state_size, self.action_size))

        for episode in range(self.num_of_episodes):
            current_state, _ = self.env.reset()
            for _ in range(self.num_of_steps):
                action = self.choose_action(
                    current_state, epsilon, qtable
                )
                next_state, reward, done, truncated, _ = self.env.step(action)

                reward = reward_function(
                    reward, next_state, current_state, done
                )
                delta = (
                    reward
                    + self.discount_factor * np.max(qtable[next_state, :])
                    - qtable[current_state, action]
                )

                qtable[current_state, action] += self.learning_rate * delta
                current_state = next_state

                if done or truncated:
                    break

            epsilon = max(
                epsilon - self.DELTA_EPSILON * episode, self.MIN_EPSILON
            )
            if reward > 0:
                rewards[episode] += 1

        self.q_table = qtable
        return rewards

    def choose_action(
        self, state: int, epsilon: float, qtable: np.array
    ) -> int:
        if np.random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        max_value = np.max(qtable[state, :])
        best_actions = np.where(qtable[state, :] == max_value)[0]
        return np.random.choice(best_actions)

    @staticmethod
    def reward_default(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        return reward

    @staticmethod
    def reward_alternative_1(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        # If the agent falls into a hole, the reward is -1
        # if the agent reaches the goal, the reward is 1
        if done and reward == 0:
            return -1
        elif done and reward == 1:
            return 10
        return 0

    @staticmethod
    def reward_alternative_2(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        # If the agent falls into a hole, the reward is -5
        # if the agent reaches the goal, the reward is 20
        # if the agent stays in the same state, the reward is -1
        if done and reward == 0:
            return -5
        elif done and reward == 1:
            return 20
        elif next_state == current_state:
            return -1
        return 0


def compare_average_rewards(
        episodes: int,
        steps: int,
        num_of_id_runs: int,
        reward_function: Callable[[int, int, int, bool], int],
        is_slippery: bool = False
        ) -> np.array:
    average_rewards = np.zeros(episodes)
    for _ in range(num_of_id_runs):
        qlearn = QLearnFrozenLake(
            episodes, steps, LEARNING_RATE, DISCOUNT_FACTOR, is_slippery
            )
        qlearn.train(reward_function, average_rewards)
    return average_rewards / num_of_id_runs


def plot_results(averaged_reward_base: np.array, averaged_reward: np.array, max_episodes: int = 1000) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    averaged_reward_base = np.convolve(
        averaged_reward_base, np.ones(50) / 50, 'valid'
    )
    averaged_reward = np.convolve(
        averaged_reward, np.ones(50) / 50, 'valid'
    )
    plt.xlim(0, max_episodes)
    plt.plot(averaged_reward_base, 'r', label='Default reward function')
    plt.plot(averaged_reward, 'b', label='Alternative reward function')
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig("results.png")


def main():
    num_of_id_runs = 25
    num_of_episodes = 1000
    num_of_steps = 200

    averaged_reward_base = compare_average_rewards(
        num_of_episodes,
        num_of_steps,
        num_of_id_runs,
        QLearnFrozenLake.reward_default,
        True
        )
    averaged_reward = compare_average_rewards(
        num_of_episodes,
        num_of_steps,
        num_of_id_runs,
        QLearnFrozenLake.reward_alternative_2,
        True
        )

    plot_results(averaged_reward_base, averaged_reward, num_of_episodes)


if __name__ == "__main__":
    main()
