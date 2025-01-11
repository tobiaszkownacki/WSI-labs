import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class QLearnFrozenLake:
    DELTA_EPSILON = 0.00001
    MIN_EPSILON = 0.1

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
        # if the agent reaches the goal, the reward is 100
        if done and reward == 0:
            return -1
        elif done and reward == 1:
            return 100
        return 0

    @staticmethod
    def reward_alternative_2(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        # If the agent falls into a hole, the reward is -5
        # if the agent reaches the goal, the reward is 100
        # if the agent hits the wall, the reward is -1
        if done and reward == 0:
            return -5
        elif done and reward == 1:
            return 100
        elif next_state == current_state:
            return -1
        return 0

    def evaluate_qtable(
            self, num_of_episodes: int, num_of_steps: int,
            ) -> int:
        num_of_successes = 0
        for _ in range(num_of_episodes):
            current_state, _ = self.env.reset()
            for _ in range(num_of_steps):
                action = np.argmax(self.q_table[current_state, :])
                next_state, reward, done, truncated, _ = self.env.step(action)
                current_state = next_state
                if done or truncated:
                    break
            if reward == 1:
                num_of_successes += 1
        success_rate = num_of_successes / num_of_episodes * 100
        print(f"Success rate: {round(success_rate, 2)}%")
        return success_rate


def compare_average_rewards(
        num_of_episodes: int,
        num_of_steps: int,
        num_of_id_runs: int,
        reward_function: Callable[[int, int, int, bool], int],
        is_slippery: bool = False
        ) -> np.array:
    average_rewards = np.zeros(num_of_episodes)
    for _ in range(num_of_id_runs):
        qlearn = QLearnFrozenLake(
            num_of_episodes, num_of_steps, 0.1, 0.8, is_slippery
            )
        qlearn.train(reward_function, average_rewards)
    return average_rewards / num_of_id_runs


def compare_success_rates(
        num_of_episodes: int,
        num_of_steps: int,
        num_of_id_runs: int,
        reward_function: Callable[[int, int, int, bool], int],
        is_slippery: bool = False
        ) -> np.array:
    success_rates = np.zeros(num_of_id_runs)
    for i in range(num_of_id_runs):
        qlearn = QLearnFrozenLake(
            num_of_episodes, num_of_steps, 0.1, 0.8, is_slippery
            )
        print(f'{qlearn.q_table}')
        qlearn.train(reward_function)
        print(f'{qlearn.q_table}')
        success_rates[i] = qlearn.evaluate_qtable(1000, 200)
    return sum(success_rates) / num_of_id_runs


def plot_results(averaged_reward_base: np.array, averaged_reward: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(averaged_reward_base, 'r')
    plt.plot(averaged_reward, 'b')
    plt.show()


def main():
    num_of_id_runs = 25
    num_of_episodes = 5000
    num_of_steps = 200

    # averaged_reward_base = compare_average_rewards(
    #     num_of_episodes,
    #     num_of_steps,
    #     num_of_id_runs,
    #     QLearnFrozenLake.reward_default,
    #     True
    #     )
    # averaged_reward = compare_average_rewards(
    #     num_of_episodes,
    #     num_of_steps,
    #     num_of_id_runs,
    #     QLearnFrozenLake.reward_alternative_1,
    #     True
    #     )

    # plot_results(averaged_reward_base, averaged_reward)

    success_rate = compare_success_rates(
        num_of_episodes,
        num_of_steps,
        num_of_id_runs,
        QLearnFrozenLake.reward_default,
        True
        )

    print(
        f"QTable trained on {num_of_episodes} episodes, "
        f"Success rate: {success_rate}"
    )


if __name__ == "__main__":
    main()
