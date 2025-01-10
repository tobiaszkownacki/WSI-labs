import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class QLearnFrozenLake:
    DELTA_EPSILON = 0.008
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
        rewards: list[int],
    ) -> None:

        epsilon: float = 1.0

        qtable = np.zeros((self.state_size, self.action_size))

        for episode in range(self.num_of_episodes):
            current_state, _ = self.env.reset()
            for step in range(self.num_of_steps):
                action = self.choose_action(
                    current_state, epsilon, qtable
                )

                next_state, reward, done, _, _ = self.env.step(action)

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

                if done:
                    break

            epsilon = max(
                epsilon - self.DELTA_EPSILON * episode, self.MIN_EPSILON
            )
            rewards[episode] += reward

        self.q_table = qtable
        return rewards

    def choose_action(
        self, state: int, epsilon: float, qtable: np.array
    ) -> int:
        if np.random.uniform(0, 1) < epsilon or np.max(qtable[state, :]) == 0:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(qtable[state, :])
        return action

    @staticmethod
    def reward_default(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        return reward

    @staticmethod
    def reward_negative_one(
        reward: int, next_state: int, current_state: int, done: bool
    ) -> int:
        if done and reward == 0:
            return -10
        elif done and reward == 1:
            return 100
        elif next_state == current_state:
            return -1
        else:
            return 0


def cos_average_rewards(
        num_of_episodes: int, num_of_steps: int, num_of_id_runs: int
        ) -> np.array:
    average_rewards = np.zeros(num_of_episodes)
    for _ in range(num_of_id_runs):
        qlearn = QLearnFrozenLake(num_of_episodes, num_of_steps, 0.25, 0.95)
        qlearn.train(QLearnFrozenLake.reward_default, average_rewards)
    return average_rewards / num_of_id_runs


def main():
    averaged_reward = cos_average_rewards(1000, 200, 25)
    averaged_reward_base = cos_average_rewards(1000, 200, 25)

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


if __name__ == "__main__":
    main()