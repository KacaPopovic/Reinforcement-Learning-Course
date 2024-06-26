import copy
from typing import Any, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        # TODO: Define your action_space and observation_space here
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low = np.array([0,0]),
                                            high = np.array([len(self.map[0])-1, len(self.map[0])-1]),
                                            dtype=np.int32)
        self.agent_position = [0, 0]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.agent_position = [0, 0]
        return self._observe(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = None
        done = None
        # TODO: Write your implementation here
        if action == 0:
            # Go up
            self.agent_position[0] = clamp(self.agent_position[0] - 1, 0, len(self.map) - 1)
        elif action == 1:  # Go right
            self.agent_position[1] = clamp(self.agent_position[1] + 1, 0, len(self.map[0]) - 1)
        elif action == 2:  # Go down
            self.agent_position[0] = clamp(self.agent_position[0] + 1, 0, len(self.map) - 1)
        elif action == 3:  # Go left
            self.agent_position[1] = clamp(self.agent_position[1] - 1, 0, len(self.map[0]) - 1)

        current_cell = self.map[self.agent_position[0]][self.agent_position[1]]

        if current_cell == "g":
            reward = 1
            done = True
        elif current_cell == "t":
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        observation = self._observe()
        return observation, reward, done, False, {}

    def render(self):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass

    def _observe(self):
        return np.array(self.agent_position)
