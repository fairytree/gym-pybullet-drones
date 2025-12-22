
from typing import List, Tuple
import numpy as np
import gymnasium as gym
from gridworld_mdp import GridworldMDP

Coord = Tuple[int, int]  # (row, col)

class GridworldEnv(gym.Env):
    """
    Gymnasium environment wrapper for GridworldMDP.
    """

    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(
            self, 
            height: int,
            width: int,
            init: Coord,
            goal: Coord,
            sink: Coord | None = None,
            wall: Coord | List[Coord] | None = None,
            reward_goal: float = +1.0,
            reward_sink: float = -1.0,
            step_cost: float = -0.1,
            slip_p: float = 0.3,
            discount : float = 0.99,
            reward_depends_on_next : bool = False,
        ):
        """
        Initialize the Gridworld Gym environment.
        """
        super().__init__()

        # Create the underlying MDP
        self.mdp = GridworldMDP(
            height=height,
            width=width,
            init=init,
            goal=goal,
            sink=sink,
            wall=wall,
            reward_goal=reward_goal,
            reward_sink=reward_sink,
            step_cost=step_cost,
            slip_p=slip_p,
            discount=discount,
            reward_depends_on_next=reward_depends_on_next,
        )

        # TODO: Define action and observation spaces and expose nS and nA
        self.action_space = gym.spaces.Discrete(self.mdp.nA)
        self.observation_space = gym.spaces.Discrete(self.mdp.nS)

        # set the initial state
        self.state=self.mdp.init


        #expose nS and nA
        self.nS=self.mdp.nS
        self.nA=self.mdp.nA
        

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        Returns:
          obs: the observation corresponding to the reset initial state 
            (in this case just return the initial state index itself)
          info: additional info (empty dict in this case)
        """
        super().reset(seed=seed)
        self.state = self.mdp.init
        return self.state, {}

    def step(self, action):
        """
        Take an action in the environment.
        Arguments:
            action: the action to take (integer in [0, nA-1])
        Returns:
          next_state: the next state after taking the action
          reward: the reward received
          done: whether the episode has ended
          info: additional info (empty dict here)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        # Sample next state based on transition probabilities
        transitions = self.mdp.P(self.state, action)
        probs = [t.prob for t in transitions]
        chosen_transition = np.random.choice(transitions, p=probs)
        # Update the current state
        self.state = chosen_transition.next_state
        # Get reward and done status
        reward = chosen_transition.reward
        done = self.mdp.is_absorbing(self.state)
        return self.state, reward, done, False, {}