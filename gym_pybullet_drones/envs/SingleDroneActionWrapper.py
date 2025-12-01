import gymnasium as gym
import numpy as np

class SingleDroneActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        
        low  = env.action_space.low.reshape(-1)
        high = env.action_space.high.reshape(-1)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    
    def action(self, action):
        if action.ndim == 1:
            return action.reshape(1, -1)

        if action.shape[0] == 1:
            return action

        return action[0].reshape(1, -1)
    
    def __getattr__(self, name):
        return getattr(self.env, name)