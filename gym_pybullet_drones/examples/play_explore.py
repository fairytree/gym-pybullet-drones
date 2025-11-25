import os
import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import PPO, TD3
from gym_pybullet_drones.envs.ExploreAviary import ExploreAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

# -------------------- DEFAULTS --------------------
DEFAULT_MODEL_PATH = "results/best_model.zip"
DEFAULT_GUI = True
DEFAULT_OBS = ObservationType('kin')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

DEFAULT_ALGO = "PPO"  # "PPO" or "TD3" RL algorithm
DEFAULT_ACT = ActionType('pid')  # 'rpm' for RL to output rpm directly or 'pid' for RL to output waypoints tracked by PID
DEFAULT_INCENTIVE_OPTIONS = {
    "new_voxel_reward": True, # Reward for exploring a new voxel
    "out_of_boundary_penalty": True, # Penalty for going out of predefined boundaries
    "change_direction_penalty": True, # Penalty for changing direction abruptly
    "collision_penalty": True, # Penalty for colliding with obstacles
    "time_penalty": True, # Penalty for time taken to encourage faster exploration
    # "exploration_percentage": True, # provide additional observation of percentage explored
    # "nearest_unexplored_voxel": True # provide additional observation of position of nearest unexplored voxel
}

def play(model_path=DEFAULT_MODEL_PATH, algo=DEFAULT_ALGO, multiagent=DEFAULT_MA, gui=DEFAULT_GUI,incentive_options=DEFAULT_INCENTIVE_OPTIONS):
    #### Load saved model ####
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        return

    if algo.upper() == "PPO":
        model_class = PPO
    elif algo.upper() == "TD3":
        model_class = TD3
    else:
        print(f"[ERROR] Unsupported algorithm: {algo}. Choose 'PPO' or 'TD3'.")
        return

    model = model_class.load(model_path)
    print(f"[INFO] Loaded {algo.upper()} model from {model_path}")

    #### Create test environment ####
    if not multiagent:
        env = ExploreAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT, incentive_options=incentive_options)
    else:
        env = MultiHoverAviary(gui=gui, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder="logs_playback/",
                    colab=False)

    #### Run the simulation ####
    obs, _ = env.reset(seed=42, options={})
    start = time.time()

    # Initialize trajectory and drawn voxels
    if not hasattr(env, 'trajectory'):
        env.trajectory = []
    if not hasattr(env, 'drawn_voxels'):
        env.drawn_voxels = set()
    if not hasattr(env, 'voxel_visual_ids'):
        env.voxel_visual_ids = []

    total_reward = 0.0

    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        
        obs2 = obs.squeeze()
        act2 = action.squeeze()

        print("action:", act2)

        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                     np.zeros(4),
                                     obs2[3:15],
                                     act2]),
                    control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                         np.zeros(4),
                                         obs2[d][3:15],
                                         act2[d]]),
                        control=np.zeros(12))

    

        # -------------------- Visualization --------------------
        if not multiagent:
            # Trajectory
            env.visualize_trajectory()

            # Visited voxels (only draw new)
            for voxel in env.visited:
                if voxel not in env.drawn_voxels:
                    voxel_center = env.bounds[:, 0] + np.array(voxel) * env.grid_size + env.grid_size/2
                    half_extents = [env.grid_size/2]*3
                    visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0,1,0,0.3])
                    voxel_id = p.createMultiBody(baseMass=0,
                                                 baseCollisionShapeIndex=-1,
                                                 baseVisualShapeIndex=visual_id,
                                                 basePosition=voxel_center,
                                                 physicsClientId=env.CLIENT)
                    env.voxel_visual_ids.append(voxel_id)
                    env.drawn_voxels.add(voxel)

        # Update camera to slowly orbit
        cam_info = p.getDebugVisualizerCamera(physicsClientId=env.CLIENT)
        cam_distance = cam_info[10]  # current distance (can be zoomed with mouse)
        speed = 0.05
        yaw = -30 + i * speed        # adjust speed of rotation by changing speed variable
        pitch = -30                  # keep pitch fixed
        p.resetDebugVisualizerCamera(cameraDistance=cam_distance,
                                     cameraYaw=yaw,
                                     cameraPitch=pitch,
                                     cameraTargetPosition=[0,0,0],
                                     physicsClientId=env.CLIENT)
                                    
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated or truncated:
            break

    print("visited voxels:", len(env.visited), "total reward:", total_reward)
    input("Press Enter to close the environment...")
    env.close()
    logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained RL policy in PyBullet drones environment.")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to saved policy zip file')
    parser.add_argument('--algo', type=str, default=DEFAULT_ALGO, help="Algorithm: 'PPO' or 'TD3'")
    parser.add_argument('--multiagent', type=bool, default=DEFAULT_MA, help='Whether to use MultiHoverAviary')
    parser.add_argument('--gui', type=bool, default=DEFAULT_GUI, help='Enable GUI rendering')
    args = parser.parse_args()

    play(**vars(args))
