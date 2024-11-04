
import minigrid
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ReseedWrapper


class vEnv(gym.Env):
    def __init__(self, num_env, env_func, **kwargs):
        self.envs = []
        for _ in range(num_env):
            env = env_func(**kwargs)
            self.envs.append(env)
        self.action_space = self.envs[0].action_space
        self.available_actions = self.envs[0].available_actions
        
    def reset(self, seed=None):
        
        if seed is not None:
            print("WARNING: seed is not support in this version")
        
        all_obs = []
        all_info = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            all_obs.append(obs)
            all_info.append(info)
        
        return all_obs, all_info
    
    def step(self, text_actions):
        
        all_obs = []
        all_reward = []
        all_done = []
        all_truncated = []
        all_info = []
        for env, text_action in zip(self.envs, text_actions):
            obs, reward, done, truncated, info = env.step(text_action)
            if done or truncated:
                last_obs = obs
                obs, info = env.reset()
                info["last_obs"] = last_obs
            all_obs.append(obs)
            all_reward.append(reward)
            all_done.append(done)
            all_truncated.append(truncated)
            all_info.append(info)
        
        return all_obs, all_reward, all_done, all_truncated, all_info
    
    def render(self):
        """
        only return the first env's render result
        """
        return self.envs[0].render()
        

if __name__ == "__main__":
    
    from env import MiniGrid, ActionWrapper, KeyDoorWrapper
    
    def env_func(scenario):
        env = MiniGrid(scenario)
        env = ActionWrapper(env)
        env = KeyDoorWrapper(env)
        return env

    # load env 
    env = vEnv(2, env_func, scenario="BabyAI-UnlockLocal-v0")
    obs, info = env.reset()

    print(obs)