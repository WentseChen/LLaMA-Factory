
import minigrid
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ReseedWrapper

COLOR_TO_IDX = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'purple': 3,
    'yellow': 4,
    'grey': 5,
}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

STATE_TO_IDX = {
    "opened": 0,
    "closed": 1,
    "locked": 2,
    "unknown": 3,
}
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

DIRECTION_TO_IDX = {
    "positive x-direction": 0,
    "positive y-direction": 1,
    "negative x-direction": 2,
    "negative y-direction": 3,
}
IDX_TO_DIRECTION = dict(zip(DIRECTION_TO_IDX.values(), DIRECTION_TO_IDX.keys()))

ACTION_TO_IDX = {
    "turn clockwise": 0,
    "turn counterclockwise": 1,
    "move forward": 2,
    "pick up": 3,
    "drop": 4,
    "toggle": 5,
    "no-op": 6,
}
IDX_TO_ACTION = dict(zip(ACTION_TO_IDX.values(), ACTION_TO_IDX.keys()))

# ALL_TEXT_ACTIONS = [
#     "move 1 step right",
#     "move 1 step up",
#     "move 1 step left",
#     "move 1 step down",
#     "pick up the thing 1 step right",
#     "pick up the thing 1 step up",
#     "pick up the thing 1 step left",
#     "pick up the thing 1 step down",
#     "drop the thing 1 step right",
#     "drop the thing 1 step up",
#     "drop the thing 1 step left",
#     "drop the thing 1 step down",
#     "toggle the door 1 step right",
#     "toggle the door 1 step up",
#     "toggle the door 1 step left",
#     "toggle the door 1 step down",
# ]

ALL_TEXT_ACTIONS = [
    "move in the positive x-direction",
    "move in the positive y-direction",
    "move in the negative x-direction",
    "move in the negative y-direction",
    "pick up an item in the positive x-direction",
    "pick up an item in the positive y-direction",
    "pick up an item in the negative x-direction",
    "pick up an item in the negative y-direction",
    "drop an item in the positive x-direction",
    "drop an item in the positive y-direction",
    "drop an item in the negative x-direction",
    "drop an item in the negative y-direction",
    "toggle a door in the positive x-direction",
    "toggle a door in the positive y-direction",
    "toggle a door in the negative x-direction",
    "toggle a door in the negative y-direction",
]

class MiniGrid(gym.Env):
    def __init__(self, scenario):
        """
        scenario: str, name of the scenario to load
        """
        self.env = gym.make(scenario, render_mode="rgb_array")
        self.env = FullyObsWrapper(self.env)
        self.action_space = gym.spaces.Discrete(7)
        self.all_actions = [action for action in ACTION_TO_IDX.keys()]
        self.env.max_steps = 30
        self.available_actions = ALL_TEXT_ACTIONS
        self._episode_cnt = 0
        self._success_cnt = 0
    
    def obs2state(self, obs):
        
        goal = obs['mission']
        image = obs['image']
        direction = obs['direction']
        
        output_str = "Your goal is to "
        output_str += goal + "\n"
        for i, row in enumerate(image):
            for j, block in enumerate(row):
                obj = IDX_TO_OBJECT[block[0]]
                color = IDX_TO_COLOR[block[1]]
                state = IDX_TO_STATE[block[2]]
                d = IDX_TO_DIRECTION[direction]
                if obj == "door":
                    if state == "opened":
                        output_str += f"You see an opened {color} door at (x={i}, y={j}).\n"
                    else:
                        output_str += f"You see a {state} {color} door at (x={i}, y={j}).\n"
                elif obj == "empty":
                    continue
                elif obj == "wall":
                    continue
                elif obj == "agent":
                    output_str += f"You are at (x={i}, y={j}), facing {d}.\n"
                else:
                    output_str += f"You see a {color} {obj} at (x={i}, y={j}).\n"
        return output_str
    
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        for key in obs.keys():
            info[key] = obs[key]
        obs = self.obs2state(obs)
        self.time_step = 0
        return obs, info
    
    @property
    def success_cnt(self):
        return self._success_cnt
    
    @property
    def episode_cnt(self):
        return self._episode_cnt

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        for key in obs.keys():
            info[key] = obs[key]
        obs = self.obs2state(obs)
        self.time_step += 1
        done = done or self.time_step >= self.env.max_steps
        if done:
            self._episode_cnt += 1
            if reward > 0:
                self._success_cnt += 1
        return obs, reward, done, truncated, info

    def render(self):
        return self.env.render()

class KeyDoorWrapper(gym.Wrapper):
    def __init__(self, env):
        super(KeyDoorWrapper, self).__init__(env)
        self._last_obs = ""
        self._repeat_time = 0

    def _add_key(self, obs):
        split_obs = obs.split("\n")
        goal = split_obs[0]
        obs = "\n".join(split_obs[1:])
        # if "key" not in obs:
        #     obs += "You carry a key.\n"
        obs = goal + "\n" + obs
        return obs
    
    def _change_coord(self, obs):
        split_obs = obs.split("\n")
        for i, o in enumerate(split_obs):
            if "at (x=" in o:
                x = int(o.split("x=")[1].split(",")[0].split(")")[0])
                y = int(o.split("y=")[1].split(")")[0])
                split_obs[i] = o.replace(f"x={x}", f"x={x-6}").replace(f"y={y}", f"y={y-6}")
        obs = "\n".join(split_obs)
        return obs
    
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        obs = self._add_key(obs)
        obs = self._change_coord(obs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, truncated, done, info = self.env.step(action)
        obs = self._add_key(obs)
        obs = self._change_coord(obs)
        # if obs == self._last_obs:
        #     self._repeat_time += 1
        #     if self._repeat_time >= 3:
        #         done = True
        self._last_obs = obs
        return obs, reward, truncated, done, info

class ActionWrapper(gym.Wrapper):
    """
    env can now take text action as input, available actions are:
    - move ** right/left/up/down
    - pick up ** right/left/up/down
    - drop the ** right/left/up/down
    - toggle ** right/left/up/down
    where ** can be anything, but don't contain "right", "left", "up", "down"
    """
    
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.direction = info["direction"]
        return obs, info

    def step(self, text_action):
        
        if "move" in text_action:
            if "positive x-direction" in text_action:
                if self.direction == 0:
                    text_action = "move forward"
                elif self.direction == 1:
                    self.env.step(0)
                    text_action = "move forward"
                elif self.direction == 2:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "move forward"
                else:
                    self.env.step(1)
                    text_action = "move forward"
            elif "negative x-direction" in text_action:
                if self.direction == 0:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "move forward"
                elif self.direction == 1:
                    self.env.step(1)
                    text_action = "move forward"
                elif self.direction == 2:
                    text_action = "move forward"
                else:
                    self.env.step(0)
                    text_action = "move forward"
            elif "negative y-direction" in text_action:
                if self.direction == 0:
                    self.env.step(0)
                    text_action = "move forward"
                elif self.direction == 1:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "move forward"
                elif self.direction == 2:
                    self.env.step(1)
                    text_action = "move forward"
                else:
                    text_action = "move forward"
            else:
                if self.direction == 0:
                    self.env.step(1)
                    text_action = "move forward"
                elif self.direction == 1:
                    text_action = "move forward"
                elif self.direction == 2:
                    self.env.step(0)
                    text_action = "move forward"
                else:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "move forward"
        elif "pick up" in text_action:
            
            if "positive x-direction" in text_action:
                if self.direction == 0:
                    text_action = "pick up"
                elif self.direction == 1:
                    self.env.step(0)
                    text_action = "pick up"
                elif self.direction == 2:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "pick up"
                else:
                    self.env.step(1)
                    text_action = "pick up"
            elif "negative x-direction" in text_action:
                if self.direction == 0:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "pick up"
                elif self.direction == 1:
                    self.env.step(1)
                    text_action = "pick up"
                elif self.direction == 2:
                    text_action = "pick up"
                else:
                    self.env.step(0)
                    text_action = "pick up"
            elif "negative y-direction" in text_action:
                if self.direction == 0:
                    self.env.step(0)
                    text_action = "pick up"
                elif self.direction == 1:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "pick up"
                elif self.direction == 2:
                    self.env.step(1)
                    text_action = "pick up"
                else:
                    text_action = "pick up"
            else:
                if self.direction == 0:
                    self.env.step(1)
                    text_action = "pick up"
                elif self.direction == 1:
                    text_action = "pick up"
                elif self.direction == 2:
                    self.env.step(0)
                    text_action = "pick up"
                else:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "pick up"
        elif "toggle" in text_action:
            # assert only one ["right", "left", "up", "down"] in the text_action
            assert sum([1 if d in text_action else 0 for d in ["right", "left", "up", "down"]]) == 1
            if "positive x-direction" in text_action:
                if self.direction == 0:
                    text_action = "toggle"
                elif self.direction == 1:
                    self.env.step(0)
                    text_action = "toggle"
                elif self.direction == 2:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "toggle"
                else:
                    self.env.step(1)
                    text_action = "toggle"
            elif "negative x-direction" in text_action:
                if self.direction == 0:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "toggle"
                elif self.direction == 1:
                    self.env.step(1)
                    text_action = "toggle"
                elif self.direction == 2:
                    text_action = "toggle"
                else:
                    self.env.step(0)
                    text_action = "toggle"
            elif "negative y-direction" in text_action:
                if self.direction == 0:
                    self.env.step(0)
                    text_action = "toggle"
                elif self.direction == 1:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "toggle"
                elif self.direction == 2:
                    self.env.step(1)
                    text_action = "toggle"
                else:
                    text_action = "toggle"
            else:
                if self.direction == 0:
                    self.env.step(1)
                    text_action = "toggle"
                elif self.direction == 1:
                    text_action = "toggle"
                elif self.direction == 2:
                    self.env.step(0)
                    text_action = "toggle"
                else:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "toggle"
        elif "drop" in text_action:
            if "positive x-direction" in text_action:
                if self.direction == 0:
                    text_action = "drop"
                elif self.direction == 1:
                    self.env.step(0)
                    text_action = "drop"
                elif self.direction == 2:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "drop"
                else:
                    self.env.step(1)
                    text_action = "drop"
            elif "negative x-direction" in text_action:
                if self.direction == 0:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "drop"
                elif self.direction == 1:
                    self.env.step(1)
                    text_action = "drop"
                elif self.direction == 2:
                    text_action = "drop"
                else:
                    self.env.step(0)
                    text_action = "drop"
            elif "negative y-direction" in text_action:
                if self.direction == 0:
                    self.env.step(0)
                    text_action = "drop"
                elif self.direction == 1:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "drop"
                elif self.direction == 2:
                    self.env.step(1)
                    text_action = "drop"
                else:
                    text_action = "drop"
            else:
                if self.direction == 0:
                    self.env.step(1)
                    text_action = "drop"
                elif self.direction == 1:
                    text_action = "drop"
                elif self.direction == 2:
                    self.env.step(0)
                    text_action = "drop"
                else:
                    self.env.step(1)
                    self.env.step(1)
                    text_action = "drop"
        else:
            raise ValueError("Unknown action")
        
        action = ACTION_TO_IDX[text_action]
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.direction = info["direction"]
        
        return obs, reward, done, truncated, info

if __name__ == "__main__":
    
    # env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    # observation, info = env.reset(seed=42)
    # exit()
    
    import PIL
    from PIL import Image
    
    env = MiniGrid("BabyAI-UnlockLocal-v0")
    
    all_img = []
    obs, info = env.reset()
    img_array = env.render()
    all_img.append(Image.fromarray(img_array))
    done = False
    print("obs:", obs)
    
    while not done:
        
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        img_array = env.render()
        all_img.append(Image.fromarray(img_array))
    
    # Save the images to a gif
    all_img[0].save("test.gif", save_all=True, append_images=all_img[1:], loop=0, duration=250)
