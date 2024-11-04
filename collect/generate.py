
import json
import numpy as np

from utils.venv import vEnv
from utils.utils import load_text
from utils.env import MiniGrid, ActionWrapper, KeyDoorWrapper

# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--env_name', type=str, default="BabyAI-UnlockLocal-v0", help='environment name')
parser.add_argument('--alpha', type=float, default=5.0, help='trust region size')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
args = parser.parse_args()

# load data from json
file_name = args.file_path + "/phase3.json"
with open(file_name, "r") as f:
    all_input_data = json.load(f)
    
# load env
def env_func(scenario):
    env = MiniGrid(scenario)
    env = ActionWrapper(env)
    env = KeyDoorWrapper(env)
    return env
env_num = 1
env = vEnv(env_num, env_func, scenario=args.env_name)


for i in range(len(all_input_data["obs"])):
    
    obs = all_input_data["obs"][i]
    updated_logp = all_input_data["updated_logp"][i]
    sampling_logp = all_input_data["sampling_logp"][i]
    
    updated_logp = np.array([float(p) for p in updated_logp])
    sampling_logp = np.array([float(p) for p in sampling_logp])
    
    star_prob = np.exp(sampling_logp) * np.exp(updated_logp/args.alpha)
    star_prob = star_prob / np.sum(star_prob)
    
    for _ in range(10):
        
        action = np.random.choice(len(star_prob), p=star_prob)
        text_action = env.available_actions[action]
        text_action = "<next action>\n" + text_action + "\n</next action>"
        
        system_msg = load_text("prompt/policy.txt")
        system_msg = {"role": "system", "content": system_msg}
        human_msg = {"role": "user", "content": obs+"After reading through the rules of the environment, your goal and all the current observation. What is the next action to achieve your goal?"}
        ai_msg = {"role": "assistant", "content": text_action}
        msg = {"messages":[system_msg, human_msg, ai_msg]}
        
        file_name = args.file_path + "/babyai.json"
        with open(file_name, "a") as f:
            json.dump(msg, f)
            f.write("\n")


