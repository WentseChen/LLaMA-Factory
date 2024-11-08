
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
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
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

all_msg = [[] for _ in range(10)]
for i in range(args.batch_size):
    
    obs = all_input_data["obs"][i]
    act = all_input_data["action"][i]
    
    system_tmp = load_text("prompt/policy_system.txt")
    system_msg = {"role": "system", "content": system_tmp}
    human_tmp = load_text("prompt/policy_human.txt")
    human_msg = {"role": "user", "content": obs + human_tmp}
    text_action = "My next action is to " + act
    ai_msg = {"role": "assistant", "content": text_action}
    msg = {"messages":[system_msg, human_msg, ai_msg]}
    
    for j in range(10):
        all_msg[j].append(msg)

# shuffle the all_msg, which is a list
file_name = args.file_path + "/babyai.json"
with open(file_name, "a") as f:
    for j in range(10):
        all_msg[j] = np.random.permutation(all_msg[j])
        for k in range(len(all_msg[j])):
            json.dump(all_msg[j][k], f)
            f.write("\n")


