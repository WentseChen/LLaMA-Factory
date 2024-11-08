
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

import json
import time

import os
import sys

from utils.env import MiniGrid, ActionWrapper, KeyDoorWrapper
from utils.venv import vEnv
from utils.utils import load_text, get_traj

# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--env_name', type=str, default="BabyAI-UnlockLocal-v0", help='environment name')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for sampling')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
parser.add_argument('--render', type=bool, default=False, help='whether to render the environment')
parser.add_argument('--num_parallel', type=int, default=3, help='number of parallel environments')
parser.add_argument('--rank', type=int, default=1, help='rank of the process')
args = parser.parse_args()

api_key = "EMPTY"
api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=api_key, base_url=api_base)
def gpt(msgs, return_logprob=False):   
    result = client.chat.completions.create(
        messages=msgs,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=4096,
    ) 
    return result.choices[0].message.content

# wait for the server to start
test_msg = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply to this message with Hello."}
]
while True:
    try:
        result = gpt(test_msg)
        break
    except Exception as e:
        print("Failed to get GPT response, retrying...")
        time.sleep(.25)

system_tmp = load_text("prompt/policy_system_cot.txt")
system_msg = {"role": "system", "content": system_tmp}

all_data = {
    "obs": [],
    "action": [],
    "traj": [],
}

env = MiniGrid(args.env_name)
env = ActionWrapper(env)
env = KeyDoorWrapper(env)

obs, info = env.reset()
done = False

obs_cache = [obs]
act_cache = []

while len(all_data["obs"]) < args.batch_size:
    
    human_tmp = load_text("prompt/policy_human.txt")
    human_tmp = obs + human_tmp
    human_msg = {"role": "user", "content": human_tmp}
    msg = [system_msg, human_msg]
    
    retry_cnt = 0
    while retry_cnt < 3:
        
        try:
            response = gpt(msg)
            if "My next action is to " in response:
                response = response.split("My next action is to ")[1]
            action_text = None
            if "move up" in response:
                action_text = "move in the positive y-direction"
            elif "move down" in response:
                action_text = "move in the negative y-direction"
            elif "move left" in response:
                action_text = "move in the negative x-direction"
            elif "move right" in response:
                action_text = "move in the positive x-direction"
            for act_candidate in env.available_actions:
                if act_candidate in response:
                    action_text = act_candidate
            if action_text is not None:
                break
        except:
            retry_cnt += 1
            
    all_data["obs"].append(obs)
    all_data["action"].append(action_text)
    
    act_cache.append(action_text)
            
    obs, reward, done, truncated, info = env.step(action_text)
    
    if done or truncated:
        if reward > 0.:
            result = "Result: You have successfully completed the task!"
        else:
            result = "Result: You have failed to complete the task."
        obs_cache.append(obs)
        for j in range(len(obs_cache)-1):
            traj = get_traj(obs_cache[j:], act_cache[j:], result)
            all_data["traj"].append(traj)
        obs_cache = []
        act_cache = []

        obs, info = env.reset()
    
    obs_cache.append(obs)
    
# save 
file_name = args.file_path + "/phase3_rank" + str(args.rank) + ".json"
if not os.path.exists(file_name):
    with open(file_name, "w") as f:
        json.dump(all_data, f)

    
print("reward:", reward)
