
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.venv import vEnv
from utils.utils import load_text
from utils.agent import LlamaAgent
from utils.env import MiniGrid, ActionWrapper, KeyDoorWrapper

from accelerate import Accelerator

# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--env_name', type=str, default="BabyAI-UnlockLocal-v0", help='environment name')
parser.add_argument('--mini_batch_size', type=int, default=16, help='number data size for each forward pass')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature for sampling')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
args = parser.parse_args()

# load data from json
file_name = args.file_path + "/phase2.json"
with open(file_name, "r") as f:
    all_input_data = json.load(f)

all_output_data = {
    "obs": [],
    "sampling_logp": [],
    "updated_logp": [],
}

# load model
accelerator = Accelerator()
device_map = {"": accelerator.process_index}
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, device_map=device_map, torch_dtype=torch.bfloat16
)
token_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(token_name)
tokenizer.pad_token = tokenizer.eos_token

# load env
def env_func(scenario):
    env = MiniGrid(scenario)
    env = ActionWrapper(env)
    env = KeyDoorWrapper(env)
    return env
env_num = 1
env = vEnv(env_num, env_func, scenario=args.env_name)

# load agent
system_tmp = load_text("prompt/policy.txt")
agent = LlamaAgent(model, tokenizer, accelerator, env, system_tmp)

# pre-process data
num_mini_batch = len(all_input_data["obs"]) // args.mini_batch_size
if len(all_input_data["obs"]) % args.mini_batch_size != 0:
    num_mini_batch += 1
    
for i in range(num_mini_batch):
    
    start_idx = i * args.mini_batch_size
    end_idx = (i+1) * args.mini_batch_size
    batch_obs = all_input_data["obs"][start_idx:end_idx]
    pad_num = args.mini_batch_size - len(batch_obs)
    
    for j in range(len(batch_obs)):
        verbal_fb = all_input_data["verbal_feedback"][start_idx+j]
        batch_obs[j] += "\nFeedback you should follow\n" + verbal_fb
    
    if pad_num > 0:
        batch_obs += [batch_obs[-1]] * pad_num
    
    act_dist = agent(batch_obs, fast=True, temperature=args.temperature)
    act_prob = act_dist.probs
    act_logp = act_prob.log().cpu().numpy()
    
    for j in range(len(batch_obs)-pad_num):
        all_output_data["obs"].append(batch_obs[j])
        all_output_data["updated_logp"].append(["{:.4f}".format(p) for p in act_logp[j]])
        all_output_data["sampling_logp"].append(all_input_data["sampling_logp"][start_idx+j])

# save data
file_name = args.file_path + "/phase3.json"
with open(file_name, "w") as f:
    json.dump(all_output_data, f)
    