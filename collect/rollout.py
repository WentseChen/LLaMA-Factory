
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.env import MiniGrid, ActionWrapper, KeyDoorWrapper
from utils.venv import vEnv
from utils.utils import load_text, build_prompt, get_assistant_index, get_traj
from utils.agent import LlamaAgent

from accelerate import Accelerator


# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--env_name', type=str, default="BabyAI-UnlockLocal-v0", help='environment name')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for sampling')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
parser.add_argument('--num_parallel', type=int, default=8, help='number of parallel environments')
args = parser.parse_args()

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
env = vEnv(args.num_parallel, env_func, scenario=args.env_name)
obs, info = env.reset()

# load agent
system_tmp = load_text("prompt/policy.txt")
agent = LlamaAgent(model, tokenizer, accelerator, env, system_tmp)

# variables
all_cnt = 0
succ_cnt = 0

# pre-process data
obs_cache = [[obs[i]] for i in range(args.num_parallel)]
act_cache = [[] for _ in range(args.num_parallel)]
act_logp_cache = [[] for _ in range(args.num_parallel)]
all_data = {
    "obs": [],
    "traj": [],
    "sampling_logp": [],
}

while len(all_data["traj"]) < args.batch_size * 1.2:
    
    act_dist = agent(obs, fast=True, temperature=args.temperature)
    sample_action = act_dist.sample()
    sample_action_text = [env.available_actions[i] for i in sample_action]
    
    for i in range(args.num_parallel):
        act_cache[i].append(sample_action_text[i])
        act_logp_cache[i].append(act_dist.probs[i].log())
    
    obs, reward, done, truncated, info = env.step(sample_action_text)

    # save data
    for i in range(args.num_parallel):
        if done[i] or truncated[i]:
            obs_cache[i].append(info[i]["last_obs"])
            if reward[i] > 0.:
                result = "Result: You have successfully completed the task!"
                all_cnt += 1
                succ_cnt += 1
            else:
                result = "Result: You have failed to complete the task."
                all_cnt += 1
            for j in range(len(obs_cache[i])-1):
                all_data["obs"].append(obs_cache[i][j])
                traj = get_traj(obs_cache[i][j:], act_cache[i][j:], result)
                all_data["traj"].append(traj)
                sampling_logp = act_logp_cache[i][j]
                slogp_str = ["{:.4f}".format(sampling_logp[k]) for k in range(len(sampling_logp))]
                all_data["sampling_logp"].append(slogp_str)
            obs_cache[i] = []
            act_cache[i] = []
            act_logp_cache[i] = []

    for i in range(args.num_parallel):
        obs_cache[i].append(obs[i])


if accelerator.is_main_process:
    print("All:", all_cnt, "Success:", succ_cnt)
    # save all_data as a json
    import json
    file_name = args.file_path + "/phase1.json"
    with open(file_name, "w") as f:
        json.dump(all_data, f)


