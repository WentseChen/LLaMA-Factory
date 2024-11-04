
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.env import MiniGrid, ActionWrapper, KeyDoorWrapper
from utils.venv import vEnv
from utils.utils import load_text, build_prompt, get_assistant_index, get_traj
from utils.agent import LlamaAgent

from accelerate import Accelerator

from PIL import Image

# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature for sampling')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
parser.add_argument('--num_parallel', type=int, default=16, help='number of parallel environments')
args = parser.parse_args()

# load model
device_index = Accelerator().process_index
device_map = {"": device_index}
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
env = vEnv(args.num_parallel, env_func, scenario="BabyAI-UnlockLocal-v0") # "BabyAI-GoToDoor-v0") 
obs, info = env.reset()
img = env.render()
img = Image.fromarray(img)
done = [False for _ in range(args.num_parallel)]

# load agent
system_tmp = load_text("prompt/policy.txt")
agent = LlamaAgent(model, tokenizer, env, system_tmp)

# variables
all_cnt = 0
succ_cnt = 0
all_img = [img]

while all([not d for d in done]):
    
    print("obs:", obs)
    
    for i in range(len(obs)):
        obs[i] = obs[i] + "After reading through the rules of the environment, your goal and all the current observation. What is the next action to achieve your goal?"
    
    
    act_dist = agent(obs, fast=True, temperature=args.temperature)
    sample_action = act_dist.sample()
    sample_action_text = [env.available_actions[i] for i in sample_action]
    obs, reward, done, truncated, info = env.step(sample_action_text)
    
    print("text_action:", sample_action_text)
    
    img = env.render()
    img = Image.fromarray(img)
    all_img.append(img)

    for r, d in zip(reward, done):
        if d and r > 0.:
            succ_cnt += 1
            all_cnt += 1
        elif d and r == 0:
            all_cnt += 1

# accelerate debug output
print("All:", all_cnt, "Success:", succ_cnt)

# save img as gif
all_img[0].save(f"{args.file_path}/rollout.gif", save_all=True, append_images=all_img[1:], duration=100, loop=0)
