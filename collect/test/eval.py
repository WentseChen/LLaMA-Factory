
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+"/utils")
from utils import load_text
from env import MiniGrid, ActionWrapper, KeyDoorWrapper

api_key = "EMPTY"
api_base = "http://localhost:8000/v1"
model = "/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct"
client = OpenAI(api_key=api_key, base_url=api_base)
def gpt(msgs, return_logprob=False):   
    result = client.chat.completions.create(
        messages=msgs,
        model=model,
        temperature=0.2,
        max_tokens=1024,
    ) 
    return result.choices[0].message.content

system_tmp = load_text("../prompt/policy_system.txt")
system_msg = {"role": "system", "content": system_tmp}
human_tmp = load_text("../prompt/policy_human.txt")

env = MiniGrid("BabyAI-GoToDoor-v0")
env = ActionWrapper(env)
env = KeyDoorWrapper(env)

succ_cnt = 0

for epi_num in range(100):
    
    obs, info = env.reset()
    # img = env.render()
    # img = Image.fromarray(img)
    # all_img = [img]
    done = False

    while not done:
        
        human_msg = {"role": "user", "content": obs + human_tmp}
        msg = [system_msg, human_msg]
        
        action_text = gpt(msg)
        if "My next action is to " in action_text:
            action_text = action_text.split("My next action is to ")[1]
        
        for act_candidate in env.available_actions:
            if act_candidate in action_text:
                action_text = act_candidate
            elif "move up" in action_text:
                action_text = "move in the positive y-direction"
            elif "move down" in action_text:
                action_text = "move in the negative y-direction"
            elif "move left" in action_text:
                action_text = "move in the negative x-direction"
            elif "move right" in action_text:
                action_text = "move in the positive x-direction"
            
                
        obs, reward, done, truncated, info = env.step(action_text)
        
        # img = env.render()
        # img = Image.fromarray(img)
        # all_img.append(img)
        

    if reward > 0.:
        succ_cnt += 1
    # # save imgs
    # all_img[0].save("forward.gif", save_all=True, append_images=all_img[1:], duration=100, loop=0)
    print("success rate:", succ_cnt/(epi_num+1))