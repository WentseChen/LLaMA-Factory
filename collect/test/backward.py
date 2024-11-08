
from openai import OpenAI

from PIL import Image, ImageDraw, ImageFont

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+"/utils")
from utils import load_text, get_traj
from env import MiniGrid, ActionWrapper, KeyDoorWrapper

font_file = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed.ttf"

api_key = "EMPTY"
api_base = "http://localhost:8000/v1"
model = "/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct"
client = OpenAI(api_key=api_key, base_url=api_base)
def gpt(msgs, return_logprob=False):   
    result = client.chat.completions.create(
        messages=msgs,
        model=model,
        temperature=0.2,
        max_tokens=4096,
    ) 
    return result.choices[0].message.content

system_tmp = load_text("../prompt/policy.txt")
system_msg = {"role": "system", "content": system_tmp}

env = MiniGrid("BabyAI-GoToDoor-v0")
env = ActionWrapper(env)
env = KeyDoorWrapper(env)

obs, info = env.reset()
img = env.render()
img = Image.fromarray(img)

done = False

obs_cache = []
act_cache = []
all_data = {"obs": [], "traj": [], "img": [img]}

while not done:
    
    human_tmp = obs + "\nAfter reading through the rules of the environment, your goal and all the current observation.\nWhat is the next action to achieve your goal?"
    human_msg = {"role": "user", "content": human_tmp}
    msg = [system_msg, human_msg]
    
    obs_cache.append(obs)
    
    action_text = gpt(msg)
    for act_candidate in env.available_actions:
        if act_candidate in action_text:
            action_text = act_candidate
            break
    
    act_cache.append(action_text)
    print("action:", action_text)
    print("="*30)
            
    obs, reward, done, truncated, info = env.step(action_text)
    
    img = env.render()
    img = Image.fromarray(img)
    all_data["img"].append(img)

if reward > 0.:
    result = "Result: You have successfully completed the task!"
else:
    result = "Result: You have failed to complete the task."

system_tmp = load_text("../prompt/reflect1.txt")
system_msg = {"role": "system", "content": system_tmp}

obs_cache.append(obs)
for j in range(len(obs_cache)-1):
    
    all_data["obs"].append(obs_cache[j])
    traj = get_traj(obs_cache[j:], act_cache[j:], result)
    all_data["traj"].append(traj)
    
    human_closer = load_text("../prompt/reflect3.txt")
    human_tmp = traj + human_closer
    human_msg = {"role": "user", "content": human_tmp}
    
    msg = [system_msg, human_msg]
    response = gpt(msg)
    
    if "Verbal feedback:" in response:
        verbal_fb = response.split("Verbal feedback:")[1]
    elif "verbal feedback:\n" in response:
        verbal_fb = response.split("verbal feedback:\n")[1]
    elif "Verbal Feedback:\n" in response:
        verbal_fb = response.split("Verbal Feedback:\n")[1]
    elif "VERBAL FEEDBACK:\n" in response:
        verbal_fb = response.split("VERBAL FEEDBACK:\n")[1]
    else:
        print(response)
        raise ValueError("No verbal feedback")
        
        
    print("verbal feedback:", verbal_fb)
    print("="*30)
    
    img = all_data["img"][j]
    # print verbal feedback on image
    word_per_line = 40
    verbal_fb = [verbal_fb[i:i+word_per_line] for i in range(0, len(verbal_fb), word_per_line)]
    verbal_fb = "\n".join(verbal_fb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_file, 25)
    draw.text((10, 10), verbal_fb, (255, 255, 255), font=font)
    # print observation on image
    obs = obs_cache[j]
    draw.text((10, 400), obs, (255, 255, 255), font=font)
    all_data["img"][j] = img
    
# save gif
all_data["img"][0].save("backward.gif", save_all=True, append_images=all_data["img"][1:], duration=1000, loop=0)

print("reward:", reward)