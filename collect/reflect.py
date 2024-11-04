
import json
import asyncio

import openai
from openai import OpenAI, AsyncOpenAI

from utils.utils import load_text

# get arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct", help='policy model name')
parser.add_argument('--batch_size', type=int, default=256, help='number batch size for each iteration')
parser.add_argument('--mini_batch_size', type=int, default=64, help='number batch size for each iteration')
parser.add_argument('--max_tokens', type=int, default=4096, help='number of max tokens for gpt prompt')
parser.add_argument('--file_path', type=str, default="/zfsauton2/home/wentsec/incontext_RL/test", help='where to save the data')
args = parser.parse_args()

# gpt api setup
api_key = "EMPTY"
api_base = "http://localhost:8000/v1"
client = AsyncOpenAI(api_key=api_key, base_url=api_base)
async def async_gpt(msg, retries=4):
    for attempt in range(retries):
        try:
            completion = await client.chat.completions.create(
                model=args.model_name,
                messages=msg,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt >= retries:
                raise e
async def async_batch_gpt(msgs_list):
    tasks = []
    for msgs in msgs_list:
        tasks.append(async_gpt(msgs["messages"]))
    return await asyncio.gather(*tasks)
def batch_gpt(msgs_list):
    results = asyncio.run(async_batch_gpt(msgs_list))
    return results

# load data from json
file_name = args.file_path + "/phase1.json"
with open(file_name, "r") as f:
    all_input_data = json.load(f)

all_output_data = {
    "obs": [],
    "sampling_logp": [],
    "verbal_feedback": [],
}


# get feedback prompt
system_msg = load_text("prompt/reflect1.txt")
system_msg = {"role": "system", "content": system_msg}
human_msg_opener = load_text("prompt/reflect2.txt")
human_msg_closer = load_text("prompt/reflect3.txt")


gpt_cache = []
idx, last_idx = 0, 0
checked = [False for _ in range(len(all_input_data["traj"]))]
while len(all_output_data["verbal_feedback"]) <= args.batch_size:
    
    traj = all_input_data["traj"][idx]
    human_msg = human_msg_opener + traj + human_msg_closer
    human_msg = {"role": "user", "content": human_msg}
    msg = [system_msg, human_msg]
    
    if not checked[idx]:
        gpt_cache.append({
            "idx": idx,
            "messages": msg,
        })
        last_idx = idx
    
    idx = (idx + 1) % len(all_input_data["traj"])
    
    if len(gpt_cache) == args.mini_batch_size or last_idx == idx:
        
        # get GPT response
        num_retry = 0
        while num_retry < 4:
            try:
                responses = batch_gpt(gpt_cache)
                break
            except Exception as e:
                num_retry += 1
        assert num_retry < 4, "Failed to get GPT response"

        # check format, save traj
        for i in range(len(responses)):
            if responses[i] is None:
                continue
            data_idx = gpt_cache[i]["idx"]
            format_correct = "<feedback you should follow>" in responses[i]
            # format_correct = format_correct and "</feedback you should follow>" in responses[i]
            if format_correct:
                verbal_fb = responses[i].split("<feedback you should follow>")[1]
                if "</feedback you should follow>" in verbal_fb:
                    verbal_fb = verbal_fb.split("\n</feedback you should follow>")[0]
                checked[data_idx] = True
                all_output_data["obs"].append(all_input_data["obs"][data_idx])
                all_output_data["sampling_logp"].append(all_input_data["sampling_logp"][data_idx])
                all_output_data["verbal_feedback"].append(verbal_fb)

        print("checked:", sum(checked))
        gpt_cache = []

# save all_output_data as a json
file_name = args.file_path + "/phase2.json"
with open(file_name, "w") as f:
    json.dump(all_output_data, f)




