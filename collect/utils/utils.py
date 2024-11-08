
import time
import torch
import numpy as np

def build_prompt(tokenizer, msgs):
    
    system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>'
    
    all_msg = ""
    for msg in msgs:
        role, content = msg['role'], msg['content']
        if role == 'system':
            role_message = system_format.format(content=content)
        elif role == 'user':
            role_message = user_format.format(content=content)
        else:
            role_message = assistant_format.format(content=content)
        all_msg += role_message
    input_ids = tokenizer.encode(all_msg, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    return input_ids

def get_assistant_index(input_ids):

    length = input_ids.shape[-1]
    # [x, 1828, 1957, 374, x,] = My next action is to
    index_begin = torch.where(input_ids == 1828)[0][-1] + 4
    # [128009] = <|eot_id|>
    index_end = torch.where(input_ids == 128009)[0][-1]
    return index_begin.item(), index_end.item()

def llm_logprob(tokenizer, model, input_msg, temperature=0.2, node_id="26"):
    
    input_ids = build_prompt(tokenizer, input_msg)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        output = model(
            input_ids, 
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool)
        )
    
    start_idx, end_idx = get_assistant_index(input_ids.flatten())
    logits = output.logits / temperature 
    logits = logits[0, start_idx-1]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    action_log_probs = []
    for token_id in [5530, 6107, 27014, 38053, 20463, 19431]:
        action_log_probs.append(log_probs[token_id])
    all_probs = torch.stack(action_log_probs).exp()
    remain_prob = 1 - all_probs.sum()
    action_log_probs.append(remain_prob.log())
    action_log_probs = torch.stack(action_log_probs)
    
    return action_log_probs

def deepcopy_list(list_msgs):
    list_msgs_copy = []
    for msg in list_msgs:
        list_msgs_copy.append({"role": msg["role"], "content": msg["content"]})
    return list_msgs_copy

def response2action(response):
    if "<next action>" in response:
        text_action = response.split("<next action>\n")[1]
        text_action = text_action.split("\n</next action>")[0]
    elif "<action>" in response:
        text_action = response.split("<action>\n")[1]
        text_action = text_action.split("\n</action>")[0]
    else:
        text_action = response.split("Action: ")[1]
    if text_action[-1] == ".":
        text_action = text_action[:-1]
    return text_action

def load_text(fpaths, by_lines=False):
    with open(fpaths, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()

def get_text_action(model, tokenizer, msgs, temperature=0.2):
    
    input_msg = deepcopy_list(msgs)
    input_msg.append( {"role": "assistant", "content": "<next action>\nForward.\n</next action>\n"} )
    
    input_ids = build_prompt(tokenizer, input_msg)
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        output = model(
            input_ids, 
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool)
        )
    
    start_idx, end_idx = get_assistant_index(input_ids.flatten())
    logits = output.logits / temperature 
    logits = logits[0, start_idx-1]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    action_log_probs = []
    for token_id in [5530, 6107, 27014, 38053, 20463, 19431]:
        action_log_probs.append(log_probs[token_id])
    all_probs = torch.stack(action_log_probs).exp()
    norm_probs = all_probs / all_probs.sum()
    sample_action = np.random.choice(len(norm_probs), p=norm_probs.cpu().numpy())
    sample_action_text = env.all_actions[sample_action]
    
    return sample_action_text

def get_traj(all_obs, all_act, result):
    # should move this to env wrapper
    
    # get goal
    split_obs = all_obs[0].split("\n")
    traj = split_obs[0] + "\n"
    traj += "State t\n"
    traj += "\n".join(split_obs[1:])
    
    for idx, (obs, act) in enumerate(zip(all_obs[1:], all_act)):
        
        traj += "You took action " + act + "\n"
        split_obs = obs.split("\n")
        obs = ""
        for o in split_obs:
            if "door at" in o:
                continue
            elif "key at" in o:
                continue
            elif "Your goal is to" in o:
                continue
            obs += o + "\n"
        traj += "State t+{}\n".format(idx+1) + obs
        
    traj += result
        
    return traj


