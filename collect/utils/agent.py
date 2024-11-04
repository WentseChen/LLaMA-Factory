
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import load_text, build_prompt, get_assistant_index

class LlamaAgent():
    
    def __init__(self, model, tokenizer, accelerator, env, system_tmp):
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.env = env
        self.system_msg = {"role": "system", "content": system_tmp}
      
    def _build_prompt(self, obs):
        """
        description:
        function to convert batch of text-form messages to tensor-form input_ids
        each message will be paired with n action messages, where n is the number of available actions
        input:
        batch_msg: list of list of dict
        output:
        batch_input_ids: torch.Tensor, padding on the right
        batch_attention_mask: torch.Tensor
        """
        
        # builtd prompt
        input_ids_list = []
        for o in obs:
            human_msg = {"role": "user", "content": o}
            for act in self.env.available_actions:
                a_str = "<next action>\n" + act + "\n</next action>\n"
                action_msg = {"role": "assistant", "content": a_str}
                input_msg = [self.system_msg, human_msg, action_msg]
                input_ids = build_prompt(self.tokenizer, input_msg)
                input_ids_list.append(input_ids)
                
    def _fast_build_prompt(self, obs):
        """
        description:
        function to convert batch of text-form messages to tensor-form input_ids
        each message will be paired with n action messages, where n is the number of available actions
        input:
        batch_msg: list of list of dict
        output:
        batch_input_ids: torch.Tensor, padding on the right
        batch_attention_mask: torch.Tensor
        """
        
        # builtd prompt
        input_ids_list = []
        for o in obs:
            human_msg = {"role": "user", "content": o+"After reading through the rules of the environment, your goal and all the current observation. What is the next action to achieve your goal?"}
            for act in self.env.available_actions[::4]:
                a_str = "<next action>\n" + act + "\n</next action>\n"
                action_msg = {"role": "assistant", "content": a_str}
                input_msg = [self.system_msg, human_msg, action_msg]
                input_ids = build_prompt(self.tokenizer, input_msg)
                input_ids_list.append(input_ids)
        
        # padding
        max_len = max([len(ids[0]) for ids in input_ids_list])
        batch_input_ids = torch.zeros((len(input_ids_list), max_len)).long().to(self.model.device)
        batch_attention_mask = torch.ones((len(input_ids_list), max_len)).long().to(self.model.device)
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i][0])
            batch_input_ids[i][:len(input_ids_list[i][0])] = input_ids_list[i][0]
            batch_attention_mask[i][len(input_ids_list[i][0]):] = 0
        
        return batch_input_ids, batch_attention_mask

    def _decode_action(self, output, batch_input_ids, temperature=0.2):
        """
        description:
        function to decode action logp from model output
        input:
        output: torch.Tensor
        batch_input_ids: torch.Tensor
        temperature: float
        output:
        all_log_probs: torch.Tensor
        """
        
        all_log_probs = []
        for i in range(len(batch_input_ids)):
            logits = output.logits[i] / temperature
            input_ids = batch_input_ids[i].flatten()
            start_idx, end_idx = get_assistant_index(input_ids)
            action_logits = logits[start_idx-1:end_idx-1] # output_token leads input_token by 1 step
            action_logp = torch.nn.functional.log_softmax(action_logits, dim=-1)
            selected_idx = input_ids[start_idx:end_idx]
            selected_logp = torch.gather(action_logp, -1, selected_idx.unsqueeze(-1))
            selected_logp = selected_logp.sum()
            all_log_probs.append(selected_logp)
        all_log_probs = torch.stack(all_log_probs)
        
        return all_log_probs

    def _fast_decode_action(self, output, batch_input_ids, temperature=0.2):
        """
        decode, but with less computation
        """
        
        all_log_probs = []
        for i in range(len(batch_input_ids)):
            logits = output.logits[i] / temperature
            input_ids = batch_input_ids[i].flatten()
            start_idx, end_idx = get_assistant_index(input_ids)
            action_logits = logits[start_idx-1:end_idx-1]
            action_logp = torch.nn.functional.log_softmax(action_logits, dim=-1)
            selected_log_probs = []
            for token_id in [1314, 709, 2163, 1523]: # [right, up, left, down]
                selected_idx = input_ids[start_idx:end_idx]
                selected_idx[-1] = token_id
                selected_logp = torch.gather(action_logp, -1, selected_idx.unsqueeze(-1))
                selected_logp = selected_logp.sum()
                selected_log_probs.append(selected_logp)
            selected_log_probs = torch.stack(selected_log_probs)
            all_log_probs.append(selected_log_probs)
        all_log_probs = torch.stack(all_log_probs)
        
        return all_log_probs

    def __call__(self, obs, temperature=0.2, fast=False):
        """
        description:
        function to generate action based on observation
        input:
        obs: list of str
        output:
        sample_action_text: list of str
        """
        
        self.accelerator.wait_for_everyone()
        
        with self.accelerator.split_between_processes(obs) as obs:
            
            if fast:
                batch_input_ids, batch_attention_mask = self._fast_build_prompt(obs)
            else:
                batch_input_ids, batch_attention_mask = self._build_prompt(obs)
            
            # forward
            # Note: by default, we will use model_parallel when setting device_map='auto'
            # todo: add support for data_parallel to speed up
            with torch.no_grad():
                
                output = self.model(
                    batch_input_ids.to("cuda"), 
                    attention_mask=batch_attention_mask.to("cuda")
                )
            
            if fast:
                all_log_probs = self._fast_decode_action(output, batch_input_ids, temperature)
            else:
                all_log_probs = self._decode_action(output, batch_input_ids, temperature)
                
            all_log_probs = all_log_probs.view(len(obs), -1)
            all_probs = all_log_probs.exp()
            norm_probs = all_probs / all_probs.sum(dim=-1, keepdim=True)
            
        probs = self.accelerator.gather(norm_probs)
        sample_action = torch.distributions.Categorical(probs)
        
        return sample_action
        
        

if __name__ == "__main__":
    
    from llamafactory.collect.env import MiniGrid, ActionWrapper, KeyDoorWrapper
    from llamafactory.collect.venv import vEnv
    import time
    begin = time.time()
    
    from PIL import Image
    
    # load model
    model_name = "/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', torch_dtype=torch.bfloat16
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
    env_num = 16
    env = vEnv(env_num, env_func, scenario="BabyAI-UnlockLocal-v0")
    
    # load agent 
    agent = LlamaAgent(model, tokenizer, env)
    
    # prepare env
    all_imgs = []
    obs, info = env.reset()
    img = env.render()
    img = Image.fromarray(img)
    all_imgs.append(img)
    done = [False for _ in range(env_num)]

    # rollout
    while not done[0]:
        
        act_dist = agent(obs, fast=True)
        sample_action = act_dist.sample()
        sample_action_text = [env.available_actions[i] for i in sample_action]
        obs, reward, done, truncated, info = env.step(sample_action_text)
        img = env.render()
        img = Image.fromarray(img)
        all_imgs.append(img)
        
    # save imgs
    all_imgs[0].save("test.gif", save_all=True, append_images=all_imgs[1:], duration=200, loop=0)
        
