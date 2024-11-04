
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from utils.utils import build_prompt

from accelerate.utils import gather_object, gather
import torch
import time

accelerator = Accelerator()

device_map={"": accelerator.process_index}
model_path = "/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   
tokenizer.pad_token = tokenizer.eos_token

accelerator.wait_for_everyone()

prompts_all = []
for _ in range(32):
    prompt = [
        {"role": "system", "content": "You are a helpful agent."*100},
        {"role": "human", "content": "Reply to this message with Hello World."},
        {"role": "assistant", "content": "Hello World."},
    ]
    prompts_all.append(prompt)

start = time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    
    all_input_ids = []
    all_attention_mask = []
    for prompt in prompts:
        input_ids = build_prompt(tokenizer, prompt).to("cuda")
        attention_mask = torch.ones_like(input_ids).to("cuda")
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
    all_input_ids = torch.cat(all_input_ids, 0)
    all_attention_mask = torch.cat(all_attention_mask, 0)
    
    with torch.no_grad():
        output = model(all_input_ids, attention_mask=all_attention_mask)
    
    timediff = time.time() - start
    
    # print("GPU {}: {} prompts received, generated {} tokens in {} seconds".format(
    #     accelerator.process_index,
    #     len(prompts),
    #     output.logits.shape,
    #     timediff,
    #     ))
    
    all_p = []
    for o in output.logits:
        p = o.log_softmax(-1)[100:101,100].sum().exp()
        all_p.append(p)
    allp_tensor = torch.stack(all_p)
 
results_gathered = accelerator.gather(allp_tensor)

#     gathered_outputs = accelerator.gather(all_p)
    
#     if accelerator.is_main_process:
        
#         all_outputs = [output for output in gathered_outputs]
        
#         # print("all_outputs:", torch.cat(all_outputs, 0).shape)

# # print all_outputs out of the accelerator context
# if all_outputs is not None:
#     print("all_outputs:", torch.cat(all_outputs, 0).shape)