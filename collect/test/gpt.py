
from openai import OpenAI

api_key = "EMPTY"
api_base = "http://localhost:8000/v1"
model = "/zfsauton2/home/wentsec/Meta-Llama-3.1-8B-Instruct"
client = OpenAI(api_key=api_key, base_url=api_base)
def gpt(msgs, return_logprob=False):   
    result = client.chat.completions.create(
        messages=msgs,
        model=model,
        temperature=0.5,
        max_tokens=1024,
    ) 
    return result.choices[0].message.content

system_text = """
You are a helpful navigation agent in a 2D grid world with these rules:
You cannot move onto a cell if there is an item on it.
You can not move out of the square between (x=0, y=0) to (x=6, y=6).

Available actions are:
- move in the positive x-direction
- move in the positive y-direction
- move in the negative x-direction
- move in the negative y-direction
- pick up an item in the positive x-direction
- pick up an item in the positive y-direction
- pick up an item in the negative x-direction
- pick up an item in the negative y-direction
- drop an item in the positive x-direction
- drop an item in the positive y-direction
- drop an item in the negative x-direction
- drop an item in the negative y-direction
- toggle a door in the positive x-direction
- toggle a door in the positive y-direction
- toggle a door in the negative x-direction
- toggle a door in the negative y-direction

I will provide you with your goal and the current state.
You should then respond to me with the next action you should take to achieve your goals.
The action needs to be in available actions.

Desired format: 
REASONING PATH
My next action is to YOUR ACTION.

"""
human_text = """
Your goal is to go to a purple door
You see a locked purple door at (x=0, y=1).
You see a closed green door at (x=2, y=6).
You see a closed grey door at (x=4, y=0).
You are at (x=5, y=1), facing down.
You see a locked purple door at (x=6, y=3).
After reading through the rules of the environment, your goal and all the current observation. 
What is the next action to achieve your goal?
"""

test_msg = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": human_text}
]
result = gpt(test_msg)

print(result)