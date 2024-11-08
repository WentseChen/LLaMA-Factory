

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("no-sandbox")
chrome_options.binary_location = "/usr/bin/google-chrome"
service = Service("/zfsauton2/home/wentsec/tmp/chromedriver-linux64/chromedriver") 
driver = webdriver.Chrome(service=service, options=chrome_options)

# import time
# import gymnasium
# import miniwob
# from miniwob.action import ActionTypes

# gymnasium.register_envs(miniwob)
# env = gymnasium.make('miniwob/click-test-2-v1', render_mode=None)

# try:

#     observation, info = env.reset()
#     assert observation["utterance"] == "Click button ONE."
#     assert observation["fields"] == (("target", "ONE"),)
#     time.sleep(2)       # Only here to let you look at the environment.
    
#     # Find the HTML element with text "ONE".
#     for element in observation["dom_elements"]:
#         if element["text"] == "ONE":
#             break

#     # Click on the element.
#     action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
#     observation, reward, terminated, truncated, info = env.step(action)

#     # Check if the action was correct. 
#     print(reward)      # Should be around 0.8 since 2 seconds has passed.
#     assert terminated is True
#     time.sleep(2)

# finally:
    
#     env.close()
