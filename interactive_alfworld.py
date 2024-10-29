import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

import os

# load config
config = generic.load_config()
env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)
print("Initialized Environment")

# interact
obs, info = env.reset()

action = ''
while True:
    print(obs[0])
    print(f"Admissible Commands: {info['admissible_commands'][0]}")
    action = input("Enter action: ")
    if action == 'exit':
        break
    print(f"Action: {action}")
    obs, scores, dones, info = env.step([action])
