import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

import os

# load config
config = generic.load_config()
eval_path = config["general"]["evaluate"]["eval_paths"][0]
eval_env_type = config["general"]["evaluate"]["envs"][0]
if eval_env_type == "AlfredThorEnv":
    controller_type = config["general"]["evaluate"]["controllers"][0]
else:
    controller_type = "tw"

config["general"]["evaluate"]["env"]["type"] = eval_env_type
config["dataset"]["eval_ood_data_path"] = eval_path
config["controller"]["type"] = controller_type

# setup environment
env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval='eval_out_of_distribution')
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
    print("scores: ", scores)
    print("dones: ", dones)
    if dones[0]:
        print("Resetting environment")
        obs, info = env.reset()
