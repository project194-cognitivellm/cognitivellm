import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=3)

# interact
obs, info = env.reset()
# the initial obs would include the task description. It can serve as the system message to LLM agent.
# example: ('-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. 
# Looking quickly around you, you see a armchair 2, a armchair 1, a cabinet 4, a cabinet 3, 
# a cabinet 2, a cabinet 1, a coffeetable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, 
# a dresser 1, a garbagecan 1, a sidetable 1, and a sofa 1.
# \n\nYour task is to: put two pillow in armchair.',)


count = 1
while True:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    random_actions = [np.random.choice(admissible_commands[0])]
    
    
    # step
    obs, scores, dones, infos = env.step(random_actions)
    print("Action: {}, Scores: {}, Obs: {}".format(random_actions[0], scores[0], obs[0]))
    
    count += 1
    if count > 10:
        break