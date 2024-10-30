import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()

eval_paths = config["general"]["evaluate"]["eval_paths"]
eval_envs = config["general"]["evaluate"]["envs"]
controllers = config["general"]["evaluate"]["controllers"]
repeats = config["general"]["evaluate"]["repeats"]

# iterate through all environments
for eval_env_type in eval_envs:
    # iterate through all controllers
    for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
        print("Setting controller: %s" % controller_type)
        # iterate through all splits
        for eval_path in eval_paths:
            print("Evaluating: %s" % eval_path)
            config["general"]["evaluate"]["env"]["type"] = eval_env_type
            config["dataset"]["eval_ood_data_path"] = eval_path
            config["controller"]["type"] = controller_type
            
            alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
            eval_env = alfred_env.init_env(batch_size=1)
            
            
            # for each set, there are num_games games we need to evaluate
            num_games = alfred_env.num_games
            
            
            # We need to set a max number of steps for each game. Refer to the ALFWorld paper? 
            max_steps = 100
            
            success_list = []
            
            for i in range(num_games):
                
                # Once we reset the environment, a new game starts
                obs, info = eval_env.reset()
                
                success = False
                for j in range(max_steps):
                    print(obs)
                    print(f"Admissible Commands: {info['admissible_commands'][0]}")
                    action = input("Enter action: ")
                    if action == 'exit':
                        break
                    # print(f"Action: {action}")
                    obs, scores, dones, info = eval_env.step([action])
                    print(f"Scores: {scores[0]}", f"Dones: {dones[0]}")
                    if dones[0]:
                        success = True
                        break
                success_list.append(success)
            
            print(f"Success Rate: {np.sum(success_list)}/{num_games}")
            






