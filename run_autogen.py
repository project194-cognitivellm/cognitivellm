import numpy as np
import os
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from agent import AutogenAgent

# Load config
config = generic.load_config()

eval_paths = config["general"]["evaluate"]["eval_paths"]
eval_envs = config["general"]["evaluate"]["envs"]
controllers = config["general"]["evaluate"]["controllers"]
repeats = config["general"]["evaluate"]["repeats"]

# Iterate through all environments
for eval_env_type in eval_envs:
    # Iterate through all controllers
    for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
        print("Setting controller: %s" % controller_type)
        # Iterate through all splits
        for eval_path in eval_paths:
            print("Evaluating: %s" % eval_path)
            config["general"]["evaluate"]["env"]["type"] = eval_env_type
            config["dataset"]["eval_ood_data_path"] = eval_path
            config["controller"]["type"] = controller_type

            alfred_env = getattr(
                environment,
                config["general"]["evaluate"]["env"]["type"]
            )(config, train_eval="eval_out_of_distribution")
            env = alfred_env.init_env(batch_size=1)

            # For each set, there are num_games games we need to evaluate
            num_games = alfred_env.num_games

            # We need to set a max number of steps for each game. Refer to the ALFWorld paper?
            max_steps = 100

            success_list = []

            llm_config = {
                "config_list": [
                    {
                        "model": "gpt-4o-mini",
                        "api_key": os.environ.get("OPENAI_API_KEY")
                    }
                ]
            }

            for i in range(num_games):
                print("Initialized Environment")

                # Interact
                obs, info = env.reset()
                print("Reset environment")
                print(f"Admissible Commands: {info['admissible_commands'][0]}")

                autogen_agent = AutogenAgent(env, obs, info, llm_config)

                print("Initialization complete; Starting Chat")

                chat_result = autogen_agent.run_chat()
                print("Finished Chat")

                success = "SUCCESS" in chat_result.chat_history[-1]['content']

                success_list.append(success)

            print(f"Success Rate: {np.sum(success_list)}/{num_games}")
