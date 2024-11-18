import numpy as np
import os
import yaml
import argparse
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from baseline_agent import BaselineAutogenAgent
from cognitive_agent import CognitiveAutogenAgent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate different Autogen Agents on the ALFWorld environment."
    )
    parser.add_argument("config_file", help="path to config file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--baseline",
        action="store_true",
        help="Use the BaselineAutogenAgent for evaluation."
    )
    group.add_argument(
        "--cognitive",
        action="store_true",
        help="Use the CognitiveAutogenAgent for evaluation."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine which agent to use
    if args.baseline:
        agent_class = BaselineAutogenAgent
        agent_name = "BaselineAutogenAgent"
    elif args.cognitive:
        agent_class = CognitiveAutogenAgent
        agent_name = "CognitiveAutogenAgent"
    else:
        raise ValueError("No agent specified. Use --baseline or --cognitive.")

    print(f"Selected Agent: {agent_name}")

    # Load the config file
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    # Iterate through all environments
    for eval_env_type in eval_envs:
        # Iterate through all controllers
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            print(f"Setting controller: {controller_type}")
            # Iterate through all splits
            for eval_path in eval_paths:
                print(f"Evaluating: {eval_path}")
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                # Initialize the environment
                alfred_env = getattr(
                    environment,
                    config["general"]["evaluate"]["env"]["type"]
                )(config, train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)

                # For each set, there are num_games games we need to evaluate
                num_games = alfred_env.num_games

                # We need to set a max number of steps for each game
                max_steps = 100  # Adjust as needed based on ALFWorld paper

                success_list = []

                # LLM configuration
                llm_config = {
                    "config_list": [
                        {
                            "model": "gpt-4o-mini",
                            "api_key": os.environ.get("OPENAI_API_KEY")
                        }
                    ]
                }

                for game_idx in range(num_games):
                    print(f"\n=== Game {game_idx + 1}/{num_games} ===")
                    print("Initialized Environment")

                    # Reset environment
                    obs, info = env.reset()
                    print("Reset environment")
                    print(f"Admissible Commands: {info['admissible_commands'][0]}")

                    # Instantiate the selected agent
                    agent = agent_class(env, obs, info, llm_config)

                    print("Initialization complete; Starting Chat")

                    # Run the agent's chat
                    chat_result = agent.run_chat()
                    print("Finished Chat")

                    # Check for success
                    if chat_result.chat_history:
                        last_message = chat_result.chat_history[-1].get('content', "")
                        success = "SUCCESS" in last_message
                        print(f"Game Success: {success}")
                        success_list.append(success)
                    else:
                        print("No chat history found. Marking as failure.")
                        success_list.append(False)

                # Calculate and print success rate
                total_success = np.sum(success_list)
                print(f"\n=== Evaluation Summary for {agent_name} ===")
                print(f"Success Rate: {total_success}/{num_games} ({(total_success / num_games) * 100:.2f}%)\n")


if __name__ == "__main__":
    main()
