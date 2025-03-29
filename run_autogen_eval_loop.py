import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import pickle
import random
import yaml
import time
import numpy as np
from datetime import datetime

# External packages
import wandb  # Make sure to run `wandb login` beforehand

# ALFWorld & Agents
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
from gwt_agent import GWTAutogenAgent
from baseline_agent import BaselineAutogenAgent
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager

global_num_games_to_evaluate = 139
global_max_actions_per_game = 60
global_rounds_per_game = 2

def parse_arguments():
    """
    Parse command-line arguments for evaluating Autogen Agents on the ALFWorld environment.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate different Autogen Agents on the ALFWorld environment."
    )
    parser.add_argument("config_file", help="Path to the YAML config file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true", help="Use BaselineAutogenAgent")
    group.add_argument("--gwt", action="store_true", help="Use GWTAutogenAgent")

    parser.add_argument("--long_term_guidance", action="store_true", help="Enable long-term guidance")

    return parser.parse_args()


def setup_environment_and_memory():
    """
    Set up directories and memory files required for the evaluation run.
    Returns paths to memory1.txt and memory2.txt.
    """
    # Create run-specific output directory with timestamp
    base_path = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(base_path, exist_ok=True)

    # Ensure memory directory and memory files exist
    memory_path = "memory"
    os.makedirs(memory_path, exist_ok=True)

    memory_path1 = os.path.join(memory_path, "memory1.txt")
    memory_path2 = os.path.join(memory_path, "memory2.txt")

    for path in [memory_path1, memory_path2]:
        if not os.path.exists(path):
            with open(path, 'w') as f:
                pass  # Create an empty file

    return base_path, memory_path1, memory_path2


if __name__ == "__main__":
    args = parse_arguments()

    # Select the agent class
    if args.baseline:
        agent_class = BaselineAutogenAgent
        agent_name = "BaselineAutogenAgent"
    elif args.gwt:
        agent_class = GWTAutogenAgent
        agent_name = "GWTAutogenAgent"
    else:
        raise ValueError("No agent specified. Use --baseline or --gwt.")

    print(f"Selected Agent: {agent_name}")

    # Load config
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    # Initialize Weights & Biases
    wandb.init(
        project="cognitive_agents",
        entity="eduardocortes1100-university-of-california-berkeley")

    # Setup memory and output directories
    base_path, memory_path1, memory_path2 = setup_environment_and_memory()
    result_list_path = os.path.join(base_path, "result_list.txt")

    # Setup API key
    API_KEY = os.environ.get("BLOCK_KEY")
    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": API_KEY}]}

    # Extract evaluation parameters
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    chat_round_list = []

    for eval_env_type in eval_envs:
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            for eval_path in eval_paths:
                print(f"Evaluating: {eval_path}")

                # Configure the evaluation environment
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                env_class = getattr(environment, eval_env_type)
                alfred_env = env_class(config, train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)
                num_games = alfred_env.num_games

                # Random selection of evaluation games
                if global_rounds_per_game > num_games:
                    num_games_to_evaluate = num_games
                else:
                    num_games_to_evaluate = global_num_games_to_evaluate

                selected_games = sorted(random.sample(range(1, num_games + 1), num_games_to_evaluate))
                #selected_games = [7, 8, 15, 25, 30]
                #num_games_to_evaluate = len(selected_games)
                print(f"Selected {num_games_to_evaluate} Games: {selected_games}")

                result_list = []
                error_list = []
                num_games_evaluated = 0

                # Track metrics
                cumulative_actions = 0

                for i in range(1, num_games + 1):
                    obs, info = env.reset()

                    if i not in selected_games:
                        print(f"Skipped Game #{i}")
                        continue

                    print(f"\n[Running Game #{i}]")
                    num_games_evaluated += 1
                    print(f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}")
                    agent = agent_class(
                        env, obs, info, llm_config,
                        log_path=base_path,
                        memory_path1=memory_path1,
                        memory_path2=memory_path2,
                        game_no=i,
                        max_actions=global_max_actions_per_game,
                        rounds_per_game=global_rounds_per_game,
                        args=args
                    )

                    log_paths = agent.get_log_paths()

                    # Log task description and initial observation
                    task_description = obs[0].split("Your task is to: ")[1]
                    initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]

                    with open(log_paths['task_path'], "w") as f:
                        f.write(f"Task: {task_description}\n")

                    with open(log_paths['history_path'], "w") as f:
                        f.write(f"action: 'None'. observation: '{initial_observation}'\n")

                    with open(log_paths['admissible_commands_path'], "w") as f:
                        f.write(f"{list(info['admissible_commands'][0])}\n")

                    initial_message = (
                        "You and all other Agents are collectively a singular conscious entity named ALFRED. " +
                        agent.obs[0] +
                        f"\nTask Status: INCOMPLETE\nActions Left: {agent.max_actions - agent.num_actions_taken}" +
                        f"\nCurrent Admissible Actions: {list(agent.info['admissible_commands'][0])}"
                    )

                    wandb.log({
                        "game_no": i,
                    }, step=num_games_evaluated)

                    start_time = time.time()
                    try:
                        chat_result, error_message = agent.run_chat(initial_message)
                    except Exception as e:
                        error_message = str(e)
                        chat_result = None
                        print(f"Chat Error: {error_message}")
                    end_time = time.time()

                    # Log errors
                    if error_message:
                        error_list.append(i)
                        with open(log_paths['error_message_path'], "a") as f:
                            f.write(f"Run Chat: {error_message}\n")

                    # Log chat history
                    if chat_result and getattr(chat_result, "chat_history", []):
                        with open(log_paths['chat_history_path'], "w") as f:
                            for message in chat_result.chat_history:
                                f.write('-' * 20 + '\n')
                                for key in ["name", "role", "content"]:
                                    if key in message:
                                        f.write(f"{key}:\n{message[key]}\n" if key == "content" else f"{key}: {message[key]}\n")
                                for k, v in message.items():
                                    if k not in ["name", "role", "content"]:
                                        f.write(f"{k}: {v}\n")
                        chat_round_list.append(len(chat_result.chat_history))
                    else:
                        chat_round_list.append(-1)
                        with open(log_paths['chat_history_path'], "w") as f:
                            f.write("Error Message: no chat history in chat result\n")

                    # Evaluate and log success
                    success = agent.success
                    result_list.append(success)
                    success_rate = np.sum(result_list) / num_games_evaluated
                    elapsed_time = end_time - start_time
                    if success:
                        cumulative_actions += agent.num_actions_taken
                    avg_actions_taken_per_successful_game = cumulative_actions / np.sum(result_list)

                    wandb.log({
                        "success": int(success),
                        "actions_taken": agent.num_actions_taken,
                        "success_rate": success_rate,
                        "avg_actions_taken_per_successful_game": avg_actions_taken_per_successful_game,
                        "runtime": elapsed_time
                    }, step=num_games_evaluated)


                    print(f"[Ran Game #{i}]")
                    print(f"Evaluation {num_games_evaluated} of {num_games_to_evaluate}")
                    print(f"Success: {success}")
                    print(f"Rounds Taken: {global_rounds_per_game - agent.rounds_left} out of {global_rounds_per_game}")
                    print(f"Actions Taken: {agent.num_actions_taken} out of {global_max_actions_per_game}")
                    print(f"Success Rate: {np.sum(result_list)}/{num_games_evaluated} = {success_rate * 100:.2f}%")
                    print(f"Average Actions per Successful Game: {avg_actions_taken_per_successful_game:.2f} out of {global_max_actions_per_game}")
                    print(f"Failures: {[j+1 for j, val in enumerate(result_list) if not val and (j+1) not in error_list]}")
                    print(f"Errors: {error_list}")
                    print(f"Error-Adjusted Success Rate: {np.sum(result_list)}/{num_games_evaluated - len(error_list)} = {np.sum(result_list) * 100/ (num_games_evaluated - len(error_list)):.2f}%")
                    print(f"Remaining Games: {selected_games[num_games_evaluated:]}\n")

                    if not selected_games[num_games_evaluated:]:
                        break

                    # Save result for this game
                    with open(log_paths['result_path'], "w") as f:
                        f.write(f"Success: {success}\n")
                        f.write(f"Chat Round: {chat_round_list[-1]}\n")

                    with open(result_list_path, "w") as f:
                        f.write(f"Success List: {result_list}\n")
                        f.write(f"Chat Round List: {chat_round_list}\n")

                # Final Success Summary
                print(f"Final Success Rate: {np.sum(result_list)}/{num_games_to_evaluate}")
                print(f"Final Error-Adjusted Success Rate: {np.sum(result_list)}/{num_games_to_evaluate - len(error_list)}")

    wandb.finish()
