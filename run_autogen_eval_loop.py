import argparse
import os
import pickle
import yaml

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import numpy as np
import time
from datetime import datetime
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
from gwt_agent import GWTAutogenAgent
from baseline_agent import BaselineAutogenAgent
import wandb    # Install wandb, use wandb login in cmd, and then run the code


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
        "--gwt",
        action="store_true",
        help="Use the GWTAutogenAgent for evaluation."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # Determine which agent to use
    if args.baseline:
        agent_class = BaselineAutogenAgent
        agent_name = "BaselineAutogenAgent"
    elif args.cognitive:
        agent_class = GWTAutogenAgent
        agent_name = "GWTAutogenAgent"
    else:
        raise ValueError("No agent specified. Use --baseline or --gwt.")
    print(f"Selected Agent: {agent_name}")
    # Load the config file
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    wandb.init(project="cognitive-agents")

    API_KEY = os.environ.get("LAMBDA_API_KEY")
    BASE_URL = "https://api.lambdalabs.com/v1"
    MODEL = "llama3.1-70b-instruct-berkeley"
    llm_config = {
        "timeout": 1000,
        "cache_seed": None,
        "max_tokens": 300,
        "config_list": [{"model": MODEL, "api_key": API_KEY, "base_url": BASE_URL}]}

    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    # run logs
    base_path = "runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(base_path, timestamp)
    os.makedirs(base_path, exist_ok=True)

    result_list_path = os.path.join(base_path, "result_list.txt")
    chat_round_list = []

    for eval_env_type in eval_envs:
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            for eval_path in eval_paths:
                print("Evaluating: %s" % eval_path)
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config,
                                                                                                train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)

                ## For each set, there are `num_games` games we need to evaluate
                num_games = alfred_env.num_games
                success_list = []

                for i in range(num_games):
                    print("Initialized Environment")

                    obs, info = env.reset()
                    agent = agent_class(env, obs, info, llm_config, log_path=base_path, max_actions=35)
                    agent.update_game_no(i)

                    log_paths = agent.get_log_paths()

                    initial_message_content = ""
                    # find the task description in the observation, save it as a txt file.
                    task_description = obs[0].split("Your task is to: ")[1]
                    initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]
                    with open(log_paths['task_path'], "w") as f:
                        f.write(f"Task: {task_description}\n")

                    initial_message_content += f"Task: {task_description}\n"

                    with open(log_paths['history_path'], "w") as f:
                        f.write(f"action: [None] observation: [{initial_observation}]\n")

                    initial_message_content += f"Observation: {initial_observation}\n"

                    admissible_commands = list(info['admissible_commands'][0])
                    # save the addmissible commands into a txt file
                    with open(log_paths['admissible_commands_path'], "w") as f:
                        f.write(f"{admissible_commands}\n")

                    initial_message_content += f"Admissible commands: {admissible_commands}\n"

                    run_chat = True

                    chat_result = None
                    error_message = None

                    print("Run chat")
                    try:
                        chat_result, error_message = agent.run_chat(initial_message_content)
                    except Exception as e:
                        print(f"Group Chat manager fails to chat with error message {e}")
                        error_message = e

                    if error_message is not None:
                        with open(log_paths['error_message_path'], "a") as f:
                            f.write(f"Run Chat: {error_message}\n")

                    if chat_result is not None:
                        if "chat_history" in chat_result.__dict__.keys() and len(chat_result.chat_history) > 0:
                            # message is a list of dictionaries, record every key-value pair into a readable file.
                            # if there is "name" and "role" in the message, record them first.
                            with open(log_paths['chat_history_path'], "w") as f:
                                for message in chat_result.chat_history:
                                    f.write('-' * 20 + '\n')

                                    first_keys = ["name", "role", "content"]

                                    for key in first_keys:
                                        if key in message.keys():
                                            if key == "content":
                                                f.write(f"{key}:\n{message[key]}\n")
                                            else:
                                                f.write(f"{key}: {message[key]}\n")

                                    for key, value in message.items():
                                        if key not in first_keys:
                                            f.write(f"{key}: {value}\n")

                            chat_round_list.append(len(chat_result.chat_history))
                        else:
                            chat_round_list.append(-1)

                            with open(log_paths['chat_history_path'], "w") as f:
                                f.write(f"Error Message: no chat history in chat result\n")

                            print(chat_result)

                    else:
                        chat_round_list.append(-1)

                    # exit()

                    success = agent.success
                    print(f'Success: {success}')
                    success_list.append(success)

                    wandb.log({"success": success, "success_rate": np.sum(success_list) / len(success_list)})

                    # save success and chat_round into a txt file
                    with open(log_paths['result_path'], "w") as f:
                        f.write(f"Success: {success}\n")
                        f.write(f"Chat Round: {chat_round_list[-1]}\n")

                    # save success list and chat_round_list into a txt file
                    with open(result_list_path, "w") as f:
                        f.write(f"Success List: {success_list}\n")
                        f.write(f"Chat Round List: {chat_round_list}\n")

                print(f"Success Rate: {np.sum(success_list)}/{num_games}")

wandb.finish()
