import os
import pickle

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import numpy as np
import time
from datetime import datetime
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
from gwt_agent import GWTAutogenAgent

# load config
config = generic.load_config()
API_KEY = os.environ.get("OPENAI_API_KEY")

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

llm_config = {
    "timeout": 1000,
    "cache_seed": None,
    # "temperature": 1,
    "max_tokens": 300,
    "config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}]}

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
            max_steps = 100
            success_list = []

            num_games = 50
            for i in range(num_games):
                print("Initialized Environment")

                obs, info = env.reset()
                agent = GWTAutogenAgent(env, obs, info, llm_config, log_path=base_path)
                agent.update_game_no(i)

                log_paths = agent.get_log_paths()

                initial_message_content = ""
                # find the task description in the observation, save it as a txt file.
                task_description = obs[0].split("Your task is to: ")[1]
                initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]
                with open(log_paths['task_path'], "w") as f:
                    f.write(f"Task: {task_description}\n")

                initial_message_content += f"Task: {task_description}\n"

                with open(log_paths['history'], "w") as f:
                    f.write(f"action: [None] observation: [{initial_observation}]\n")

                initial_message_content += f"Observation: {initial_observation}\n"

                admissible_commands = list(info['admissible_commands'][0])
                # save the addmissible commands into a txt file
                with open(log_paths['admissible_commands_path'], "w") as f:
                    f.write(f"{admissible_commands}\n")

                initial_message_content += f"Addmissible commands: {admissible_commands}\n"

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

                print("Resume chat")
                max_num_of_resume = 5

                if chat_result is None:
                    for i in range(max_num_of_resume):
                        time.sleep(10)

                        with open(log_paths['message_path'], "rb") as f:
                            last_message = pickle.load(f)

                        remove_index = 0
                        for j in range(len(last_message)):
                            if last_message[-j - 1]['role'] == 'user':
                                if last_message[-j - 1]['name'] == 'chat_manager' or last_message[-j - 1][
                                    'name'] == 'Task_Agent':
                                    remove_index = - j
                                    break

                        last_message = last_message[:len(last_message) + remove_index]

                        chat_result, error_message = agent.resume_chat(last_message)

                        if chat_result is not None:
                            break

                        if error_message is not None:
                            with open(log_paths['error_message_path'], "a") as f:
                                f.write(f"Resume Chat {i + 1}: {error_message}\n")

                if chat_result is not None:
                    if "chat_history" in chat_result.__dict__.keys() and len(chat_result.chat_history) > 0:

                        # Two cases: if last message is tool calls, "content" is None.
                        # Otherwise, "content" is not None.
                        # When it is tool calls, success should be  False.
                        # The game will not completed by tool calls.
                        if chat_result.chat_history[-1]['content'] is not None:
                            success = "SUCCESS" in chat_result.chat_history[-1]['content']
                        else:
                            success = False

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

                        success = False

                        with open(log_paths['chat_history_path'], "w") as f:
                            f.write(f"Error Message: no chat history in chat result\n")

                        print(chat_result)

                else:
                    chat_round_list.append(-1)

                    success = False

                # exit()

                success_list.append(success)

                # save success and chat_round into a txt file
                with open(log_paths['result_path'], "w") as f:
                    f.write(f"Success: {success}\n")
                    f.write(f"Chat Round: {chat_round_list[-1]}\n")

                # save success list and chat_round_list into a txt file
                with open(result_list_path, "w") as f:
                    f.write(f"Success List: {success_list}\n")
                    f.write(f"Chat Round List: {chat_round_list}\n")

            print(f"Success Rate: {np.sum(success_list)}/{num_games}")
