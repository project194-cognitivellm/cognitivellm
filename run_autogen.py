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

from nltk.translate.bleu_score import sentence_bleu
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import os
import time
import csv
# load config
config = generic.load_config()


eval_paths = config["general"]["evaluate"]["eval_paths"]
eval_envs = config["general"]["evaluate"]["envs"]
controllers = config["general"]["evaluate"]["controllers"]
repeats = config["general"]["evaluate"]["repeats"]


# print(eval_envs,controllers,eval_paths)
# exit()

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
            env = alfred_env.init_env(batch_size=1)
            
            # for each set, there are num_games games we need to evaluate
            num_games = alfred_env.num_games
            
            # We need to set a max number of steps for each game. Refer to the ALFWorld paper? 
            max_steps = 100
            
            success_list = []
            
            for i in range(50):
                
                print("Initialized Environment")

                # interact
                obs, info = env.reset()
                print("Reset environment")
                print(f"Admissible Commands: {info['admissible_commands'][0]}")

                def get_best_candidate(reference_sentence, candidate_sentences):
                    # Tokenize the reference sentence
                    # print(f"Reference Sentence: {reference_sentence}")
                    reference = [reference_sentence.split()]
                    best_score = 0.0
                    best_candidate = ""

                    # Iterate through each candidate sentence and calculate the BLEU score
                    for candidate_sentence in candidate_sentences:
                        candidate = candidate_sentence.split()
                        bleu_score = sentence_bleu(reference, candidate)
                        # print(f"Candidate Sentence: {candidate_sentence}, BLEU: {bleu_score}")

                        # Update best score and best candidate if this candidate is better
                        if bleu_score > best_score:
                            best_score = bleu_score
                            best_candidate = candidate_sentence

                    # print(f"Best Candidate: {best_candidate}, BLEU: {best_score}")
                    return best_candidate

                def execute_action(suggested_action: str) -> str:
                    global info
                    assert len(list(info['admissible_commands'])) == 1
                    admissible_commands = list(info['admissible_commands'][0])
                    assert len(admissible_commands) > 0
                    
                    # print(f"Suggested Action: {suggested_action}")
                    # print(f"Admissible Commands: {admissible_commands}")
                    
                    action = get_best_candidate(suggested_action, admissible_commands)
                    obs, scores, dones, info = env.step([action])
                    return obs[0], f"Admissible Commands: {admissible_commands}, Scores: {scores[0]}, Dones: {dones[0]}"

                llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

                assistant_agent = ConversableAgent(
                    name="Assistant_Agent",
                    system_message="You generate plans and make action decisions to solve a task. You will receive feedback from"
                                "other agents on the result of executing your action decisions. You must use this feedback when "
                                "generating further plans and action decisions."
                                "The feedback includes the observations, the admissible commands, scores, and dones."
                                "Your output should have the following format: THOUGHT[contents of your plan.] ACTION[proposed action.]"
                                "You proposed action should be one of the admissible commands."
                                "Example 1: "
                                "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1,"
                                " a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                                "Your task is to: look at bowl under the desklamp.]"
                                "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                                "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                                "ACTION [go to desk 1]"
                                "Example 2 (After assistant finds the desklamp at desk 1, then goes to desk 2.): "
                                "Feedback: [on the desk 2, you see a bowl 1, and a cd 3]"
                                "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                                "I'll go to the desklamp now.] ACTION [go to desk 1]",
                    llm_config=llm_config,
                    is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"]),
                    human_input_mode="NEVER"
                )

                print("Initialized assistant agent")

                environment_proxy = ConversableAgent(
                    name="Environment Proxy",
                    llm_config=False,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"]) ,
                )

                print("Initialized environment proxy")

                executor_agent = ConversableAgent(
                    name="Executor_Agent",
                    system_message="You call the execute_action function with the proposed action as the argument",
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"]),
                )

                print("Initialized executor agent")

                grounding_agent = ConversableAgent(
                    name="Grounding_Agent",
                    system_message="You provide general knowledge at the start of task when the chat begins and whenever the "
                                    "environment_proxy reports the same results three times in a row.",
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"]),
                )

                print("Initialized grounding agent")

                allowed_transitions = {
                    assistant_agent: [executor_agent],
                    executor_agent: [environment_proxy],
                    grounding_agent: [executor_agent, assistant_agent],
                    environment_proxy: [assistant_agent, grounding_agent]
                }

                assistant_agent.description = "generates plans and makes action decisions to solve the task"
                executor_agent.description = "calls execute_action with the proposed action as the argument to perform the suggested action"
                environment_proxy.description = "reports action execution results as feedback."
                grounding_agent.description = ("provides general knowledge at the start of task when the chat begins and whenever the "
                                            "environment_proxy reports the same results three times in a row. If it is the start of the task"
                                            ", call assistant_agent to generate the first plan."
                                            "If the task is completed, output SUCCESS."
                                            "If you think we have rolled out all the options, output FAILURE.")

                group_chat = GroupChat(
                    agents=[assistant_agent, executor_agent, grounding_agent, environment_proxy],
                    messages=[],
                    allowed_or_disallowed_speaker_transitions=allowed_transitions,
                    speaker_transitions_type="allowed",
                    max_round=100,
                    send_introductions=True
                )

                print("Initialized group chat")

                group_chat_manager = GroupChatManager(
                    groupchat=group_chat,
                    llm_config=llm_config,
                )

                print("Initialized group chat manager")

                register_function(
                    execute_action,
                    caller=executor_agent,
                    executor=environment_proxy,
                    name ="execute_function",
                    description="Call this function to execute the suggested action"
                )

                print("Initialization complete; Starting Chat")

                if isinstance(obs, (list, tuple)):
                    initial_message_content = obs[0]
                else:
                    initial_message_content = obs

                chat_result = grounding_agent.initiate_chat(
                        group_chat_manager,
                        message={"role": "system", "content": initial_message_content},
                        summary_method="reflection_with_llm"
                )
                print("Finished Chat")


                success = "SUCCESS" in chat_result.chat_history[-1]['content']

                success_list.append(success)
                
                csv_filename = f"{eval_env_type}_{controller_type}_{eval_path.replace('/', '_')}.csv"
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['game_index', 'success'])
                    for idx, success in enumerate(success_list):
                        writer.writerow([idx, int(success)])  # Convert boolean to 0/1
                    
                
                
                print('-'*10)
                print(i, success)
                print(success_list)
                time.sleep(10)            
            print(f"Success Rate: {np.sum(success_list)}/{num_games}")

