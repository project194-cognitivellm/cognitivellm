import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml

from nltk.translate.bleu_score import sentence_bleu
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import os

# load config
config = generic.load_config()
API_KEY = "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"

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
            
            for i in range(num_games):
                
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

                llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}]}

                # assistant_agent = ConversableAgent(
                #     name="Assistant_Agent",
                #     system_message="You generate plans and make action decisions to solve a task. You will receive feedback from"
                #                 "other agents on the result of executing your action decisions. You must use this feedback when "
                #                 "generating further plans and action decisions."
                #                 "The feedback includes the observations, the admissible commands, scores, and dones."
                #                 "Your output should have the following format: THOUGHT[contents of your plan.] ACTION[proposed action.]"
                #                 "You proposed action should be one of the admissible commands."
                #                 "Example 1: "
                #                 "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1,"
                #                 " a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                #                 "Your task is to: look at bowl under the desklamp.]"
                #                 "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                #                 "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                #                 "ACTION [go to desk 1]"
                #                 "Example 2 (After assistant finds the desklamp at desk 1, then goes to desk 2.): "
                #                 "Feedback: [on the desk 2, you see a bowl 1, and a cd 3]"
                #                 "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                #                 "I'll go to the desklamp now.] ACTION [go to desk 1]",
                #     llm_config=llm_config,
                #     is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                #     human_input_mode="NEVER"
                # )

                print("Initialized assistant agent")

                planning_agent = ConversableAgent(
                    name="Planning_Agent",
                    system_message="You generate plans to solve a task. You will utilize memory when you generate your plans. You must use appropriate memory"
                                "to generate plans and action decisions. You will be given feedback to help you as you generate your plan."
                                "The feedback includes the observations, the admissible commands, scores, and dones."
                                "Your output should have the following format: THOUGHT[contents of your plan.]"
                                "Example 1: "
                                "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1,"
                                " a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                                "Your task is to: look at bowl under the desklamp.]"
                                "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                                "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                                "Example 2 (After assistant finds the desklamp at desk 1, then goes to desk 2.): "
                                "Feedback: [on the desk 2, you see a bowl 1, and a cd 3]"
                                "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                                "I'll go to the desklamp now.]",
                    llm_config=llm_config,
                    is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                    human_input_mode="NEVER"
                )
                action_agent = ConversableAgent(
                    name="Action_Agent",
                    system_message="You receive plans as input and make action decisions to solve a task. "
                                "Your output should have the following format:  ACTION[proposed action.]"
                                "You proposed action should be one of the admissible commands."
                                "Example 1: "
                                "THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                                "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.]"
                                "Your Output: ACTION [go to desk 1]"
                                "Example 2  THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                                "I'll go to the desklamp now.]"
                                "Your Output:  ACTION [go to desk 1]",
                    llm_config=llm_config,
                    is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                    human_input_mode="NEVER"
                )

                environment_proxy = ConversableAgent(
                    name="Environment Proxy",
                    llm_config=False,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                )

                print("Initialized environment proxy")

                executor_agent = ConversableAgent(
                    name="Executor_Agent",
                    system_message="You call the execute_action function with the proposed action as the argument",
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                )

                print("Initialized executor agent")

                grounding_agent = ConversableAgent(
                    name="Grounding_Agent",
                    system_message="You provide general knowledge at the start of task when the chat begins and whenever the "
                                    "environment_proxy reports the same results three times in a row",
                    llm_config=llm_config,
                    human_input_mode="NEVER",
                    is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
                )

                print("Initialized grounding agent")

                allowed_transitions = {
                    action_agent: [executor_agent],
                    executor_agent: [environment_proxy],
                    grounding_agent: [executor_agent, planning_agent],
                    environment_proxy: [planning_agent, grounding_agent],
                    planning_agent: [action_agent]
                }


                #assistant_agent.description = "generates plans and makes action decisions to solve the task"
                planning_agent.description = "generates plans for solving a task based on feedback and information from the grounding agent"
                action_agent.description = "provides an actions, given the output of the planning agent"
                executor_agent.description = "calls execute_action with the proposed action as the argument to perform the suggested action"
                environment_proxy.description = "reports action execution results as feedback."
                grounding_agent.description = ("provides general knowledge at the start of task when the chat begins and whenever the "
                                            "environment_proxy reports the same results three times in a row. If it is the start of the task"
                                            ", call assistant_agent to generate the first plan."
                                            "If the task is completed, output SUCCESS.")

                group_chat = GroupChat(
                    agents=[planning_agent, action_agent, executor_agent, grounding_agent, environment_proxy],
                    messages=[],
                    allowed_or_disallowed_speaker_transitions=allowed_transitions,
                    speaker_transitions_type="allowed",
                    max_round=50,
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

                try:
                    chat_result = grounding_agent.initiate_chat(
                            group_chat_manager,
                            message={"role": "system", "content": initial_message_content},
                            summary_method="reflection_with_llm"
                    )
                    print("Finished Chat")
                except Exception as e:
                    print(f"Group Chat manager fails to chat with error message {e}")


                success = "SUCCESS" in chat_result.chat_history[-1]['content']

                success_list.append(success)
            
            print(f"Success Rate: {np.sum(success_list)}/{num_games}")
