# cognitive_agent.py

import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import time
from datetime import datetime
import pickle
from sentence_transformers import SentenceTransformer, util


sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')





def get_best_candidate(reference_sentence, candidate_sentences):
    # Compute embeddings
    target_embedding = sentence_transformer_model.encode(reference_sentence, convert_to_tensor=True)
    command_embeddings = sentence_transformer_model.encode(candidate_sentences, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(target_embedding, command_embeddings)

    # Find the most similar command
    most_similar_idx = similarities.argmax()
    most_similar_command = candidate_sentences[most_similar_idx]
    score = similarities.detach().cpu().numpy()[0,most_similar_idx]
    

    return most_similar_command, score

# def get_best_candidate(reference_sentence, candidate_sentences):
#     # Tokenize the reference sentence
#     reference = [reference_sentence.split()]
#     best_score = 0.0
#     best_candidate = ""

#     # Iterate through each candidate sentence and calculate the BLEU score
#     for candidate_sentence in candidate_sentences:
#         candidate = candidate_sentence.split()
#         bleu_score = sentence_bleu(reference, candidate)

#         # Update best score and best candidate if this candidate is better
#         if bleu_score > best_score:
#             best_score = bleu_score
#             best_candidate = candidate_sentence

#     return best_candidate


class CognitiveAutogenAgent:
    def __init__(self, env, obs, info, llm_config):
        self.env = env
        self.obs = obs
        self.info = info
        self.llm_config = llm_config

        # Initialize plan memory database
        self.plan_memory_database = {}

        self.initialize_agents()
        self.register_functions()
        self.initialize_groupchat()

    def initialize_agents(self):
        
        # Retrive Memory Agent
        self.retrive_memory_agent = ConversableAgent(
            name="Retrive_Memory_Agent",
            system_message="""You are the Retrive Memory Agent. You task is ONLY to call the function `retrive_memory` to retrieve the memory.
            DO NOT analyze any information such as task, history, addmissible commands, guidance, etc.
            **RULES:**
            The TOOL you can only use is `retrive_memory`.
            DO NOT call any other tools.
            You can only use `retrive_memory` once per step.            
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )
        
        self.start_agent = ConversableAgent(
            name="Start_Agent",
            llm_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
            human_input_mode="NEVER",
        )
        
        # Guidance Agent
        self.guidance_agent = ConversableAgent(
            name="Guidance_Agent",
            system_message="""You are the Guidance Agent. Your task is to extract beneficial guidance from history.
            If you tried to do something but failed (e.g., commands is not admissible), but after trying different plans or commands you succeeded, you should learn from the failures and successes, and the successful methods is a good guidance.
            
            Avoid including specific items, locations, or overly concrete steps in the rules. 
            Focus on broadly applicable principles that summarize patterns of success or failure.
            NOTE: the history is not always correct, the rules you learned must be successful and beneficial.

            **Guidance:**
                **Analysis Process:**
                - Understand Your Capabilities: Note limitations based on observed failures or non-admissible actions. Record rules that prevent you from repeating errors.
                - Extract Successful Strategies: Identify and save successful subplans or tactics that can be reused in similar situations to enhance efficiency.
                - Avoid Redundant Guidance: Always summarize findings into a maximum of 2–3 rules. If the guidance is already covered by previous guidance, do not record it.

                **Output Guidelines:**
                - If no new guidance are found, explicitly state: "NO NEW GUIDANCE at this time."
                - Avoid summarizing or repeating history directly; focus on actionable principles.

                **Examples:**
                History you get: 
                1. You tried to open a drawer but failed. After examining it, you succeeded. 
                2. You tried carrying two objects simultaneously but failed. You succeeded after dividing the tasks, carry one object at a time.
                Guidance you learned from these history:
                1. Always examine an object before attempting to interact with it.
                2. Cannot carry more than one object at a time. Divide tasks accordingly.
                
                **Output format:**
                Guidance: 
                1. ...
                2. ...
                3. ...
                ...
                

            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )
        
        
        self.record_guidance_agent = ConversableAgent(
            name="Record_Guidance_Agent",
            system_message="""You are the Record Guidance Agent. You task is ONLY to call the function `record_guidance` to record the new guidance.
            DO NOT analyze any information such as task, history, addmissible commands, etc. You only need to record the new guidance, not repeat the previous guidance.
            
            **IMPORTANT:**
            If 'No new guidance at this time.', do not call the function `record_guidance`.
            
            **RULES:**
            The TOOL you can only use is `record_guidance`.
            DO NOT call any other tools.
            You can only use `record_guidance` once per step.            
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )
        

        
        # Task Agent
        self.task_agent = ConversableAgent(
            name="Task_Agent",
            system_message="""You are the Task Agent. First, you need to analyze the current information, think about the task, set a series of goals to accomplish the task, and then propose several candidate commands for the next step.
            Your candidate commands should be compact, and be from ADMISSIBLE_COMMANDS. Maximum number of candidate commands is 3.
            According to the feedback from the environment and other agents, you will gradually know the capability of yourself, and change the goals.
            You are improving yourself.
            
            Examples of candidate commands:
            1. go to drawer 1
            2. go to shelf 1
            3. examine drawer 1
            
            Examples of not candidate commands:
            1. inventory to check if I already possess a spray bottle.
            2. go to drawer 1 to check for a spray bottle.
                        
            Format your response as:
            TASK ANALYSIS: ...
            CURRENT GOAL: ...
            CANDIDATE COMMANDS: ...
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )

        # Command Evaluation Agent
        self.command_evaluation_agent = ConversableAgent(
            name="Command_Evaluation_Agent",
            
            system_message="""You are the Command Evaluation Agent. You evaluate the candidate commands and choose the best one.
            suggestions for evaluation: 
            1. evaluate the commands leads to the achieve of current goal.
            2. evaluate the commands according to the guidance.
            3. evaluate the commands is addmissible.
            
            Format your response as:
            Best Command You Choose for execution: ...
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )

        # Executor Agent
        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="""You are the Executor Agent. You will execute the best command by calling the function `execute_action`.
            Rules:
            The TOOL you can only use is `execute_action`.
            You can only use `execute_action` once per step. 
            """,
            llm_config=self.llm_config,  # Ensure llm_config is set
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
        )
        
        

        # Agent descriptions
        self.task_agent.description = "analyzes the task and proposes a plan to accomplish the task"
        self.retrive_memory_agent.description = "retrives the memory"
        self.guidance_agent.description = "analyzes the history and proposes guidance for your capability, envrionment, and task"
        self.record_guidance_agent.description = "records the new guidance"
        self.command_evaluation_agent.description = "evaluates the outcome of the command"
        self.executor_agent.description = "executes actions and returns observations"

        

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            action, action_score = get_best_candidate(suggested_action, admissible_commands)
            
            
            if action_score < 0.8:
                output = f"action '{suggested_action}' is not addmissible.\n"
                with open(history_path, "a+") as f:
                    f.write(output)
                return f"Observation: action '{suggested_action}' is not addmissible."
            
            self.obs, scores, dones, self.info = self.env.step([action])
            
            # save the addmissible commands into a txt file
            with open(admissible_commands_path, "w") as f:
                f.write(f"{admissible_commands}\n")
                            
            
            # def record_memory(important_information: str) -> str:
            with open(history_path, "a+") as f:
                f.write(f"action: [{action}] observation: [{self.obs[0]}]\n")
                            
                            
                            
            # time.sleep(1)
            if dones[0]:
                return f"Observation: {self.obs[0]} SUCCESS"
            else:
                return f"Observation: {self.obs[0]} IN_PROGRESS"

        # Register the execute_action function with Executor_Agent
        register_function(
            execute_action,
            caller=self.executor_agent,  # Executor_Agent has llm_config=True
            executor=self.executor_agent,  # Executor_Agent handles execution
            name="execute_action",
            description="Execute the action in the environment and return the observation"
        )

        # # Define record_memory function
        # def record_memory(important_information: str) -> str:
        #     with open(history_path, "a+") as f:
        #         # Move cursor to beginning of file
        #         f.seek(0)
        #         # Read all lines
        #         lines = f.readlines()
                
        #         # Get step number
        #         step = 1  # Default to 1 if file is empty
        #         if lines:  # If file has content
        #             try:
        #                 last_line = lines[-1]
        #                 step = int(last_line.split(":")[0].split(" ")[-1]) + 1
        #             except (IndexError, ValueError):
        #                 step = 1
                
        #         # Write new memory entry
        #         f.write(f"step {step}: {important_information}\n")
        #     # time.sleep(1)
        #     return "Memory recorded."
        
        # Define record_memory function
        def record_guidance(guidance: str) -> str:
            
            # the maximum number of lines are 5; if more than 5, delete the oldest one.
            with open(guidance_path, "a+") as f:
                f.write(f"{guidance}\n")
                lines = f.readlines()
                if len(lines) > 5:
                    f.seek(0)
                    f.truncate()
                    for line in lines[:-1]:
                        f.write(line)
                        
            # time.sleep(1)
            return "Guidance recorded."
        
        # Register the record_memory function with Memory_Agent
        register_function(
            record_guidance,
            caller=self.record_guidance_agent,  # Record_Guidance_Agent has llm_config=True
            executor=self.record_guidance_agent,  # Record_Guidance_Agent handles execution
            name="record_guidance",
            description="Record guidance learned from history"
        )
        
        # Define retrive_memory function, return all the content in the memory.txt file
        def retrive_memory() -> str:
            informations = ""
            
            if os.path.exists(task_path):
                informations += "Task: \n"
                with open(task_path, "r") as f:
                    informations += f.read()
            
            
            if os.path.exists(history_path):
                # latest 10 steps. last 10 lines
                informations += "\nRecent 10 steps History: \n"
                with open(history_path, "r") as f:
                    for line in f.readlines()[-10:]:
                        informations += line
                   
            if os.path.exists(admissible_commands_path):
                informations += "\nAddmissible commands for current step: \n"
                with open(admissible_commands_path, "r") as f:
                    informations += f.read()
                
                
            if os.path.exists(guidance_path):
                informations += "\nGuidance: \n"
                with open(guidance_path, "r") as f:
                    informations += f.read()

            
            # time.sleep(1)
            return informations

        # Register the retrive_memory function with Retrive_Memory_Agent
        register_function(
            retrive_memory,
            caller=self.retrive_memory_agent,  # Retrive_Memory_Agent has llm_config=True
            executor=self.retrive_memory_agent,  # Retrive_Memory_Agent handles execution
            name="retrive_memory",
            description="Retrive the memory"
        )
        
        
    def initialize_groupchat(self):
        customized_state_transition = True
        max_chat_round = 300
        if customized_state_transition:

            def state_transition(last_speaker, groupchat):
                messages = groupchat.messages

                # print(len(messages))
                
                # record the last message
                # last message is a Dict. use pickle to save it.
                with open(message_path, "wb") as f:
                    pickle.dump(messages, f)
                

                if last_speaker is self.start_agent:
                    next_speaker = self.task_agent
                elif last_speaker is self.retrive_memory_agent:
                    next_speaker = self.guidance_agent
                elif last_speaker is self.guidance_agent:
                    if "NO NEW GUIDANCE" in messages[-1]["content"]:
                        next_speaker = self.task_agent
                    else:
                        next_speaker = self.record_guidance_agent
                elif last_speaker is self.record_guidance_agent:
                    next_speaker = self.task_agent
                elif last_speaker is self.task_agent:
                    next_speaker = self.command_evaluation_agent
                elif last_speaker is self.command_evaluation_agent:
                    if "Task_Agent" in messages[-1]["content"]:
                        next_speaker = self.task_agent
                    else:
                        next_speaker = self.executor_agent
                elif last_speaker is self.executor_agent:
                    next_speaker = self.retrive_memory_agent

                tool_call_flag = "tool_calls" in messages[-1]
                if tool_call_flag:
                    return last_speaker
                else:
                    return next_speaker
            
            
            # Group Chat
            self.group_chat = GroupChat(
                agents=[
                    self.start_agent,
                    self.record_guidance_agent,
                    self.guidance_agent,
                    self.retrive_memory_agent,
                    self.task_agent,
                    self.command_evaluation_agent,
                    self.executor_agent,
                ],
                messages=[],
                speaker_selection_method=state_transition,
                max_round=max_chat_round,
                send_introductions=True
            )
            
        else:
            # Allowed transitions between agents
            self.allowed_transitions = {
                self.memory_agent: [self.retrive_memory_agent],
                self.retrive_memory_agent: [self.task_agent],
                self.task_agent: [self.command_evaluation_agent],
                self.command_evaluation_agent: [self.executor_agent, self.task_agent],
                self.executor_agent: [self.memory_agent],
            }
            
            
            # Group Chat
            self.group_chat = GroupChat(
                agents=[
                    self.memory_agent,
                    self.retrive_memory_agent,
                    self.task_agent,
                    self.command_evaluation_agent,
                    self.executor_agent,
                ],
                messages=[],
                allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
                speaker_transitions_type="allowed",
                max_round=max_chat_round,
                send_introductions=True
            )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def run_chat(self, initial_message_content):
        chat_result = None
        error_message = None
        try:
            # Start the chat with the Planner Agent proposing a plan
            chat_result = self.start_agent.initiate_chat(
                self.group_chat_manager,
                message={"role": "system", "content": initial_message_content},
                summary_method="reflection_with_llm"
            )
        except Exception as e:
            print(f"Group Chat manager fails to chat with error message {e}")
            error_message = e
            
        return chat_result, error_message
    
    def resume_chat(self, last_message):
        chat_result = None
        error_message = None
        try:
            last_agent, last_message = self.group_chat_manager.resume(messages=last_message)

            # Resume the chat using the last agent and message
            chat_result = last_agent.initiate_chat(recipient=self.group_chat_manager, message=last_message, clear_history=False)
            
        except Exception as e:
            print(f"Group Chat manager fails to chat with error message {e}")
            error_message = e

        return chat_result, error_message
    
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
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

for eval_env_type in eval_envs:
    for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
        for eval_path in eval_paths:
            print("Evaluating: %s" % eval_path)
            config["general"]["evaluate"]["env"]["type"] = eval_env_type
            config["dataset"]["eval_ood_data_path"] = eval_path
            config["controller"]["type"] = controller_type

            alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
            env = alfred_env.init_env(batch_size=1)

            ## For each set, there are `num_games` games we need to evaluate
            num_games = alfred_env.num_games
            max_steps = 100
            success_list = []

            for i in range(num_games):
                
                game_path = os.path.join(base_path, f"game_{i}")    
                os.makedirs(game_path, exist_ok=True)
                
                task_path = os.path.join(game_path, "task.txt")
                history_path = os.path.join(game_path, "history.txt")
                guidance_path = os.path.join(game_path, "guidance.txt")
                admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
                chat_history_path = os.path.join(game_path, "chat_history.txt")
                
                message_path = os.path.join(game_path, "last_message.pkl")
                
                result_path = os.path.join(game_path, "result.txt")
                
                error_message_path = os.path.join(game_path, "error_message.txt")
                
                print("Initialized Environment")

                obs, info = env.reset()
                
                # if i not in [1]:
                    
                #     continue
                
                
                # if i not in [10,11,13,14,16,18,19,21,22,23,24,25,26,28,30,31,33,34,35,36,38,39,42,43,44,45,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,73]:
                #     continue
                
                # print("Reset environment")

                llm_config = {
                    "timeout": 1000,
                    "cache_seed": None,
                    # "temperature": 1,
                    "max_tokens": 300,
                    "config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}]}
                
                
                initial_message_content = ""
                # find the task description in the observation, save it as a txt file. 
                task_description = obs[0].split("Your task is to: ")[1]
                initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]
                with open(task_path, "w") as f:
                    f.write(f"Task: {task_description}\n")
                    
                initial_message_content += f"Task: {task_description}\n"
                    
                with open(history_path, "w") as f:
                    f.write(f"action: [None] observation: [{initial_observation}]\n")
                    
                initial_message_content += f"Observation: {initial_observation}\n"
                    
                admissible_commands = list(info['admissible_commands'][0])
                # save the addmissible commands into a txt file
                with open(admissible_commands_path, "w") as f:
                    f.write(f"{admissible_commands}\n")
                                
                                
                initial_message_content += f"Addmissible commands: {admissible_commands}\n"
                
                agent = CognitiveAutogenAgent(env, obs, info, llm_config)
                
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
                    with open(error_message_path, "a") as f:
                        f.write(f"Run Chat: {error_message}\n")
                
                
                print("Resume chat")
                max_num_of_resume = 5
                
                if chat_result is None:
                    for i in range(max_num_of_resume):
                        time.sleep(10)
                        if os.path.exists(message_path):
                            with open(message_path, "rb") as f:
                                last_message = pickle.load(f)   
                            
                            remove_index = 0
                            for j in range(len(last_message)):
                                if last_message[-j-1]['role'] == 'user':
                                    if last_message[-j-1]['name'] == 'chat_manager' or last_message[-j-1]['name'] == 'Task_Agent':
                                        remove_index = - j
                                        break
                            
                            last_message = last_message[:len(last_message)+remove_index]
                            
                        chat_result, error_message = agent.resume_chat(last_message)
                        
                        if chat_result is not None:
                            break
                        
                        if error_message is not None:
                            with open(error_message_path, "a") as f:
                                f.write(f"Resume Chat {i+1}: {error_message}\n")
                        
                # print(type(chat_result))
                # print(chat_result)
                # print(list(chat_result.__dict__.keys()))
                # print(f"Run chat: {run_chat}")
                # exit()
                
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
                        with open(chat_history_path, "w") as f:
                            for message in chat_result.chat_history:
                                f.write('-'*20 + '\n')
                                
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
                        
                        with open(chat_history_path, "w") as f:
                            f.write(f"Error Message: no chat history in chat result\n")
                            
                        print(chat_result)
                        
                else:
                    chat_round_list.append(-1)
                    
                    success = False
                    
                
                # exit()
                    
                    
                        
                success_list.append(success)
                
                # save success and chat_round into a txt file
                with open(result_path, "w") as f:
                    f.write(f"Success: {success}\n")
                    f.write(f"Chat Round: {chat_round_list[-1]}\n")
                
                
                # save success list and chat_round_list into a txt file
                with open(result_list_path, "w") as f:
                    f.write(f"Success List: {success_list}\n")
                    f.write(f"Chat Round List: {chat_round_list}\n")
                                
                
            print(f"Success Rate: {np.sum(success_list)}/{num_games}")
