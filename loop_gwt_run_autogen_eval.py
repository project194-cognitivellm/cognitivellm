# cognitive_agent.py

import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import time
from datetime import datetime


def get_best_candidate(reference_sentence, candidate_sentences):
    # Tokenize the reference sentence
    reference = [reference_sentence.split()]
    best_score = 0.0
    best_candidate = ""

    # Iterate through each candidate sentence and calculate the BLEU score
    for candidate_sentence in candidate_sentences:
        candidate = candidate_sentence.split()
        bleu_score = sentence_bleu(reference, candidate)

        # Update best score and best candidate if this candidate is better
        if bleu_score > best_score:
            best_score = bleu_score
            best_candidate = candidate_sentence

    return best_candidate


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

    def initialize_agents(self):

        # Memory Agent
        self.memory_agent = ConversableAgent(
            name="Memory_Agent",
            system_message="""You are the Memory Agent. According to the observation and commond you took, you will record the important information related to the task.
            You can only call the function `record_memory`, not other functions.
            You cannot reply directly, you must call the function `record_memory`.
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )
        
        # Retrive Memory Agent
        self.retrive_memory_agent = ConversableAgent(
            name="Retrive_Memory_Agent",
            system_message="""You are the Retrive Memory Agent. You call the function `retrive_memory` to retrieve the memory.
            You can only call the function `retrive_memory`, not other functions.
            You cannot reply directly, you must call the function `retrive_memory`.
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )
        
        # Task Agent
        self.task_agent = ConversableAgent(
            name="Task_Agent",
            system_message="""You are the Task Agent. First, you need to analyze the current information, think about the task, set a series of goals to accomplish the task.
            According to the feedback from the environment and other agents, you will gradually know the capability of yourself, and change the goals.
            You are improving yourself.
                        
            Format your response as:
            TASK_AGENT: 
            TASK ANALYSIS: ...
            CURRENT GOAL: ...
            TASK STATUS: SUCCESS or IN_PROGRESS
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )

        # Command Evaluation Agent
        self.command_evaluation_agent = ConversableAgent(
            name="Command_Evaluation_Agent",
            
            ## this prompt is good, but it is too slow. gpt-4o-mini will report 500 error.
            # You are the Command Evaluation Agent. You must evaluate all the addmissible commands,
            # check their alignment with the current goal. If none of addmissible commands is aligned with the current goal,
            # you need to report the failure to Task_Agent and let it change the goal.
            
            # You need to analyze all the addmissible commands.
            # Your thinking process should be:
            # 1. first addmissible command: evaluation
            # 2. second addmissible command: evaluation
            # 3. ...
            # n. last addmissible command: evaluation
            # After evaluating all the addmissible commands, you need to choose the best addmissible command.
            # Do not respond the evaluation process.
            
            system_message="""You are the Command Evaluation Agent. You first fast select candidate commands from
            the addmissible commands. Second, evaluate all the candidate commands, check whether they are aligned with the current goal.
            If none of addmissible commands is aligned with the current goal,
            you need to report the failure to Task_Agent and let it change the goal.
            
            Do not respond the evaluation process.
            
            Format your response as:
            COMMAND_EVALUATION_AGENT: 
            The best addmissible command. If there is no addmissible command aligned with the current goal, the current goal is out of our capability. TASK_AGENT, you need to change the goal and plan.
            """,
            llm_config=llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
        )

        # Executor Agent
        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="""You are the Executor Agent. You will execute the best addmissible command.
            The action you choose should be one of the addmissible commands.
            Check whether the action is in the addmissible commands before you call the function `execute_action`.
            You can only call the function `execute_action`, not other functions.
            You cannot reply directly, you must call the function `execute_action`.
            """,
            llm_config=self.llm_config,  # Ensure llm_config is set
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
        )


        # Allowed transitions between agents
        self.allowed_transitions = {
            self.memory_agent: [self.retrive_memory_agent],
            self.retrive_memory_agent: [self.task_agent],
            self.task_agent: [self.command_evaluation_agent],
            self.command_evaluation_agent: [self.executor_agent, self.task_agent],
            self.executor_agent: [self.memory_agent],
        }

        # Agent descriptions
        self.task_agent.description = "analyzes the task and proposes a plan to accomplish the task"
        self.retrive_memory_agent.description = "retrives the memory"
        self.memory_agent.description = "records the important information into memory"
        self.command_evaluation_agent.description = "evaluates the outcome of the command"
        self.executor_agent.description = "executes actions and returns observations"

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
            max_round=200,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            action = get_best_candidate(suggested_action, admissible_commands)
            
            self.obs, scores, dones, self.info = self.env.step([action])
            
            # save the addmissible commands into a txt file
            with open(admissible_commands_path, "w") as f:
                f.write(f"{admissible_commands}\n")
                            
            # time.sleep(1)
            return f"Observation: {self.obs[0]}, Success: {dones[0]}"

        # Register the execute_action function with Executor_Agent
        register_function(
            execute_action,
            caller=self.executor_agent,  # Executor_Agent has llm_config=True
            executor=self.executor_agent,  # Executor_Agent handles execution
            name="execute_action",
            description="Execute the action in the environment and return the observation"
        )

        # Define record_memory function
        def record_memory(important_information: str) -> str:
            with open(memory_path, "a+") as f:
                # Move cursor to beginning of file
                f.seek(0)
                # Read all lines
                lines = f.readlines()
                
                # Get step number
                step = 1  # Default to 1 if file is empty
                if lines:  # If file has content
                    try:
                        last_line = lines[-1]
                        step = int(last_line.split(",")[0].split(":")[1]) + 1
                    except (IndexError, ValueError):
                        step = 1
                
                # Write new memory entry
                f.write(f"step: {step}, important_information: {important_information}\n")
            # time.sleep(1)
            return "Memory recorded."
        
        # Register the record_memory function with Memory_Agent
        register_function(
            record_memory,
            caller=self.memory_agent,  # Memory_Agent has llm_config=True
            executor=self.memory_agent,  # Memory_Agent handles execution
            name="record_memory",
            description="Record the observation and action into memory"
        )
        
        # Define retrive_memory function, return all the content in the memory.txt file
        def retrive_memory() -> str:
            informations = ""
            
            with open(memory_path, "r") as f:
                for line in f:
                    informations += line
                    
            informations += "\nAddmissible Commands: "
            with open(admissible_commands_path, "r") as f:
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

    def run_chat(self):
        try:
            if isinstance(self.obs, (list, tuple)):
                initial_message_content = self.obs[0]
            else:
                initial_message_content = self.obs

            # Start the chat with the Planner Agent proposing a plan
            chat_result = self.memory_agent.initiate_chat(
                self.group_chat_manager,
                message={"role": "system", "content": initial_message_content},
                summary_method="reflection_with_llm"
            )
        except Exception as e:
            return f"Group Chat manager fails to chat with error message {e}"

        return chat_result
    
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment
# load config
config = generic.load_config()
API_KEY = "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"

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
                
                memory_path = os.path.join(game_path, "memory.txt")
                admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
                chat_history_path = os.path.join(game_path, "chat_history.txt")
                
                result_path = os.path.join(game_path, "result.txt")
                
                print("Initialized Environment")

                obs, info = env.reset()
                
                # if i not in [1,2,4,5,6,10,11,13,14,16,18,19,21,22,23,24,25,26,28,30,31,33,34,35,36,38,39,42,43,44,45,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,73]:
                #     continue
                
                print("Reset environment")

                llm_config = {
                    # "timeout": 6000,
                    "cache_seed": None,
                    # "temperature": 1,
                    "max_tokens": 4000,
                    "config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}]}
                
                # find the task description in the observation, save it as a txt file. 
                task_description = obs[0].split("Your task is to: ")[1]
                initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]
                with open(memory_path, "w") as f:
                    f.write(f"Task: {task_description}\n")
                    
                admissible_commands = list(info['admissible_commands'][0])
                # save the addmissible commands into a txt file
                with open(admissible_commands_path, "w") as f:
                    f.write(f"{admissible_commands}\n")
                                
                agent = CognitiveAutogenAgent(env, obs, info, llm_config)
                chat_result = agent.run_chat()
                # is chat_result a string?
                if isinstance(chat_result, str):
                    success = "SUCCESS" in chat_result
                    
                    # record the chat history into a txt file
                    with open(chat_history_path, "w") as f:
                        f.write(chat_result)
                    
                    chat_round_list.append(-1)
                else:
                    success = "SUCCESS" in chat_result.chat_history[-1]['content']
                    
                    # record the chat history into a txt file
                    # chat_result.chat_history is a list of dictionaries
                    with open(chat_history_path, "w") as f:
                        for message in chat_result.chat_history:
                            f.write('-'*100 + '\n')
                            f.write(f"{message['role']}: {message['content']}\n")
                    
                    chat_round_list.append(len(chat_result.chat_history))
                        
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
