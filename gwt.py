import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import time
from datetime import datetime
import alfworld.agents.modules.generic as generic
import alfworld.agents.environment as environment

def get_best_candidate(reference_sentence, candidate_sentences):
    reference = [reference_sentence.split()]
    best_score = 0.0
    best_candidate = ""

    for candidate_sentence in candidate_sentences:
        candidate = candidate_sentence.split()
        bleu_score = sentence_bleu(reference, candidate)

        if bleu_score > best_score:
            best_score = bleu_score
            best_candidate = candidate_sentence

    return best_candidate

def initialize_agents(llm_config):
    # Memory Agent
    memory_agent = ConversableAgent(
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
    retrive_memory_agent = ConversableAgent(
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
    task_agent = ConversableAgent(
        name="Task_Agent",
        system_message="""You are the Task Agent. First, you need to analyze the current information, think about the task, set a series of goals to accomplish the task.
        According to the feedback from the environment and other agents, you will gradually know the capability of yourself, and change the goals.
        You are improving yourself.
                    
        Format your response as:
        TASK_AGENT: 
        TASK ANALYSIS: ...
        CURRENT GOAL: ...
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    # Command Evaluation Agent
    command_evaluation_agent = ConversableAgent(
        name="Command_Evaluation_Agent",
        system_message="""You are the Command Evaluation Agent. You first fast analyze all the addmissible commands, and 
        check whether they are aligned with the current goal, choose the best addmissible command.
        Do not respond the evaluation process.
        
        Format your response as:
        COMMAND_EVALUATION_AGENT: 
        The best command. Addmissible or not. 
        If not addmissible, Task_Agent, current goal is out of our capability, please change the goal and replan.
        If addmissible, Executor_Agent, execute the command.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    # Executor Agent
    executor_agent = ConversableAgent(
        name="Executor_Agent",
        system_message="""You are the Executor Agent. You will execute the best addmissible command.
        The action you choose should be one of the addmissible commands.
        Check whether the action is in the addmissible commands before you call the function `execute_action`.
        You can only call the function `execute_action`, not other functions.
        You cannot reply directly, you must call the function `execute_action`.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
    )

    # Agent descriptions
    task_agent.description = "analyzes the task and proposes a plan to accomplish the task"
    retrive_memory_agent.description = "retrives the memory"
    memory_agent.description = "records the important information into memory"
    command_evaluation_agent.description = "evaluates the outcome of the command"
    executor_agent.description = "executes actions and returns observations"

    return memory_agent, retrive_memory_agent, task_agent, command_evaluation_agent, executor_agent

# Define execute_action
def execute_action(suggested_action: str) -> str:
    assert len(list(info['admissible_commands'])) == 1
    admissible_commands = list(info['admissible_commands'][0])
    assert len(admissible_commands) > 0

    action = get_best_candidate(suggested_action, admissible_commands)
    
    obs, scores, dones, info = env.step([action])
    
    # save the addmissible commands into a txt file
    with open(admissible_commands_path, "w") as f:
        f.write(f"{admissible_commands}\n")
                    
    if dones[0]:
        return f"Observation: {obs[0]}, SUCCESS"
    else:
        return f"Observation: {obs[0]}, IN_PROGRESS"

# Define record_memory
def record_memory(important_information: str) -> str:
    with open(memory_path, "a+") as f:
        f.seek(0)
        lines = f.readlines()
        
        step = 1  # Default to 1 if file is empty
        if lines:  # If file has content
            try:
                last_line = lines[-1]
                step = int(last_line.split(",")[0].split(":")[1]) + 1
            except (IndexError, ValueError):
                step = 1
        
        f.write(f"step: {step}, important_information: {important_information}\n")
    return "Memory recorded."

# Define retrive_memory
def retrive_memory() -> str:
    informations = ""
    
    with open(memory_path, "r") as f:
        for line in f:
            informations += line
            
    informations += "\nAddmissible Commands: "
    with open(admissible_commands_path, "r") as f:
        informations += f.read()
    
    return informations

def register_functions(env, obs, info, memory_path, admissible_commands_path, agents, memory_agent, retrive_memory_agent, executor_agent):
    

    # Register all functions
    register_function(
        execute_action,
        caller=executor_agent,
        executor=executor_agent,
        name="execute_action",
        description="Execute the action in the environment and return the observation"
    )

    register_function(
        record_memory,
        caller=memory_agent,
        executor=memory_agent,
        name="record_memory",
        description="Record the observation and action into memory"
    )

    register_function(
        retrive_memory,
        caller=retrive_memory_agent,
        executor=retrive_memory_agent,
        name="retrive_memory",
        description="Retrive the memory"
    )

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    
    
    
    if last_speaker is memory_agent:
        next_speaker = retrive_memory_agent
    elif last_speaker is retrive_memory_agent:
        next_speaker = task_agent
    elif last_speaker is task_agent:
        next_speaker = command_evaluation_agent
    elif last_speaker is command_evaluation_agent:
        if "Task_Agent" in messages[-1]["content"]:
            next_speaker = task_agent
        else:
            next_speaker = executor_agent
    elif last_speaker is executor_agent:
        next_speaker = memory_agent
    
    tool_call_flag = "tool_calls" in messages[-1]
    if tool_call_flag:
        return last_speaker
    else:
        return next_speaker

def initialize_groupchat(agents, llm_config):
    allowed_transitions = {
        memory_agent: [retrive_memory_agent],
        retrive_memory_agent: [task_agent],
        task_agent: [command_evaluation_agent],
        command_evaluation_agent: [executor_agent, task_agent],
        executor_agent: [memory_agent],
    }
    group_chat = GroupChat(
        agents=agents,
        messages=[],
        # allowed_or_disallowed_speaker_transitions=allowed_transitions,
        # speaker_transitions_type="allowed",
        
        speaker_selection_method=state_transition,
        max_round=200,
        send_introductions=True
    )

    return GroupChatManager(groupchat=group_chat, llm_config=llm_config)

def run_chat(obs, group_chat_manager, memory_agent):
    if isinstance(obs, (list, tuple)):
        initial_message_content = obs[0]
    else:
        initial_message_content = obs

    chat_result = memory_agent.initiate_chat(
        group_chat_manager,
        message={"role": "system", "content": initial_message_content},
        summary_method="reflection_with_llm"
    )
    return chat_result


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
                
                memory_path = os.path.join(game_path, "memory.txt")
                admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
                chat_history_path = os.path.join(game_path, "chat_history.txt")
                result_path = os.path.join(game_path, "result.txt")
                
                print("Initialized Environment")

                obs, info = env.reset()
                
                if i not in [4,5,6,10,11,13,14,16,18,19,21,22,23,24,25,26,28,30,31,33,34,35,36,38,39,42,43,44,45,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,73]:
                    continue
                
                print("Reset environment")

                llm_config = {
                    "cache_seed": None,
                    "max_tokens": 4000,
                    "config_list": [{"model": "gpt-4o-mini", "api_key": API_KEY}]
                }
                
                # Initialize agents and components
                memory_agent, retrive_memory_agent, task_agent, command_evaluation_agent, executor_agent = initialize_agents(llm_config)
                agents = [memory_agent, retrive_memory_agent, task_agent, command_evaluation_agent, executor_agent]
                
                # Register functions
                register_functions(env, obs, info, memory_path, admissible_commands_path, agents, 
                                memory_agent, retrive_memory_agent, executor_agent)
                
                # Initialize group chat
                group_chat_manager = initialize_groupchat(agents, llm_config)
                
                # find the task description in the observation, save it as a txt file
                task_description = obs[0].split("Your task is to: ")[1]
                initial_observation = obs[0].split("Your task is to: ")[0].split("\n\n")[1]
                with open(memory_path, "w") as f:
                    f.write(f"Task: {task_description}\n")
                    
                admissible_commands = list(info['admissible_commands'][0])
                with open(admissible_commands_path, "w") as f:
                    f.write(f"{admissible_commands}\n")
                
                run_chat_success = True
                
                try:
                    chat_result = run_chat(obs, group_chat_manager, memory_agent)                                        
                except Exception as e:
                    print(f"Group Chat manager fails to chat with error message {e}")
                    run_chat_success = False
                    error_message = e
                
                if run_chat_success:
                    if isinstance(chat_result, str):
                        success = "SUCCESS" in chat_result
                        
                        with open(chat_history_path, "w") as f:
                            f.write(chat_result)
                        
                        chat_round_list.append(-1)
                    elif "chat_history" in chat_result.__dict__.keys() and len(chat_result.chat_history) > 0 and isinstance(chat_result.chat_history[-1]['content'], str):
                        success = "SUCCESS" in chat_result.chat_history[-1]['content']
                        
                        with open(chat_history_path, "w") as f:
                            for message in chat_result.chat_history:
                                f.write('-'*100 + '\n')
                                f.write(f"{message['role']}: {message['content']}\n")
                        
                        chat_round_list.append(len(chat_result.chat_history))
                    else:
                        chat_round_list.append(-1)
                        success = False
                        
                        with open(chat_history_path, "w") as f:
                            f.write(f"Error Message: no chat history in chat result\n")
                else:
                    chat_round_list.append(-1)
                    success = False
                    
                    with open(chat_history_path, "w") as f:
                        f.write(f"Error Message: {error_message}\n")
                    
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

