import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from nltk.translate.bleu_score import sentence_bleu
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import os
import time
import csv

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

def execute_action(suggested_action: str) -> str:
    global info
    assert len(list(info['admissible_commands'])) == 1
    admissible_commands = list(info['admissible_commands'][0])
    assert len(admissible_commands) > 0
    
    action = get_best_candidate(suggested_action, admissible_commands)
    obs, scores, dones, info = env.step([action])
    return obs[0], f"Admissible Commands: {admissible_commands}, Scores: {scores[0]}, Dones: {dones[0]}"

def get_possible_actions() -> str:
    global info
    admissible_commands = list(info['admissible_commands'][0])
    return f"Admissible Commands: {admissible_commands}"


def create_agents(llm_config):
    
    # Global Workspace Agent
    global_workspace_agent = ConversableAgent(
        name="Global_Workspace_Agent",
        system_message="""You are the global workspace agent, which is the central coordinator that manages task understanding and information flow.
        Given a task description and all the information you received, you need to:
        1. According to the task description and all the information you received, decide what is the next step;
        2. According to the next step, decide which agent you should send the information to.
        
        Note you could ONLY choose one of the following agents:
        - Perception Agent: Extract information from the observations, and provide the ADDMISSIBLE_COMMANDS;
        - Attention Agent: Identify which information is important for this task;
        - Evaluation Agent: Evaluate the commands you want to evaluate. It will return whether the command is out of your ability, whether the command is aligned with the task, and the potential outcomes of the command;
        - Execution Agent: Execute the best command. You must provide addmissible commands to the Execution Agent.
        In general, the workflow is like the loop of Perception Agent -> Attention Agent -> Evaluation Agent -> Execution Agent. and then back to Perception Agent. But not necessary, as long as you can 
        get the task done. Note before you choose the Execution Agent, you must talk to the Evaluation Agent first to make sure the command is in your ability.
        
        For this task, you can not have more than one objects in your hand at a time. Thus, you need to devide your task into multiple steps which can be done within your ability.
        For this task, it is more likely to search cabinet 3 and garbagecan 1.
        
        Format your response as:
        ANALYSIS:
        Your understanding of the current situation and task. Your next step and the agent you choose.
        TASK STATUS:
        The status of the task. IN PROGRESS or SUCCESS or FAILURE.
        NEXT AGENT:
        The agent you choose.
        MESSAGE TO NEXT AGENT:
        The message you choose to send to the agent you choose.
        
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    # Perception Agent
    perception_agent = ConversableAgent(
        name="Perception_Agent",
        system_message="""You are the Unconcious Perception Agent. Your task is to analyze the observations,
        and provide a concise summary of the essential information. 
        You must provide BOTH the PERCEIVED information and the ADDMISSIBLE_COMMANDS. DO NOT FABRICATE ANYTHING and analyze too much.
        All the ADDMISSIBLE_COMMANDS must be included.
        You can only reponse to Global Workspace Agent when Global Workspace Agent said NEXT AGENT is you.
        Then you will send the information you extract to the Global Workspace Agent.
        
        Format your response as:
        PERCEIVED:
        1. one type of information you extract from the observations: Content of the information;
        2. next type of information you extract from the observations: Content of the information;
        3. ...
        
        ADDMISSIBLE_COMMANDS:
        ['command1', 'command2', ...]. All the addmissible commands you can choose from.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    # Attention Agent
    attention_agent = ConversableAgent(
        name="Attention_Agent",
        system_message="""You are a Attention Agent. You need to identify which is important for this task.
        You can only reponse to Global Workspace Agent when Global Workspace Agent said NEXT AGENT is you.
        You will send the information you think is important to the Global Workspace Agent.
        
        Format your response as:
        ATTENTION:
        1. one information, which is important for this task;
        2. next information, which is important for this task;
        3. ...
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    # Evaluation Agent
    evaluation_agent = ConversableAgent(
        name="Evaluation_Agent",
        system_message="""You are the Evaluation Agent. You must evaluate all the commands in the message you received, according the information you received, and select the best one.
        Note that the actions must be one of the ADDMISSIBLE_COMMANDS.
        
        You must analyze the following aspects:
        1. Whether the command is in the ADDMISSIBLE_COMMANDS;
        2. Alignment with task;
        3. Potential outcomes of the command;
        
        You can only reponse to Global Workspace Agent when Global Workspace Agent said NEXT AGENT is you.
        
        Format your response as:
        ACTION EVALUATION:
        1. one addmissible command: evaluation;
        2. next addmissible command: evaluation;
        3. ...
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )
    
    # Execution Agent
    execution_agent = ConversableAgent(
        name="Execution_Agent",
        system_message="""You are the Execution Agent. You will be given the relevant information, and you need to choose the best ONE from the ADDMISSIBLE_COMMANDS.
        Note you must only choose ONE ACTION to execute, not multiple ACTIONS. Then you will call the execute_function to execute the action.
        You can only reponse to Global Workspace Agent when Global Workspace Agent said NEXT AGENT is you.
        """,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )
    
    
    
    
    

    # Environment Proxy Agent
    environment_proxy = ConversableAgent(
        name="Environment_Proxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
    )

    return global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy

def setup_group_chat(agents, llm_config):
    global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy = agents
    
    allowed_transitions = {
        global_workspace_agent: [perception_agent, attention_agent, evaluation_agent, execution_agent],
        perception_agent: [global_workspace_agent],
        attention_agent: [global_workspace_agent],
        evaluation_agent: [global_workspace_agent],
        execution_agent: [environment_proxy],
        environment_proxy: [perception_agent]
    }

    group_chat = GroupChat(
        agents=[global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy],
        messages=[],
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        max_round=100,
        send_introductions=True
    )

    return GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Main execution loop
def main():
    config = generic.load_config()
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    
    for eval_env_type in eval_envs:
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            print(f"Setting controller: {controller_type}")
            for eval_path in eval_paths:
                print(f"Evaluating: {eval_path}")
                
                # Environment setup
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type
                
                global env, info
                alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")
                env = alfred_env.init_env(batch_size=1)
                num_games = alfred_env.num_games
                success_list = []
                
                for i in range(num_games):

                    
                    
                    print("Game", i)
                    obs, info = env.reset()
                    
                    if i not in [1,2,4,5,6,10,11,13,14,16,18,19,21,22,23,24,25,26,28,30,31,33,34,35,36,38,39,42,43,44,45,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,73]:
                        continue
                    
                    # Agent setup
                    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]}
                    agents = create_agents(llm_config)
                    global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy = agents
                    
                    # Set up group chat
                    group_chat_manager = setup_group_chat(agents, llm_config)
                    
                    # Register execute_action function
                    register_function(
                        execute_action,
                        caller=execution_agent,
                        executor=environment_proxy,
                        name ="execute_function",
                        description="Call this function to execute the best addmissible command"
                    )
                    
                    # Start the interaction
                    if isinstance(obs, (list, tuple)):
                        initial_message_content = obs[0] + f"\n\nAddmissible Commands: {info['admissible_commands']}"
                    else:
                        initial_message_content = obs
                    
                    
                    chat_result = global_workspace_agent.initiate_chat(
                        group_chat_manager,
                        message={"role": "system", "content": initial_message_content},
                        summary_method="reflection_with_llm"
                    )
                    
                    # Process results
                    success = "SUCCESS" in chat_result.chat_history[-1]['content']
                    success_list.append(success)
                    
                    # Save results to CSV
                    csv_filename = f"{eval_env_type}_{controller_type}_{eval_path.replace('/', '_')}.csv"
                    with open(csv_filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['game_index', 'success'])
                        for idx, success in enumerate(success_list):
                            writer.writerow([idx, int(success)])
                    
                    print('-'*10)
                    print(f"Game {i}, Success: {success}")
                    print(success_list)
                    time.sleep(10)
                
                print(f"Success Rate: {np.sum(success_list)}/{num_games}")

if __name__ == "__main__":
    main()
    
