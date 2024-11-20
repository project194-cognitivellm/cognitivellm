import os 
import time 
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager 
import numpy as np
import alfworld.agents.environment as environment 
import alfworld.agents.modules.generic as generic 
from nltk.translate.bleu_score import sentence_bleu

# Step 1: create agent 
def create_agents(llm_config):     
    # Global Workspace Agent     
    global_workspace_agent = ConversableAgent(         
        name="Global_Workspace_Agent",         
        system_message="""You are the global workspace agent, which coordinates the overall task and information flow.         
        You need to decide the next step of the task and delegate it to one of the subsystems:          
        - Perception Agent: Extract relevant information from the environment.         
        - Attention Agent: Focus on which information is important for the task.         
        - Evaluation Agent: Evaluate the commands and align them with the task.         
        - Execution Agent: Execute the best possible command.         
        Format your response as:          
        ANALYSIS: Your analysis of the task and next step.         
        TASK STATUS: IN PROGRESS or SUCCESS or FAILURE.         
        NEXT AGENT: The agent you choose.         
        MESSAGE TO NEXT AGENT: The message you send to the next agent.""",         llm_config=llm_config,         
        human_input_mode="NEVER",         
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )

    # Perception Agent     
    perception_agent = ConversableAgent(         
        name="Perception_Agent",         
        system_message="""You are the Unconscious Perception Agent. Your job is to filter out unnecessary information and          
        extract key details. You will pass both the perceived information and admissible commands to the Global Workspace Agent.""",        
        llm_config=llm_config,         
        human_input_mode="NEVER",         
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )

    # Attention Agent     
    attention_agent = ConversableAgent(         
        name="Attention_Agent",         
        system_message="""You are the Attention Agent. Your task is to analyze which pieces of information are critical         
        for the ongoing task and highlight them to the Global Workspace Agent.""",         llm_config=llm_config,         
        human_input_mode="NEVER",         
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )

    # Evaluation Agent     
    evaluation_agent = ConversableAgent(         
        name="Evaluation_Agent",         
        system_message="""You are the Evaluation Agent. Your task is to evaluate the admissible commands and choose the best one.         
        You need to evaluate the alignment with the task and its potential outcomes.""",         llm_config=llm_config,         
        human_input_mode="NEVER",         
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )

    # Execution Agent     
    execution_agent = ConversableAgent(         
        name="Execution_Agent",         
        system_message="""You are the Execution Agent. You will execute the best admissible command chosen by the Global Workspace Agent.""",         
        llm_config=llm_config,         
        human_input_mode="NEVER",        
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )

     # Environment Proxy     
    environment_proxy = ConversableAgent(         
        name="Environment_Proxy",         
        llm_config=False,         
        human_input_mode="NEVER",         
        is_termination_msg=lambda msg: msg["content"] and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])     
    )      
    return global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy  

# Step 2: groupchat
def setup_group_chat(agents, llm_config):     
    global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy = agents      
    
    allowed_transitions = {        
         global_workspace_agent: [perception_agent, attention_agent, evaluation_agent, execution_agent],         
         perception_agent: [global_workspace_agent],         
         attention_agent: [global_workspace_agent],         
         evaluation_agent: [global_workspace_agent],         
         execution_agent: [environment_proxy],        
         environment_proxy: [perception_agent]     }      
    
    group_chat = GroupChat(         
        agents=[global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy],         
        messages=[],         
        allowed_or_disallowed_speaker_transitions=allowed_transitions,         speaker_transitions_type="allowed",         
        max_round=100,         
        send_introductions=True     
    )

    return GroupChatManager(groupchat=group_chat, llm_config=llm_config)  

# Step 3: main
def main():     
    success_count = 0      
    failure_count = 0  
    # Load ALFWorld config     
    config = generic.load_config()    
    eval_paths = config["general"]["evaluate"]["eval_paths"]     
    eval_envs = config["general"]["evaluate"]["envs"]     
    controllers = config["general"]["evaluate"]["controllers"]     
    repeats = config["general"]["evaluate"]["repeats"]

    llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}     
    agents = create_agents(llm_config)     
    
    global_workspace_agent, perception_agent, attention_agent, evaluation_agent, execution_agent, environment_proxy = agents     
    group_chat_manager = setup_group_chat(agents, llm_config)
   
    # Step through all environments and controllers for evaluation    
    for eval_env_type in eval_envs:         
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):            
            print(f"Setting controller: {controller_type}")             
            for eval_path in eval_paths:                 
                print(f"Evaluating: {eval_path}")                 
                config["general"]["evaluate"]["env"]["type"] = eval_env_type                 
                config["dataset"]["eval_ood_data_path"] = eval_path                 
                config["controller"]["type"] = controller_type                  
                alfred_env = getattr(environment, config["general"]["evaluate"]["env"]["type"])(config, train_eval="eval_out_of_distribution")                 
                env = alfred_env.init_env(batch_size=1)                 
                num_games = alfred_env.num_games                 
                max_steps = 100     

                # Interaction loop for each game                 
                for i in range(num_games):                     
                    print("="*50)                     
                    print(f"Starting Game {i+1}/{num_games}...")                     
                    start_time = time.time()                                
                    
                    obs, info = env.reset()                     
                    print(f"Game {i+1} Initialized: Environment Reset")                     
                    print(f"Admissible Commands: {info['admissible_commands'][0]}")                      
                                      
                    for step in range(max_steps):                         
                        print(f"Step {step+1}/{max_steps}")                                                  
                       
                        obs, info = perception_agent.get_perception_info(env)                         
                        print(f"Game {i+1}, Step {step+1}: Perception: {obs}")                          
                        
                        
                        attention_info = attention_agent.focus_attention(info)                         
                        print(f"Attention focused on: {attention_info}")      

                                               
                        action_feedback = execution_agent.perform_action(attention_info, env)                         
                        print(f"Game {i+1}, Step {step+1} Action Feedback: {action_feedback}")                          
                        
                                                
                        if action_feedback['dones']:                             
                            if action_feedback['scores'] >= 50:                              
                                success_count += 1                                 
                                print(f"Game {i+1} Success! Score: {action_feedback['scores']}")                             
                            else:                                 
                                failure_count += 1                                 
                                print(f"Game {i+1} Failure. Score: {action_feedback['scores']}")                             
                            break                      
                    
                                        
                    elapsed_time = time.time() - start_time                     
                    print(f"Time for Game {i+1}: {elapsed_time:.2f} seconds")                     
                    print("="*50 + "\n")      
      
    print("="*50)     
    print(f"Total Successes: {success_count}")     
    print(f"Total Failures: {failure_count}")     
    print("="*50)

def execute_action(suggested_action: str, info: dict, env) -> dict:     
    assert len(list(info['admissible_commands'])) == 1     
    admissible_commands = list(info['admissible_commands'][0])          
        
    action = get_best_candidate(suggested_action, admissible_commands)     
    obs, scores, dones, info = env.step([action])          
    
        
    result = {         
        'action': action,         
        'admissible_commands': admissible_commands,         
        'scores': scores[0],         
        'dones': dones[0],         
        'observation': obs[0]     
    }          
    return result

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

if __name__ == "__main__":     
    main()