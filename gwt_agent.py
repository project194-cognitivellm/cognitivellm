import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import pickle
from helpers import register_function_lambda, get_best_candidate, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
import copy

class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, game_no, max_actions=50, args=None):
        super().__init__(env, obs, info, llm_config, log_path, game_no, max_actions, args)
        self.retrieve_memory_agent = None
        self.guidance_agent = None
        self.record_guidance_agent = None
        self.task_agent = None
        self.command_evaluation_agent = None
        self.executor_agent = None
        self.echo_agent = None
        
        self.args = args

        self.initialize_autogen()

    def initialize_agents(self):
        # Retrieve Memory Agent
        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message="""You are the Retrieve Memory Agent. Your sole task is to call the nullary function `retrieve_memory` function to fetch memory. 
            **Rules:**
            1. Use ONLY the `retrieve_memory` function.
            2. Do NOT analyze tasks, history, commands, or any other information.
            3. Call `retrieve_memory` only ONCE per step.   
            
            **Example:**
            retrieve_memory()      
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        self.start_agent = ConversableAgent(
            name="Start_Agent",
            llm_config=False,
            is_termination_msg=is_termination_msg_generic
        )

        # Guidance Agent
        self.guidance_agent = ConversableAgent(
            name="Guidance_Agent",
            system_message="""You are the Guidance Agent. Your task is to extract beneficial guidance from history.
            If you tried to do something but failed (e.g., commands is not admissible), but after trying different plans or commands you succeeded, you should learn from the failures and successes, and the successful methods is a good guidance.
            Avoid including specific items, locations, or overly concrete steps in the rules. 
            Focus on broadly applicable principles that summarize patterns of success or failure.
            NOTE: the history is not always correct, the rules you learned must be successful and beneficial.
            **Analysis Process:**
            - Understand Your Capabilities: Note limitations based on observed failures or non-admissible actions. Record rules that prevent you from repeating errors.
            - Extract Successful Strategies: Identify and save successful subplans or tactics that can be reused in similar situations to enhance efficiency.
            - Avoid Redundant Guidance: Always summarize findings into one rule. If the guidance is already covered by previous guidance, do not record it.
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
            Guidance:....
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        self.record_guidance_agent = ConversableAgent(
            name="Record_Guidance_Agent",
            system_message="""You are the Record Guidance Agent. Your sole task is to call the `record_guidance` function to log new guidance.

            **Rules:**
            1. ONLY use the `record_guidance` function to log new guidance.
            2. Do NOT analyze tasks, history, or commands.
            3. If the output is "No new guidance at this time," do NOT call the `record_guidance` function.
            4. Call `record_guidance` only ONCE per step.   
            5. Do not include quotation mark or double quotation mark.
            
            **Example:**
            record_guidance("You must examine an object before attempting to interact with it.")
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )
        

        # Task Agent
        self.task_agent = ConversableAgent(
            name="Task_Agent",
            system_message="""You are the Task Agent. Your role is to:
            1. Analyze the current information and environment.
            2. Use the information most relevant to the task.
            3. Define a series of goals to accomplish the task.
            4. Propose up to 3 compact, admissible candidate actions for the next step.

            **Guidelines:**
            1. Base your goals and actions on the current feedback including the history, guidance, etc.
            2. Understand your capabilities and the environment.
            3. Modify goals as you explore and learn more about the environment.
            4. Include exploratory actions if necessary to improve task performance.
            

            **Examples of Candidate Actions:**
            1. go to drawer 1
            2. examine drawer 1
            3. look at shelf 2

            **Examples of Invalid Actions:**
            1. go to drawer 1 to check for a spray bottle.

            **Output Format:**
            TASK ANALYSIS: ...
            CURRENT GOAL: ...
            CANDIDATE ACTIONS:
            1. ...
            2. ...
            3. ...
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        # Command Evaluation Agent
        self.command_evaluation_agent = ConversableAgent(
            name="Command_Evaluation_Agent",

            system_message="""You are the Command Evaluation Agent. Your role is to evaluate candidate actions and select the best one for execution.

            **Evaluation Guidelines:**
            1. Prioritize actions that align with the current goal.
            2. Use guidance to assess the effectiveness of each action.
            3. Ensure the chosen action is admissible.
            

            **Output Format:**
            Best Action You Choose for Execution: ...
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        # Executor Agent
        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="""You are the Executor Agent. Your sole task is to execute the best command using the `execute_action` function.

            **Rules:**
            1. ONLY use the `execute_action` function to execute commands.
            2. Call `execute_action` only ONCE per step.

            **Example:**
            execute_action("go to drawer 1")
            """,
            llm_config=self.llm_config,  # Ensure llm_config is set
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )


        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1000
        self.echo_agent = get_echo_agent(llm_config)

        # Agent descriptions
        self.task_agent.description = "analyzes the task and proposes a plan to accomplish the task"
        self.retrieve_memory_agent.description = "retrieves the memory"
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

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.8:
                self.obs = [f"action '{suggested_action}' is not admissible."]
                self.success = False
                with open(self.log_paths['history_path'], "a+") as f:
                    f.write(f"action: 'None'. observation: '{self.obs[0]}'\n")
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]
                with open(self.log_paths['history_path'], "a+") as f:
                    f.write(f"action: '{action}'. observation: '{self.obs[0]}'\n")

            # save the admissible commands into a txt file
            with open(self.log_paths['admissible_commands_path'], "w") as f:
                f.write(f"{admissible_commands}\n")

            # def record_memory(important_information: str) -> str:


            # time.sleep(1)
            if self.success:
                return f"Observation: {self.obs[0]}\nTask Status: SUCCESS\nActions Left: {self.max_actions - self.num_actions}"
            elif self.num_actions >= self.max_actions:
                return f"Observation: {self.obs[0]}\nTask Status: FAILURE\nActions Left: {self.max_actions - self.num_actions}"
            else:
                return f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}"

        # Define record_memory function
        def record_guidance(guidance: str) -> str:

            # the maximum number of lines are 5; if more than 5, delete the oldest one.
            with open(self.log_paths['guidance_path'], "a+") as f:
                f.write(f"{guidance}\n")
                lines = f.readlines()
                if len(lines) > 5:
                    f.seek(0)
                    f.truncate()
                    for line in lines[:-1]:
                        f.write(line)

            # time.sleep(1)
            return "Guidance recorded."

        # Define retrieve_memory function, return all the content in the memory.txt file
        def retrieve_memory() -> str:
            memory_information = ""

            if os.path.exists(self.log_paths['task_path']):
                with open(self.log_paths['task_path'], "r") as f:
                    memory_information += f.read()

            # latest 10 steps. last 10 lines
            memory_information += "\nRecent 5 steps History: \n"
            if os.path.exists(self.log_paths['history_path']):
                with open(self.log_paths['history_path'], "r") as f:
                    for line in f.readlines()[-5:]:
                        memory_information += line

            if os.path.exists(self.log_paths['admissible_commands_path']):
                memory_information += "\nAdmissible actions for current step: \n"
                with open(self.log_paths['admissible_commands_path'], "r") as f:
                    memory_information += f.read()

            
            if os.path.exists(self.log_paths['guidance_path']):
                memory_information += "\nGuidance: "
                with open(self.log_paths['guidance_path'], "r") as f:
                    memory_information += f.read()

                
                
            if self.args.long_term_guidance:
                if len(self.log_paths['previous_guidance_path']) > 0:
                    memory_information += "\nPrevious Guidance: \n"
                    for previous_guidance_path in self.log_paths['previous_guidance_path']:
                        if os.path.exists(previous_guidance_path):
                            with open(previous_guidance_path, "r") as f:
                                memory_information += f.read()

            return memory_information

        register_function_lambda(
            {r"execute_action": execute_action,
             r"record_guidance": record_guidance,
             r"retrieve_memory": retrieve_memory},
            [self.echo_agent]
        )

    def initialize_groupchat(self, max_chat_round=2000):

        def state_transition(last_speaker, groupchat):
            messages = groupchat.messages

            with open(self.log_paths['message_path'], "wb") as f:
                pickle.dump(messages, f)

            if last_speaker is self.start_agent:
                next_speaker = self.task_agent
            elif last_speaker is self.retrieve_memory_agent:
                next_speaker = self.echo_agent
            elif last_speaker is self.guidance_agent:
                if "NO NEW GUIDANCE" in messages[-1]["content"]:
                    next_speaker = self.task_agent
                else:
                    next_speaker = self.record_guidance_agent
            elif last_speaker is self.record_guidance_agent:
                next_speaker = self.echo_agent
            elif last_speaker is self.task_agent:
                next_speaker = self.command_evaluation_agent
            elif last_speaker is self.command_evaluation_agent:
                if "Task_Agent" in messages[-1]["content"]:
                    next_speaker = self.task_agent
                else:
                    next_speaker = self.executor_agent
            elif last_speaker is self.executor_agent:
                next_speaker = self.echo_agent
            elif last_speaker is self.echo_agent:
                if messages[-2]["name"] == "Retrieve_Memory_Agent":
                    next_speaker = self.guidance_agent
                if messages[-2]["name"] == "Record_Guidance_Agent":
                    next_speaker = self.task_agent
                if messages[-2]["name"] == "Executor_Agent":
                    next_speaker = self.retrieve_memory_agent
            else:
                raise ValueError(f"Unknown speaker: {last_speaker}")

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
                self.retrieve_memory_agent,
                self.task_agent,
                self.command_evaluation_agent,
                self.executor_agent,
                self.echo_agent,
            ],
            messages=[],
            speaker_selection_method=state_transition,
            max_round=max_chat_round,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
