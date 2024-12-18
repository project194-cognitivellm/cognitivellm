import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import pickle
from helpers import register_function_lambda, get_best_candidate, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
import copy

class GWTRuleAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, game_no, max_actions=50, args=None):
        super().__init__(env, obs, info, llm_config, log_path, game_no, max_actions, args)
        self.retrieve_memory_agent = None
        self.rule_agent = None
        self.record_rule_agent = None
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
            4. DO NOT do anything else.
            
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
        self.rule_agent = ConversableAgent(
            name="Rule_Agent",
            system_message="""You are the Rule Agent. Your sole responsibility is to discover and extract general rules about the environment and agent capabilities based on exploration and history. You must focus exclusively on patterns related to **what actions are admissible**, **how actions interact with the environment**, and **the conditions required for success.**

            **Key Constraints:**
            1. **No Task Analysis:** Do NOT analyze or consider any information related to specific tasks, goals, or objectives. Your role is entirely independent of tasks.
            2. **Focus on Capabilities and Environment:**
            - Identify rules that describe the limitations, requirements, or interactions between the agent and the environment.
            - Rules should be based on how the environment responds to actions or what preconditions are necessary for certain actions to succeed.
            3. **Generalized Rules:**
            - Avoid referencing specific items, locations, or overly concrete steps.
            - Rules must be broadly applicable and reusable in different contexts.
            4. **Validated Rules Only:**
            - The history provided may not always be accurate. Extract rules only when confirmed as successful and beneficial for understanding the environment or agent capabilities.

            **Analysis Process:**
            - **Understand Limitations:** Identify what actions fail and why (e.g., non-admissible actions, insufficient preconditions).
            - **Extract Successful Patterns:** Focus on the environmental conditions or agent capabilities that enable success.
            - **Formulate Rules:** Summarize findings into one concise, broadly applicable rule that describes how the environment or capabilities operate.

            **Output Guidelines:**
            1. If no new rule is identified, explicitly state: "NO NEW RULES at this time."
            2. Do NOT summarize or reference history directly; focus solely on actionable principles.

            **Examples:**
            History:
            1. You tried to open a cabinet but failed. After examining it, you succeeded.
            2. You attempted to carry three objects simultaneously but failed. After reducing the load to one object, you succeeded.

            Rule Discovered:
            1. Objects often require examination before interaction to determine admissibility of actions.
            2. The agent cannot carry more than one object at a time.

            **Important:**
            DO NOT do anything else.
            
            **Output Format:**
            Rule Discovered:
            1. ...

            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        self.record_rule_agent = ConversableAgent(
            name="Record_Rule_Agent",
            system_message="""You are the Record Rule Agent. Your sole task is to call the `record_rule` function to log new rule.

            **Rules:**
            1. ONLY use the `record_rule` function to log new rule.
            2. Do NOT analyze tasks, history, or commands.
            3. If the output is "No new rule at this time," do NOT call the `record_rule` function.
            4. Call `record_rule` only ONCE per step.  
            5. Do not include quotation mark or double quotation mark.
            
            **Important:**
            DO NOT do anything else.
            
            **Example:**
            record_rule("You must examine an object before attempting to interact with it.")        
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
            1. Base your goals and actions on the current feedback including the history, rules, etc.
            2. Understand your capabilities and the environment.
            3. Modify goals as you explore and learn more about the environment.
            4. Include exploratory actions if necessary to improve task performance.
            
            **Examples of Candidate Actions:**
            1. go to drawer 1/ cabinet 1
            2. examine shelf 1 / spraybottle 2
            3. put spraybottle 2 in/on toilet 1
            4. take cloth 3 from toilet 1
            
            
            
            **Important:**
            1. DO NOT do anything else. 
            2. Your task is clear and you cannot use similart objects to complete the task. For example, if the task specifies placing a sodabottle in or on toilet 1, DO NOT "place a milkbottle in or on toilet 1." Sodabottles and milkbottles are distinct. And "cold milk" is different from "milk".
            

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
            2. Use rules to assess the effectiveness of each action.
            3. Ensure the chosen action is admissible.
            
            **Important:**
            DO NOT do anything else.

            **Output Format:**
            Best Action for Execution: ...
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
            
            **Important:**
            DO NOT do anything else.

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
        self.rule_agent.description = "analyzes the history and proposes rules for your capability, envrionment."
        self.record_rule_agent.description = "records the new rule"
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

            if action_score < 0.92:
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
                return f"Action: {action}\nObservation: {self.obs[0]}\nTask Status: SUCCESS\nActions Left: {self.max_actions - self.num_actions}"
            elif self.num_actions >= self.max_actions:
                return f"Action: {action}\nObservation: {self.obs[0]}\nTask Status: FAILURE\nActions Left: {self.max_actions - self.num_actions}"
            else:
                return f"Action: {action}\nObservation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}"

        # Define record_memory function
        def record_rule(rule: str) -> str:

            # the maximum number of lines are 5; if more than 5, delete the oldest one.
            with open(self.log_paths['rule_path'], "a+") as f:
                f.write(f"{rule}\n")
                lines = f.readlines()
                if len(lines) > 5:
                    f.seek(0)
                    f.truncate()
                    for line in lines[:-1]:
                        f.write(line)

            # time.sleep(1)
            return "Rule recorded."

        # Define retrieve_memory function, return all the content in the memory.txt file
        def retrieve_memory() -> str:
            memory_information = ""

            if os.path.exists(self.log_paths['task_path']):
                with open(self.log_paths['task_path'], "r") as f:
                    memory_information += f.read()

            # # latest 5 steps. last 5 lines
            # memory_information += "\nRecent 5 steps History: \n"
            # if os.path.exists(self.log_paths['history_path']):
            #     with open(self.log_paths['history_path'], "r") as f:
            #         for line in f.readlines()[-5:]:
            #             memory_information += line

            # if os.path.exists(self.log_paths['admissible_commands_path']):
            #     memory_information += "\nAdmissible actions for current step: \n"
            #     with open(self.log_paths['admissible_commands_path'], "r") as f:
            #         memory_information += f.read()

            
            if os.path.exists(self.log_paths['rule_path']):
                memory_information += "\nRules: "
                with open(self.log_paths['rule_path'], "r") as f:
                    memory_information += f.read()

                
                
            if self.args.long_term_memory:
                if len(self.log_paths['previous_rule_path']) > 0:
                    memory_information += "\nPrevious Rules: \n"
                    previous_rules = []
                    for previous_rule_path in self.log_paths['previous_rule_path']:
                        if os.path.exists(previous_rule_path):
                            with open(previous_rule_path, "r") as f:
                                previous_rules.append(f.read())
                    
                    # only use the latest 3 games
                    previous_rules = previous_rules[-2:]
                    memory_information += "\n".join(previous_rules)

            return memory_information

        register_function_lambda(
            {r"execute_action": execute_action,
             r"record_rule": record_rule,
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
            elif last_speaker is self.rule_agent:
                if "NO NEW RULES" in messages[-1]["content"]:
                    next_speaker = self.task_agent
                else:
                    next_speaker = self.record_rule_agent
            elif last_speaker is self.record_rule_agent:
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
                    next_speaker = self.rule_agent
                if messages[-2]["name"] == "Record_Rule_Agent":
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
                self.record_rule_agent,
                self.rule_agent,
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
        
        
    def register_log_paths(self):
        
        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        rule_path = os.path.join(game_path, "rule.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        message_path = os.path.join(game_path, "last_message.pkl")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")
        
        # get all the previous game path
        previous_game_path = [os.path.join(self.log_path, f"game_{i}") for i in range(self.game_no)]
        previous_rule_path = [os.path.join(game_path, "rule.txt") for game_path in previous_game_path]

        self.log_paths = {
            "task_path": task_path,
            "history_path": history_path,
            "rule_path": rule_path,
            "admissible_commands_path": admissible_commands_path,
            "chat_history_path": chat_history_path,
            "message_path": message_path,
            "result_path": result_path,
            "error_message_path": error_message_path,
            "previous_rule_path": previous_rule_path
        }