import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import pickle
from helpers import register_function_lambda, get_best_candidate, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent


class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path=None, max_actions=50):
        self.retrieve_memory_agent = None
        self.guidance_agent = None
        self.record_guidance_agent = None
        self.task_agent = None
        self.command_evaluation_agent = None
        self.executor_agent = None
        self.echo_agent = None

        super().__init__(env, obs, info, llm_config, log_path, max_actions)

    def initialize_agents(self):
        # Retrieve Memory Agent
        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message="""You are the Retrieve Memory Agent. You task is ONLY to call the function `retrieve_memory` to retrieve the memory.
            DO NOT analyze any information such as task, history, admissible commands, guidance, etc.
            **RULES:**
            The TOOL you can only use is `retrieve_memory`.
            DO NOT call any other tools.
            You can only use `retrieve_memory` once per step.            
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
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )

        self.record_guidance_agent = ConversableAgent(
            name="Record_Guidance_Agent",
            system_message="""You are the Record Guidance Agent. You task is ONLY to call the function `record_guidance` to record the new guidance.
            DO NOT analyze any information such as task, history, admissible commands, etc. You only need to record the new guidance, not repeat the previous guidance.

            **IMPORTANT:**
            If 'No new guidance at this time.', do not call the function `record_guidance`.

            **RULES:**
            The TOOL you can only use is `record_guidance`.
            DO NOT call any other tools.
            You can only use `record_guidance` once per step.            
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
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
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
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
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
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
            is_termination_msg=is_termination_msg_generic
        )

        self.echo_agent = get_echo_agent(self.llm_config)

        # Agent descriptions
        self.task_agent.description = "analyzes the task and proposes a plan to accomplish the task"
        self.retrieve_memory_agent.description = "retrieves the memory"
        self.guidance_agent.description = "analyzes the history and proposes guidance for your capability, envrionment, and task"
        self.record_guidance_agent.description = "records the new guidance"
        self.command_evaluation_agent.description = "evaluates the outcome of the command"
        self.executor_agent.description = "executes actions and returns observations"

    def register_functions(self):
        log_paths = self.get_log_paths()

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
                with open(log_paths['history_path'], "a+") as f:
                    f.write(f"action: 'None'. observation: '{self.obs[0]}'\n")
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]
                with open(log_paths['history_path'], "a+") as f:
                    f.write(f"action: '{action}'. observation: '{self.obs[0]}'\n")

            # save the admissible commands into a txt file
            # with open(log_paths['admissible_commands_path'], "w") as f:
            #     f.write(f"{admissible_commands}\n")

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
            with open(log_paths['guidance_path'], "a+") as f:
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

            memory_information += "Task: \n"
            if os.path.exists(log_paths['task_path']):
                with open(log_paths['task_path'], "r") as f:
                    memory_information += f.read()
            else:
                memory_information += "No task information found.\n"

            # latest 10 steps. last 10 lines
            memory_information += "\nRecent 5 steps History: \n"
            if os.path.exists(log_paths['history_path']):
                with open(log_paths['history_path'], "r") as f:
                    for line in f.readlines()[-5:]:
                        memory_information += line
            else:
                memory_information += "No history information found.\n"

            # if os.path.exists(log_paths['admissible_commands_path']):
            #     memory_information += "\nAdmissible commands for current step: \n"
            #     with open(log_paths['admissible_commands_path'], "r") as f:
            #         memory_information += f.read()
            # else:
            #     memory_information += "No admissible commands information found.\n"

            if os.path.exists(log_paths['guidance_path']):
                memory_information += "\nGuidance: \n"
                with open(log_paths['guidance_path'], "r") as f:
                    memory_information += f.read()
            else:
                memory_information += "\nNo guidance information found.\n"

            return memory_information

        register_function_lambda(
            {r"execute_action": execute_action,
             r"record_guidance": record_guidance,
             r"retrieve_memory": retrieve_memory},
            [self.echo_agent]
        )

    def initialize_groupchat(self, max_chat_round=2000):
        log_paths = self.get_log_paths()

        def state_transition(last_speaker, groupchat):
            messages = groupchat.messages

            with open(log_paths['message_path'], "wb") as f:
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
                self.echo_agent
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
