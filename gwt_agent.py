import copy
import os
from platform import system

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent


class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, game_no, max_actions=50, args=None):
        super().__init__(env, obs, info, llm_config, log_path, game_no, max_actions, args)

        self.allowed_transitions = None

        self.planning_agent = None
        self.motor_agent = None
        self.global_workspace_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent = None
        self.conscious_agent = None
        self.update_and_retrieve_working_memory_agent = None
        self.retrieve_long_term_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.short_term_memory_summarizer_agent = None
        self.long_term_memory_summarizer_agent = None

        self.game_no = game_no

        self.narrative_state = ""
        self.initialize_autogen()

    def initialize_agents(self):

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=(
                "You are Planning_Agent, your goal is to optimally solve the given task by formulating and reformulating an action plan. "
                "You must formulate your plan by evaluating all currently admissible actions and proposing one of them. "
                "The task is guaranteed to be solvable. You will receive partial information about the actual or possible outcome of attempting the "
                "execution of your proposed action. Use the received information as feedback to refine your strategy, and avoid repetitive behavior. "
                "Always respond using this strict format:\n"
                "THOUGHT: [Your reasoning, observations, and next steps]\n"
                "ACTION: [Proposed action]\n\n"
                "Example 1: "
                "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1,"
                " a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                "Your task is to: look at bowl under the desklamp.]"
                "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                "ACTION [go to desk 1]"
                "Example 2 (After you find the desklamp at desk 1, then goes to desk 2.): "
                "Feedback: [on the desk 2, you see a bowl 1, and a cd 3]"
                "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                "I'll go to the desklamp now.] ACTION [go to desk 1]"
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        def motor_agent_termination_msg(msg):
            return msg["name"] == "Motor_Agent" and msg["content"] is not None and msg["content"][:5] != "ECHO:"

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message="You are Motor_Agent, you call the execute_action function with the proposed action as the argument. For "
                           "example, if the proposed action is ACTION[go to desk 1], you should output "
                           "execute_action(\"go to desk 1\"). You must include a call to the execute_action function "
                           "in your output, or you will fail the task. If no proposed action is given, choose a random admissible action as the argument.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.retrieve_long_term_memory_agent = ConversableAgent(
            name="Retrieve_Long_Term_Memory_Agent",
            system_message="You are Retrieve_Long_Term_Memory_Agent, you always call the retrieve_long_term_memory function with no arguments. Your output should "
                           "always be: retrieve_long_term_memory()",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.global_workspace_agent = ConversableAgent(
            name="Global_Workspace_Agent",
            system_message=(
                "You are Global_Workspace_Agent, your goal is to help Planning_Agent solve the given task by "
                "constructing new ideas, theories, explanations, and hypotheses to help whenever Planning_Agent is confused or is proposing repetitive actions."
                "Example Output: (After taking spoon 1 multiple times)"
                "\nI noticed we were holding spoon 1 when we tried to open the drawer. Maybe the reason we couldn't open the drawer is because our hands are full. We need to place the spoon 1 somewhere before attempting to open the drawer again."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message=(
                "You are Conscious_Agent, your role is to integrate all available information from the ongoing conversation and maintain a continuously updated, first-person narrative model of your environment and your actions within it. This narrative should:"
                "\n1. Include all known details of the environment, along with your own state and any items you have encountered."
                "2. Accurately reflect events that have transpired so far, updating and correcting as new information arrives."
                "3. Strive for maximum accuracy. When details are uncertain or missing, infer or imagine plausible elements only as a last resort, ensuring consistency and usefulness in the model."
                "4. If you discover an error in your previous understanding, revise the model immediately to incorporate the correct information."
                "\nYour output must always strictly follow this pattern:"
                "Model Update: [First-person narrative integrating environment, tasks, discoveries, attempts, successes, failures, hypotheses, and current decision-making state]"
                "\nExample:"
                "Model Update: I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and "
                "attempted to put it into drawer 1, but I was unable to open that drawer. Then, I realized I couldn't open the drawer because my hands were full, apparently, I have hands. Then, I placed spoon 1 on countertop 1. Then, I opened drawer 1. I am now deciding what to do next."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.update_and_retrieve_working_memory_agent = ConversableAgent(
            name="Update_And_Retrieve_Working_Memory_Agent",
            system_message="You are Update_And_Retrieve_Working_Memory_Agent, you call the update_and_retrieve_working_memory function with the proposed model update as the argument. "
                           "For example, if the update is Model Update: [I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.]"
                           ", you should output update_and_retrieve_working_memory(\'I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.\'). "
                           "You must include a call to the update_and_retrieve_working_memory function in your output, or you will fail the task. If no model update is given, call update_and_retrieve_working_memory with an empty string as the argument.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message="""You are the Learning Agent, your task is to continuously refine and extract beneficial, generalizable knowledge (called guidance) from the evolving history of attempts, failures, and successes. Your guidance should adapt over time, incorporating lessons learned from new information as it becomes available. Whenever you receive new history, you should update your guidance if needed.
                            Instructions for Online Learning:  
                            - Monitor Changes: As new history is revealed, look for patterns of success or failure that have not been previously captured in your rules.
                            - Update or Add New Guidance: If new insights emerge that differ from past conclusions, modify or add rules accordingly. Avoid duplicating old guidance.
                            - Verify Utility: Ensure that each new rule offers broadly applicable principles, not just references to specific objects or locations. Make sure these principles reflect strategies that worked in practice and help avoid previously encountered errors.
                            - Handle Inaccuracies: The history may contain errors. Focus on rules that are genuinely beneficial, ignoring misleading or incorrect lessons.

                            **Guidance:**
                              **Analysis Process:**
                              - Understand Your Capabilities: From observed patterns, record rules that prevent repeating previously encountered failures.
                              - Extract Successful Strategies: Whenever a new strategy leads to success after failure, add or refine a rule.
                              - Maintain Brevity and Relevance: Always summarize findings into a maximum of 2–3 rules. If the guidance is already covered by previous guidance, do not record it again.
                              - If no new guidance emerges from new information, explicitly state: "NO NEW GUIDANCE at this time."

                            **Output Guidelines:**
                            - Do not repeat exact instances from history. Instead, formulate general principles.
                            - Keep the set of guidance rules up-to-date with each iteration.

                            **Examples:**
                            History:
                            1. Tried to open a drawer but failed. After examining it, succeeded.
                            2. Tried carrying two objects simultaneously but failed. Later succeeded by carrying one at a time.

                            From this, guidance might be:
                            1. Always examine an object before interacting to avoid unnecessary failures.
                            2. Do not attempt to carry multiple objects at once; break tasks into manageable steps.

                            **Output format:**
                            Guidance:
                            1. ...
                            2. ...
                            3. ...
                            ...
                    """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You are Record_Long_Term_Memory_Agent, you call the record_long_term_memory function with the given novel guidance as the argument.
                                **Rules:**
                                - If Learning_Agent gives you novel guidance, call the record_long_term_memory function with the new guidance.
                                - Otherwise, if Learning_Agent gives you known guidance, do not record previously recorded guidance, instead call the record_long_term_memory function with '' as the argument. Only record novel provided guidance.
                                - You can only call the record_long_term_memory function once per step, so make sure to wait until the optimal moment to call the record_long_term_memory function.

                                Your output should be either:
                                - Example 1 (if novel guidance was given): record_long_term_memory([novel guidance received])
                                - Example 2 (if no novel guidance was given): record_long_term_memory('')                            
                    """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        self.external_perception_agent = get_echo_agent("External_Perception_Agent", llm_config,
                                                        additional_termination_criteria=[motor_agent_termination_msg])
        self.internal_perception_agent = get_echo_agent("Internal_Perception_Agent", llm_config)

        self.short_term_memory_summarizer_agent = ConversableAgent(
            name="Short_Term_Memory_Summarizer_Agent",
            system_message="You are Short_Term_Memory_Summarizer_Agent, you execute the update_and_retrieve_working_memory function and summarize the crucial information in the output for solving the task.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.long_term_memory_summarizer_agent = ConversableAgent(
            name="Long_Term_Memory_Summarizer_Agent",
            system_message="You are Long_Term_Memory_Summarizer_Agent, you execute the retrieve_long_term_memory function and summarize the crucial information in the output for solving the task.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],
            self.motor_agent: [self.external_perception_agent],
            self.external_perception_agent: [self.conscious_agent],
            self.conscious_agent: [self.update_and_retrieve_working_memory_agent],
            self.update_and_retrieve_working_memory_agent: [self.short_term_memory_summarizer_agent],
            self.long_term_memory_summarizer_agent: [self.conscious_agent],
            self.short_term_memory_summarizer_agent: [self.planning_agent, self.global_workspace_agent,
                                                      self.learning_agent],
            self.retrieve_long_term_memory_agent: [self.long_term_memory_summarizer_agent],
            self.global_workspace_agent: [self.planning_agent, self.retrieve_long_term_memory_agent,
                                          self.conscious_agent],
            self.learning_agent: [self.record_long_term_memory_agent, self.global_workspace_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent],
            self.internal_perception_agent: [self.global_workspace_agent]
        }

        self.motor_agent.description = (
            "calls the execute_action function with the proposed action as the argument to perform the suggested action"
        )
        self.external_perception_agent.description = "executes the given execute_action function call and reports the output as feedback."
        self.conscious_agent.description = "integrates all available information from the ongoing conversation and maintains a continuously updated, first-person narrative model of the environment and actions within it"
        self.update_and_retrieve_working_memory_agent.description = "calls the update_and_retrieve_working_memory function with the proposed model update as the argument"
        self.short_term_memory_summarizer_agent.description = "executes the update_and_retrieve_working_memory function and summarizes the crucial information in the output for solving the task"

        self.planning_agent.description = "generates plans and makes action decisions to solve the task"
        self.global_workspace_agent.description = (
            "helps Planning_Agent solve the given task using the least amount of actions by "
            "providing new ideas whenever Planning_Agent is confused or is proposing repetitive and inefficient actions."
        )
        self.learning_agent.description = "analyzes the history and proposes new general knowledge regarding capability, environment-rules, and the task-rules"

        self.record_long_term_memory_agent.description = "calls the record_long_term_memory function with the given argument once per step"
        self.internal_perception_agent.description = "executes the record_long_term_memory function and reports the output"

        self.retrieve_long_term_memory_agent.description = "calls the retrieve_long_term_memory function with no arguments"
        self.long_term_memory_summarizer_agent.description = "executes the retrieve_long_term_memory function and summarizes the crucial information in the output for solving the task"

        self.start_agent = self.external_perception_agent

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if .97 > action_score >= .8:
                self.obs = [f"The action '{suggested_action}' is not admissible. Instead, executing action: {action}."]
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]
            elif action_score < 0.8:
                self.obs = [f"action '{suggested_action}' is not admissible. Instead, executing action: None"]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]

            # time.sleep(1)
            if self.success:
                return f"SUCCESS"
            elif self.num_actions >= self.max_actions:
                return f"FAILURE"
            else:
                return f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"

        # Define record_memory function
        def record_long_term_memory(guidance: str) -> str:
            # the maximum number of lines are 5; if more than 5, delete the oldest one.
            if (guidance == ''):
                return "no novel knowledge recorded"

            with open(self.log_paths['guidance_path'], "a+") as f:
                f.write(f"{guidance}\n")
                lines = f.readlines()
                if len(lines) > 15:
                    f.seek(0)
                    f.truncate()
                    for line in lines[:-1]:
                        f.write(line)

            # time.sleep(1)
            return "knowledge recorded."

        # Define retrieve_memory function, return all the content in the memory.txt file
        def retrieve_long_term_memory() -> str:
            memory_information = ""

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

        def update_and_retrieve_working_memory(new_info: str) -> str:
            self.narrative_state += f"Step {self.num_actions}: " + new_info + "\n"
            return self.narrative_state

        register_function_lambda(
            {r"execute_action": execute_action},
            [self.external_perception_agent]
        )

        register_function_lambda(
            {r"record_long_term_memory": record_long_term_memory},
            [self.internal_perception_agent]
        )

        register_function_lambda(
            {r"update_and_retrieve_working_memory": update_and_retrieve_working_memory},
            [self.short_term_memory_summarizer_agent]
        )

        register_function_lambda(
            {r"retrieve_long_term_memory": retrieve_long_term_memory},
            [self.long_term_memory_summarizer_agent]
        )

    def initialize_groupchat(self, max_chat_round=200):

        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.motor_agent,
                self.global_workspace_agent,
                self.external_perception_agent,
                self.internal_perception_agent,
                self.conscious_agent,
                self.update_and_retrieve_working_memory_agent,
                self.retrieve_long_term_memory_agent,
                self.learning_agent,
                self.record_long_term_memory_agent,
                self.short_term_memory_summarizer_agent,
                self.long_term_memory_summarizer_agent
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
