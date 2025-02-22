import copy
import os

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent

class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, memory_path, game_no, max_actions=50, args=None):
        super().__init__(env, obs, info, llm_config, log_path, memory_path, game_no, max_actions, args)

        self.allowed_transitions = None
        self.planning_agent = None
        self.motor_agent = None
        self.imagination_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent_1 = None
        self.internal_perception_agent_2 = None
        self.awareness_agent = None
        self.update_and_retrieve_episodic_memory_agent = None
        self.retrieve_long_term_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.short_term_memory_summarizer_agent = None
        self.long_term_memory_summarizer_agent = None
        self.focus_agent = None
        self.task = ''

        self.game_no = game_no

        self.episodic_memory = ''
        self.initialize_autogen()

    def initialize_agents(self):

        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message=(
                "You call the focus function with no arguments. "
                "Your output should always be: focus()"
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=(
                '''Your job is to optimally solve the current task by formulating and executing an action plan. 
                You must execute your plan by evaluating all currently admissible actions and proposing one of them. "
                You will receive feedback, ideas and information to help you improve your plan.
                Output Format = 
                    PLAN: [current step-by-step plan]
                    ACTION: [proposed action]
                \nExample 1: 
                    Information = [You are in the middle of a room. Looking quickly around you, you see a bed 1,
                     a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. 
                    Your task is to: look at a book under the desklamp.]
                    Your Output =
                        PLAN: [I will try to find a book. Then, I will look for the desklamp. Finally, I will look at the book under the desklamp.]
                        ACTION: [go to desk 1]
                \nExample 2 (After you've found the desklamp at desk 1, then went to desk 2.): 
                    Feedback = [on the desk 2, you see a book 1, and a cd 3] 
                    Your Output =
                        PLAN: [I will try to look at book 1 on desk 2 under the desklamp at desk 1]
                        ACTION: [take book 1]'''
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )
        #REASON: [A bowl is more likely to appear in desk(1-2), drawer (1-2), shelf (1-3)]
        #REASON: [Now that I've found a book, I need to take it.]
        def motor_agent_termination_msg(msg):
            return msg["name"] == "Motor_Agent" and msg["content"] is not None and msg["content"][:5] != "ECHO:"

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message='''You call the execute_action function with the given action as the argument.
                           If no action is given, then your output = execute_action(\'\')       
                           \nExample: 
                                If the given action = ACTION [go to desk 1] 
                                Then your output = execute_action(\'go to desk 1\')''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.retrieve_long_term_memory_agent = ConversableAgent(
            name="Retrieve_Long_Term_Memory_Agent",
            system_message="You always call the retrieve_long_term_memory function with no arguments. Your output should "
                           "always be: retrieve_long_term_memory()",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""Your sole task is to call the record_long_term_memory function with the given rule as the argument. 
                    If no rule is given, then your output = record_long_term_memory(\'NO NEW RULES at this time.\')
                    \nExample:
                        If the given rule = Rule Discovered: [You must examine an object before attempting to interact with it.]
                        Then your output = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')
                    """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        self.imagination_agent = ConversableAgent(
            name="Imagination_Agent",
            system_message=(
                "You retrieve and integrate all relevant information, including information stored in memory, to construct new ideas, theories, and hypotheses."
                "You must construct these new ideas through creativity and reasoning, and by building-on previous ideas or even random ideas."
                "\nExample Output (After trying to open a drawer multiple times while holding a spoon):"
                "I noticed we were holding spoon 1 when we tried to open the drawer. Maybe the reason we couldn't open the drawer is because our hands are full. We need to place the spoon 1 somewhere before attempting to open the drawer again."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.awareness_agent = ConversableAgent(
            name="Awareness_Agent",
            system_message=(
                "You integrate all available information from the ongoing conversation and maintain a continuously updated, first-person narrative model of your environment and your actions within it. This narrative should:"
                "1. Include all known details of the environment, along with your own state and any items you have encountered."
                "2. Accurately reflect events that have transpired so far, updating and correcting as new information arrives."
                "3. Strive for maximum accuracy. When details are uncertain or missing, infer plausible elements only as a last resort, ensuring consistency and usefulness in the model."
                "\nIf you discover an error in your previous understanding, revise the model immediately to incorporate the correct information."
                "You will create this narrative one model update at a time."
                "Your narrative model updates should NOT include any reasoning or suggestions on how to solve the task, they must simply describe your environment and your past actions within it."
                "\nYour output must always strictly follow this pattern:"
                "Model Update: [First-person narrative integrating environment, tasks, discoveries, attempts, successes, failures, hypotheses, and current decision-making state]"
                "\nExample 1:"
                "Model Update: I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and "
                "attempted to put it into drawer 1, but I was unable to open that drawer. Then, I realized I couldn't open the drawer because my hands were full. Then, I placed spoon 1 on countertop 1. Then, I opened drawer 1."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.update_and_retrieve_episodic_memory_agent = ConversableAgent(
            name="Update_And_Retrieve_Episodic_Memory_Agent",
            system_message='''You call the update_and_retrieve_episodic_memory function with the given model update as the argument. 
                           If no model update is given, then your output = update_and_retrieve_episodic_memory(\'\')
                           \nExample: 
                                If the given model update = Model Update: [I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.]
                                Then your output = update_and_retrieve_episodic_memory(\'I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.\')''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        #self.learning_agent = ConversableAgent(
        #    name="Learning_Agent",
        #    system_message='''You execute the update_and_retrieve_episodic_memory function and analyze the resulting output in order to discover generalizable knowledge including truths, rules, and patterns that help the other agents operate more rationally.
        #                      Your output should always be a call to the record_long_term_memory function.
        #                      If no new knowledge is identified, then your output = record_long_term_memory(\'NO NEW RULES at this time.\')
        #                      \nExample:
        #                        After executing the update_and_retrieve_episodic_memory function you noticed that you attempted to carry two objects simultaneously but failed. However, after carrying one object at a time, you succeeded.
        #                        Your output = record_long_term_memory(\'The agent cannot carry more than one object at a time.\')''',
        #    llm_config=self.llm_config,
        #    human_input_mode="NEVER",
        #    is_termination_msg=lambda msg: False,
        #)

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        self.external_perception_agent = get_echo_agent("External_Perception_Agent", llm_config,
                                                        additional_termination_criteria=[motor_agent_termination_msg])
        self.internal_perception_agent_1 = get_echo_agent("Internal_Perception_Agent_1", llm_config)
        self.internal_perception_agent_2 = get_echo_agent('Internal_Perception_Agent_2', llm_config)

        self.short_term_memory_summarizer_agent = ConversableAgent(
            name="Short_Term_Memory_Summarizer_Agent",
            system_message="You execute the update_and_retrieve_episodic_memory function and then summarize the crucial information for solving the task that is within the resulting output.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.long_term_memory_summarizer_agent = ConversableAgent(
            name="Long_Term_Memory_Summarizer_Agent",
            system_message="You execute the retrieve_long_term_memory function and then summarize the crucial information for solving the task that is within the resulting output.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message='''You analyze the resulting output in order to discover generalizable knowledge including truths, rules, and patterns that help the other agents operate more rationally.
                                      If no new rule is identified, explicitly state: "NO NEW RULES at this time."
                                      \nExample: 
                                        If within the resulting output you attempted to carry two objects simultaneously and failed, but after carrying one object at a time, you succeeded.
                                        Then your Output = Rule Discovered: [I cannot carry more than one object at a time.]''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],
            self.motor_agent: [self.external_perception_agent],
            self.external_perception_agent: [self.awareness_agent],
            self.awareness_agent: [self.retrieve_long_term_memory_agent, self.focus_agent, self.update_and_retrieve_episodic_memory_agent, self.planning_agent],
            self.update_and_retrieve_episodic_memory_agent: [self.short_term_memory_summarizer_agent, self.learning_agent],
            self.long_term_memory_summarizer_agent: [self.imagination_agent],
            self.short_term_memory_summarizer_agent: [self.imagination_agent],
            self.retrieve_long_term_memory_agent: [self.long_term_memory_summarizer_agent],
            self.imagination_agent: [self.planning_agent, self.learning_agent, self.retrieve_long_term_memory_agent, self.imagination_agent],
            self.learning_agent: [self.record_long_term_memory_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.imagination_agent],
            self.internal_perception_agent_2: [self.awareness_agent],
            self.focus_agent: [self.internal_perception_agent_2]
        }

        self.motor_agent.description = "calls the execute_action function with the proposed action as the argument"
        self.external_perception_agent.description = "executes the given execute_action function call and then parrots the resulting output as feedback."
        self.awareness_agent.description = "integrates all available information from the ongoing conversation and maintains a continuously updated, first-person narrative model of the environment and past actions within it"
        self.planning_agent.description = "makes final action decisions to solve the current task"

        self.update_and_retrieve_episodic_memory_agent.description = "helps the other agents process and recall useful information for solving the current task"
        self.short_term_memory_summarizer_agent.description = "executes the update_and_retrieve_episodic_memory function and then summarizes the crucial information for solving the task that is within the resulting output"

        self.imagination_agent.description = "retrieves and integrates all available information from the ongoing conversation and from long-term memory in order to construct new ideas from previous ideas"

        self.learning_agent.description = "helps the other agents learn generalizable knowledge in order to solve the current task and future tasks"
        self.record_long_term_memory_agent.description = "calls the record_long_term_memory function with the given rule as the argument"
        self.retrieve_long_term_memory_agent.description = "helps the other agents recall useful knowledge for solving the current task"
        self.long_term_memory_summarizer_agent.description = "executes the retrieve_long_term_memory function and then summarizes the crucial information for solving the task that is within the resulting output"
        self.internal_perception_agent_1.description = "executes the record_long_term_memory function and then parrots the resulting output"

        self.focus_agent.description = "helps the other agents focus on solving the task"
        self.internal_perception_agent_2.description = "executes the focus function and then parrots the resulting output"

        self.start_agent = self.external_perception_agent

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.98:
                self.obs = [f"action '{suggested_action}' is not admissible. Instead, executing action: None"]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]


            self.episodic_memory += f"Step {self.num_actions}: " + self.obs[0] + "\n\n"

            # time.sleep(1)
            if self.success:
                return f"SUCCESS"
            elif self.num_actions >= self.max_actions:
                return f"FAILURE"
            else:
                return f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"

        # Define record_memory function
        def record_long_term_memory(rule: str) -> str:
            if rule == "NO NEW RULES at this time.":
                return "NO NEW RULES at this time."

            with open(self.log_paths['rule_path'], 'a+') as f:
                f.write(f"- {rule}\n")

            with open(self.log_paths['memory_path'], 'a+') as f:
                f.write(f"- {rule}\n")

            return "Rule Recorded"

        # Define retrieve_memory function, return all the content in the memory.txt file
        def retrieve_long_term_memory() -> str:
            memory_information = ""
            #previous_rules = []

            if os.path.exists(self.log_paths['memory_path']):
                with open(self.log_paths['memory_path'], "r") as f:
                    memory_information = f.read()

            #if os.path.exists(self.log_paths['rule_path']):
            #    memory_information += "\nRules: "
            #    with open(self.log_paths['rule_path'], "r") as f:
            #        memory_information += f.read()

            #if len(self.log_paths['previous_rule_path']) > 0:
            #    memory_information += "\nPrevious Rules: \n"
            #    for previous_rule_path in self.log_paths['previous_rule_path']:
            #        if os.path.exists(previous_rule_path):
            #            with open(previous_rule_path, "r") as f:
            #                previous_rules.append(f.read())

            #memory_information += "\n".join(previous_rules)
            print("long_term_memory:\n", memory_information)
            return memory_information

        def update_and_retrieve_episodic_memory(new_info: str) -> str:
            self.episodic_memory += f"Step {self.num_actions}: " + new_info + "\n\n"
            return self.episodic_memory

        def focus() -> str:
            return f"{self.task}\nLast Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"

        register_function_lambda(
            {r"execute_action": execute_action},
            [self.external_perception_agent]
        )
        #short-term -> long-term
        #
        register_function_lambda(
            {r"record_long_term_memory": record_long_term_memory},
            [self.internal_perception_agent_1]
        )

        register_function_lambda(
            {r"update_and_retrieve_episodic_memory": update_and_retrieve_episodic_memory},
            [self.short_term_memory_summarizer_agent, self.learning_agent]
        )

        register_function_lambda(
            {r"retrieve_long_term_memory": retrieve_long_term_memory},
            [self.long_term_memory_summarizer_agent]
        )

        register_function_lambda(
                {r"focus": focus},[self.internal_perception_agent_2]
        )

    def initialize_groupchat(self, max_chat_round=200):

        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.motor_agent,
                self.imagination_agent,
                self.external_perception_agent,
                self.internal_perception_agent_1,
                self.internal_perception_agent_2,
                self.awareness_agent,
                self.update_and_retrieve_episodic_memory_agent,
                self.retrieve_long_term_memory_agent,
                self.learning_agent,
                self.record_long_term_memory_agent,
                self.short_term_memory_summarizer_agent,
                self.long_term_memory_summarizer_agent,
                self.focus_agent,
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
