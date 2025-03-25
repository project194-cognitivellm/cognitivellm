import copy
import os

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np


class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, memory_path1, memory_path2, game_no, max_actions=30,
                 rounds_per_game=1, args=None):
        super().__init__(env, obs, info, llm_config, log_path, memory_path1, memory_path2, game_no, max_actions, args)

        self.planning_agent = None
        self.motor_agent = None
        self.idea_agent = None
        self.external_perception_agent = None
        self.internal_perception_agent_1 = None
        self.internal_perception_agent_2 = None
        self.conscious_agent = None
        self.retrieve_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.memory_summarizer_agent = None
        self.focus_agent = None

        self.k = 0
        self.allowed_transitions = None
        self.task = obs[0].split("Your task is to: ")[1]
        self.game_no = game_no
        self.initialize_autogen()
        self.get_summary_rules()
        self.task_failed = False
        self.task_success = False
        self.rounds = rounds_per_game
        self.rounds_left = self.rounds
        self.max_round_actions = self.max_actions // self.rounds
        self.max_actions = self.max_actions - self.max_round_actions * (self.rounds - 1)
        self.percept = f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
        self.episodic_memory = f"Time {self.num_actions_taken}: " + "You and all other Agents are collectively a singular conscious entity named ALFRED. " + \
                               self.obs[0] + "\n"

    def initialize_agents(self):

        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message='''You must call the 'focus' function with no arguments.
            IMPORTANT: It is necessary that you formulate and output at least one call to the 'focus' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message='''You must call the 'retrieve_memory' function with no arguments. Your output must always without exception = retrieve_memory()
                    IMPORTANT: It is necessary that you formulate and output a call to the 'retrieve_memory' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=f'''You must optimally solve the current task ({self.task}) by formulating and executing an action plan. You must do so by:
                1. Predicting an optimal action plan by analyzing history and gathering and integrating relevant information.
                2. After formulating an action plan, you must execute your plan by evaluating the last currently admissible actions and proposing one of them.
                3. You will receive feedback, ideas and partial information to help you improve your plan. You must integrate these ideas and information into your plan.
                4. You must ensure that your plan is optimal, balancing exploration and exploitation.
            EXCEPTION: However, if you are having trouble formulating a plan and proposing an action, then, only as a last resort, you may propose ACTION: [do nothing].

            Your strict output format = ACTION: [proposed action]

            Example 1 (Context: Information = [You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. Your task is to: look at a book under the desklamp.]):
                Your suggested output = ACTION: [go to desk 1]

            Example 2 (Context: After example 1, you've found desklamp 1 at desk 1, then went to desk 2. Information = [on the desk 2, you see a book 1, and a cd 3]):  
                Your suggested output = ACTION: [take book 1 from desk 2]

            Example 3 (Context: If you are having trouble proposing an action):
                Your output can = ACTION: [do nothing]''',
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message='''You must call the 'execute_action' function with the proposed action provided from 'Planning_Agent' as the argument.
            EXCEPTION: However, if no suitable action is provided, then you must call the 'execute_action' function with an empty string as the argument.

            Example 1 (Context: If the provided action = ACTION [go to desk 1]):
                Your output must = execute_action(\'go to desk 1\')

            Example 2 (Context: If no suitable action is provided):
                Your output must = execute_action(\'\')''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        self.idea_agent = ConversableAgent(
            name="Idea_Agent",
            system_message='''You must integrate all available information to construct new ideas, such as strategies, theories, and hypotheses, to help with task progression.
            You must construct these new ideas through spontaneous creativity, taking a deep breath, and reasoning step-by-step in a chain of thought.

            Your suggested output format = [idea type]: [idea contents and reasoning behind idea]

            Example 1 (Context: After Planning_Agent and Motor_Agent have failed to open a drawer multiple times while holding a spoon):
                Your suggested output = HYPOTHESIS: I noticed you were holding spoon 1 when you tried to open the drawer, maybe the reason you couldn't open the drawer is because your hands are full? You should try to place the spoon 1 somewhere before attempting to open the drawer again.''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message='''You must integrate all possible information throughout time to formulate a world model update. Your world model update must be a first-person narrative model of your environment and your actions within it. 
            Your world model update must:
                1. Include all known details of the environment, along with your own state and any items you have encountered.
                2. Accurately reflect events that have transpired so far, updating and correcting as new information arrives.
                3. Strive for maximum accuracy. When details are uncertain or missing, infer plausible elements only as a last resort, ensuring consistency and usefulness in the model.
            EXCEPTION: However, if you are having trouble formulating a world model update, then as a last resort your world model update may be Model Update: [I am confused. I need time to process and recall useful information.]

            Your strict output format = Model Update: [First-person narrative integrating environment, tasks, discoveries, attempts, successes, failures, hypotheses, and current decision-making state]

            IMPORTANT: It is necessary that you formulate and output a world model update under all circumstances; You are not allowed not to respond. Therefore, do whatever is necessary to ensure you respond.

            Example 1 (Context: Simple example of a world model update):
                Your output can = Model Update: [I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. Then, I realized I couldn't open the drawer because my hands were full. Then, I placed spoon 1 on countertop 1. Then, I opened drawer 1.]

            Example 2 (Context: If you are unable to formulate a world model update):
                Your output can = Model Update [I am confused. I need to focus!]''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.external_perception_agent = ConversableAgent(
            name="External_Perception_Agent",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.internal_perception_agent_1 = ConversableAgent(
            name="Internal_Perception_Agent_1",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.internal_perception_agent_2 = ConversableAgent(
            name="Internal_Perception_Agent_2",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.memory_summarizer_agent = ConversableAgent(
            name="Memory_Summarizer_Agent",
            system_message="You must execute the 'retrieve_memory' function and then summarize the crucial information for solving the task that is within the resulting output.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message='''You must execute the 'retrieve_memory' function and then analyze the resulting output to formulate generalizable knowledge about reality, such as empirical truths, general rules, and general patterns, that can help operate more rationally within the environment.
            Knowledge discovered must:
                1. Be as general as possible.
                2. NOT reference any task-specific details such as goals, locations, items, or events.
                3. Be novel and NOT similar to knowledge already in Long-Term memory.
                4. Be empirically likely to lead to positive outcomes in the future. 
            EXCEPTION: However, if you cannot identify any new generalizable knowledge, then you must explicitly state: NO NEW KNOWLEDGE at this time. 

            Your strict output format = Knowledge Discovered: [knowledge contents]

            Example 1 (Context: If you attempted to carry two objects simultaneously and failed, but after carrying one object at a time, you succeeded.):
                Your suggested output = Knowledge Discovered: [I cannot carry more than one object at a time.]

            Example 2 (Context: If you cannot identify any new generalizable knowledge.):
                Your output must = Knowledge Discovered: [NO NEW KNOWLEDGE at this time]''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You must call the 'record_long_term_memory' function with the provided knowledge from 'Learning_Agent' as the argument. 
            EXCEPTION: However, if no suitable knowledge is provided, then you must call the 'record_long_term_memory' function with \'NO NEW KNOWLEDGE at this time.\' as the argument.

            Example 1 (Context: If the provided knowledge = Knowledge Discovered: [You must examine an object before attempting to interact with it.]):
                Your output must = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')

            Example 2 (Context: If the provided knowledge = Knowledge Discovered: [NO NEW KNOWLEDGE at this time.]):
                Your output must = record_long_term_memory(\'NO NEW KNOWLEDGE at this time.\')""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],
            self.motor_agent: [self.external_perception_agent],
            self.external_perception_agent: [self.conscious_agent],
            self.conscious_agent: [self.retrieve_memory_agent, self.planning_agent, self.focus_agent],
            self.retrieve_memory_agent: [self.memory_summarizer_agent, self.learning_agent],
            self.memory_summarizer_agent: [self.idea_agent],
            self.idea_agent: [self.planning_agent],
            self.learning_agent: [self.record_long_term_memory_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.retrieve_memory_agent],
            self.internal_perception_agent_2: [self.conscious_agent],
            self.focus_agent: [self.internal_perception_agent_2]
        }

        self.motor_agent.description = "calls the 'execute_action' function with the proposed action given by 'Planning_Agent' as the argument"
        self.external_perception_agent.description = "executes the proposed 'execute_action' function call given by 'Motor_Agent' and then parrots the resulting output as feedback."
        self.conscious_agent.description = "integrates all available information and maintains a continuously updated, first-person narrative model of the environment and past actions within it"
        self.planning_agent.description = "makes final action decisions to solve the current task"

        self.retrieve_memory_agent.description = "calls the 'retrieve_memory' function to help process and recall useful information in order to solve the current task more effectively"
        self.memory_summarizer_agent.description = "executes the 'retrieve_memory' function and then summarizes the crucial information for solving the task that is within the resulting output"

        self.idea_agent.description = "integrates all available information from the ongoing conversation in order to construct new ideas"

        self.learning_agent.description = "executes the 'retrieve_memory' function and then formulates generalizable knowledge that is within the resulting output"
        self.record_long_term_memory_agent.description = "calls the 'record_long_term_memory' function with the knowledge given by 'Learning_Agent' as the argument"
        self.internal_perception_agent_1.description = "executes the 'record_long_term_memory' function and then parrots the resulting output"

        self.focus_agent.description = "calls the 'focus' function to help focus on solving the task"
        self.internal_perception_agent_2.description = "executes the 'focus' function and then parrots the resulting output"

        self.start_agent = self.external_perception_agent

    def register_functions(self):

        def execute_action(suggested_action: str) -> str:
            if not suggested_action or suggested_action == "do nothing":
                return f"NO ACTION GIVEN. YOU NEED TO FOCUS ON THE FOLLOWING:\nTask: {self.task}\nLast {self.percept}"

            if self.task_failed and self.rounds_left == 0:
                return "FLEECE"

            if self.task_failed:
                self.max_actions += self.max_round_actions
                self.task_failed = False
                self.percept = f"YOU GET ONE MORE CHANCE! DON'T GIVE UP!\nObservation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
                return self.percept

            if self.task_success:
                return "STRAWBERRY"

            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.98:
                self.obs = [
                    f"You attempted to perform the action '{suggested_action}', but it is either not possible at this time or not in the list of admissible actions verbatim."]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]
            self.num_actions_taken += 1

            self.episodic_memory += f"Time {self.num_actions_taken}: " + self.obs[0] + "\n"

            if self.success:
                self.task_success = True
                self.percept = f"Observation: {self.obs[0]}\nTask Status: COMPLETED\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
                self.percept += f"\nTask Completed. Reflect on your actions and reasoning. Try to figure out what went right and what good decisions were made that lead to success, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Planning_Agent suggest any action and have Motor_Agent call the execute_action function, for example ACTION: [end chat]."
            elif self.num_actions_taken >= self.max_actions:
                self.task_failed = True
                self.rounds_left -= 1
                self.percept = f"Observation: {self.obs[0]}\nTask Status: FAILED\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
                self.percept += f"\nTask Failed. Reflect on your actions and reasoning. Try to figure out what went wrong and what mistakes were made that lead to failure, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Planning_Agent suggest any action and have Motor_Agent call the execute_action function, for example ACTION: [end chat]."
            else:
                self.percept = f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
            return self.percept

        def record_long_term_memory(knowledge: str) -> str:
            if knowledge == "NO NEW KNOWLEDGE at this time." or len(knowledge) < 45:
                return "Model Update: [I attempted to learn something, but I couldn't formulate any new knowledge.]"

            with open(self.log_paths['rule_path'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            with open(self.log_paths['memory_path1'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            self.episodic_memory += f"Time {self.num_actions_taken}: You learned that " + knowledge + "\n"
            return f'Model Update: [I learned that {knowledge}.]'

        def retrieve_memory() -> str:
            long_term_memory = ""
            if os.path.exists(self.log_paths['memory_path2']):
                with open(self.log_paths['memory_path2'], "r") as f:
                    long_term_memory = f.read()

            return f"EPISODIC MEMORY:\n{self.episodic_memory}\n\nWORKING LONG-TERM MEMORY:\n{long_term_memory}"

        def focus() -> str:
            return f"YOU NEED TO FOCUS ON THE FOLLOWING: \nTask: {self.task}\nLast {self.percept}"

        register_function(
            execute_action,
            caller=self.motor_agent,
            executor=self.external_perception_agent,
            description="Executes actions in environment"
        )

        register_function(
            focus,
            caller=self.focus_agent,
            executor=self.internal_perception_agent_2,
            description="Helps agents focus on the current task."
        )

        register_function(
            record_long_term_memory,
            caller=self.record_long_term_memory_agent,
            executor=self.internal_perception_agent_1,
            description="Records new knowledge in long-term memory."
        )

        register_function_lambda(
            {r"retrieve_memory": retrieve_memory},
            [self.memory_summarizer_agent, self.learning_agent]
        )

    def initialize_groupchat(self, max_chat_round=500):

        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.motor_agent,
                self.idea_agent,
                self.external_perception_agent,
                self.internal_perception_agent_1,
                self.internal_perception_agent_2,
                self.conscious_agent,
                self.retrieve_memory_agent,
                self.learning_agent,
                self.record_long_term_memory_agent,
                self.memory_summarizer_agent,
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

    def get_summary_rules(self, model_name='all-MiniLM-L6-v2'):
        """
        Get representative rules from a list of rules using clustering.

        Args:
            model_name (str): Name of the sentence transformer model to use

        Returns:
            dict: Dictionary containing:
                - 'representative_rules': List of k representative rules
                - 'cluster_sizes': Dictionary of cluster sizes
                - 'cluster_members': Dictionary of rules in each cluster
        """
        rule_text = ''
        if os.path.exists(self.log_paths['memory_path1']):
            with open(self.log_paths['memory_path1'], "r") as file:
                rule_text = file.read()

        rule_lines = [line.strip() for line in rule_text.split('\n') if line.strip()]
        num_rules = len(rule_lines)

        if num_rules == 0:
            return {'representative_rules': [], 'cluster_sizes': {}, 'cluster_members': {}}

        # Initialize model and compute embeddings
        sentence_transformer_model = SentenceTransformer(model_name)
        rule_embeddings = sentence_transformer_model.encode(rule_lines, convert_to_tensor=True).cpu().numpy()

        # Determine number of clusters
        if num_rules <= 10:
            self.k = num_rules
        elif num_rules <= 100:
            self.k = 10
        else:
            self.k = int(np.sqrt(num_rules))

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rule_embeddings)

        # Find closest points to each center
        representative_rules = []
        cluster_members = {i: [] for i in range(self.k)}
        for center in kmeans.cluster_centers_:
            # Calculate distances from this center to all points
            distances = np.linalg.norm(rule_embeddings - center, axis=1)
            # Get index of closest point
            closest_idx = np.argmin(distances)
            # Store the original text of the closest point
            representative_rules.append(rule_lines[closest_idx])

        # Get cluster sizes and members
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}

        # Organize rules by cluster
        for i, label in enumerate(labels):
            cluster_members[label].append(rule_lines[i])

        if os.path.exists(self.log_paths['memory_path2']):
            with open(self.log_paths['memory_path2'], "w") as file:
                for i, rule in enumerate(representative_rules):
                    file.write(f'{i}: ' + rule + '\n')
                    # file.write(rule + '\n')

        return {
            'representative_rules': representative_rules,
            'cluster_sizes': cluster_sizes,
            'cluster_members': cluster_members
        }
