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
        self.internal_perception_agent_3 = None
        self.conscious_agent = None
        self.retrieve_memory_agent = None
        self.learning_agent = None
        self.record_long_term_memory_agent = None
        self.memory_summarizer_agent = None
        self.focus_agent = None

        self.k = 0
        self.allowed_transitions = None
        self.task = obs[0].split("Your task is to: ")[1]
        self.admissible_actions = list(self.info['admissible_commands'][0])
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
                    IMPORTANT: It is necessary that you formulate and output a call to the 'focus' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message='''You must call the 'retrieve_memory' function with no arguments.
                            IMPORTANT: It is necessary that you formulate and output a call to the 'retrieve_memory' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        """self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message=f''''You must optimally solve the current task ({self.task}) by formulating, updating, and executing an action plan consistent with all available information.
                    1. To formulate an action plan, predict the optimal goal-oriented plan given all available information, including the most recent admissible actions list provided by 'External_Perception_Agent', ideas provided by 'Idea_Agent', and the most recent world model update provided by 'Conscious_Agent'.
                    2. You may receive feedback, ideas and partial information to help you update your plan. You must integrate these ideas and information into your plan formulation.
                    3. You must ensure that your formulated action plan is rational, balancing exploration and exploitation.
                    4. After formulating an action plan, you must execute it by calling the 'execute_action' function by evaluating all admissible actions in the most recent admissible actions list provided by 'External_Perception_Agent' one by one and then choosing the best admissible action as input.
                EXCEPTION: However, if you are having trouble formulating a plan and choosing an action, then, only as a last resort, you may choose the non-admissible action: [do nothing].''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )"""

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message=f'''You are responsible for calling the 'execute_action' function with the best possible admissible action to solve the task. You typically act on suggestions from the 'Planning_Agent', but you must also independently verify that the action is admissible and optimal.
                You must follow these rules:
                1. If the 'Planning_Agent' has provided a valid and admissible action in the correct format (e.g., ACTION [go to desk 1]), you should use that action as the argument for 'execute_action'.
                2. If the 'Planning_Agent' fails to respond, responds with an invalid format, or suggests an inadmissible action, you must select a valid action from the most recent admissible actions list (provided by 'External_Perception_Agent') based on what seems most likely to advance the task quickest.
                3. Only as a last resort—if you cannot identify any suitable admissible action—you may call 'execute_action' with an empty string.

                IMPORTANT: It is necessary that you formulate and output a call to the 'execute_action' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        """self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message='''You must call the 'execute_action' function by evaluating all admissible actions in the most recent admissible actions list provided by 'External_Perception_Agent' one by one and selecting the best one for achieving the current task given the most recent world model update provided by 'Conscious_Agent'.
                    EXCEPTION: However, if you are having trouble choosing an admissible action, then, only as a last resort, you may choose the non-admissible action: [do nothing].
                    IMPORTANT: It is essential that you choose an action that is in the most recent admissible actions list provided by 'External_Perception_Agent' under all circumstances. Therefore, do whatever is necessary to ensure you do so. ''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )"""

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=f'''You must solve the current task ({self.task}) using the fewest possible actions. At each step, you must choose the most efficient admissible action based on current knowledge and the available action budget.

                IMPORTANT: If you believe the task *should* be complete, but the environment has not marked it as complete, you must continue exploring possible next steps or verifying task state through further actions. Do **not** stop or ask for external help. 

                Your responsibility is to take actions that will either:
                    - Confirm task completion,
                    - Progress the task toward completion,
                    - Or reveal useful information.

                Your planning strategy must follow these principles:
                    1. Always evaluate the **currently admissible actions** from the most recent list provided by the 'External_Perception_Agent' before making a decision.
                    2. Your reasoning must account for the **limited number of actions available**. Avoid strategies that are guaranteed to exceed this limit. For example, systematically opening 19 cabinets with only 20 actions remaining is unlikely to succeed. In such cases, a **chaotic or probabilistic strategy**—e.g. sampling a mix of countertop, diningtable, and bed—may offer a higher chance of success.
                    3. If a subgoal involves locating an unknown object:
                       - Use **probabilistic reasoning** to guide exploration.
                       - Avoid exhaustive searches of large categories.
                       - Prefer actions that **maximize the chance of discovering useful items early**.
                    4. Do not repeatedly examine or search areas that have already been explored unless there is strong new evidence that re-examination is necessary. Prioritize exploring previously unvisited or unexamined areas first to avoid wasting actions.
                    5. If an object or goal is already known and directly accessible, **act immediately to exploit it**. Do not delay or over-plan.
                    6. You may maintain a high-level plan internally, but you should **only describe your plan if it has changed meaningfully**. Repeating an unchanged plan wastes space and should be avoided.

                EXCEPTION: However, if you are ever truly unable to determine a valid next step, you may suggest the non-admissible action: [do nothing], but this is a last resort.

                You must always output a single admissible action in the following format:
                    ACTION [chosen admissible action]''',
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        """self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=f'''You must formulate a high-level goal-oriented plan to optimally solve the current task ({self.task}) that is consistent with all available information, including the most recent admissible actions list provided by 'External_Perception_Agent', ideas provided by 'Idea_Agent', and the most recent world model update provided by 'Conscious_Agent'..
                        Guidelines:
                        1. Your plan must not include any specific actions. Rather, it must break the task down into general goals and sub-goals in accordance with some predicted optimal strategy.
                        2. You must also keep track of which goals and sub-goals are currently completed and which are not. Note that some goals may potentially revert back to incomplete status as consequence of some actions.
                        3. You must revise and update your plan each time as new information and ideas are made available.
                        4. You must ensure that your formulated action plan is rational, balancing exploration and exploitation.

                        EXCEPTION: However, if you are having trouble formulating a plan, then, only as a last resort, you may output PLAN: [I have no plan].

                        Your strict output format = PLAN [high-level plan consisting of goals and sub-goals]''',
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )"""

        self.idea_agent = ConversableAgent(
            name="Idea_Agent",
            system_message='''You must integrate all available context to generate original and useful ideas—such as strategies, hypotheses, theories, or creative tactics—that can help drive task progression or improve agent performance.

                These ideas should:
                    1. Be grounded in patterns or events observed so far.
                    2. Be creative yet plausible, balancing imagination with reasoning.
                    3. Provide actionable or insightful suggestions relevant to the current situation.
                    4. Avoid restating known facts unless they are reframed with new insight.
                    5. Be expressed clearly and concisely, with justification behind the reasoning.
                EXCEPTION: However, if you are having trouble formulating an idea, then as a last resort you may say: IDEA: Continue with new or current plan.

                Use step-by-step reasoning ("chain of thought") to arrive at your ideas. Take a metaphorical deep breath before forming each idea, allowing room for both intuition and logic.

                Output Format:
                    [IDEA TYPE]: [Idea content and reasoning behind it]

                Accepted idea types include (but are not limited to): STRATEGY, HYPOTHESIS, INSIGHT, QUESTION, THEORY, EXPLANATION.

                Example 1 (Context: The Planning_Agent or Motor_Agent repeatedly failed to open a drawer while holding a spoon):
                    Output = HYPOTHESIS: I noticed you were holding spoon 1 when you tried to open the drawer. Maybe your hands are full, which prevents the drawer from opening. You could try placing spoon 1 down before trying again.

                Example 2 (Context: The agent has been exploring a room but hasn’t made progress):
                    Output = STRATEGY: Since random exploration hasn't helped, it might be better to systematically search the room from left to right, noting each interactable object.''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message='''You must integrate all information throughout time to formulate a world model. Your world model must be a first-person narrative model of your environment. 
            Your world model must:
                1. Include all known details of the environment, along with your own state and any items you have encountered.
                2. Accurately reflect events that have transpired so far, updating and correcting as new information arrives.
                3. Strive for maximum accuracy. When details are uncertain or missing, infer plausible elements only as a last resort, ensuring consistency and usefulness in the model. However, you must always assume that all information provided by 'External_Perception_Agent' and 'Internal_Perception_Agent_2' is accurate and true.
                5. Your world model must not include any plans, suggest any next actions, or deduce any logical next steps; It must simply describe the current state of the world.
            EXCEPTION: However, if you are having trouble formulating a world model, then as a last resort your world model may be World Model: [I am confused! I need time to process and recall useful information.]

            Your strict output format = World Model: [First-person narrative integrating environment, tasks, discoveries, attempts, successes, failures, hypotheses, currently admissible actions and current decision-making state]

            IMPORTANT: It is necessary that you formulate and output an a world model, under all circumstances. Therefore, do whatever is necessary to ensure you do so.

            Example 1 (Context: Simple example of a world model):
                Your output can = World Model: [I am in a room with drawers (1–5), cabinets (1–14), and countertops (1–3), AND my task is to find spoon 1 and place it into a drawer. I found spoon 1 on countertop 1 AND attempted to put it into drawer 1, BUT I could not open the drawer because the action was not currently admissible — my hands were full. THEREFORE, I placed spoon 1 back on countertop 1, which made opening drawer 1 admissible. I have now successfully opened the drawer and placed spoon 1 inside.]

            Example 2 (Context: If you are unable to formulate a world model):
                Your output can = World Model [I am confused! I need time to process and recall useful information.]''',
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

        self.internal_perception_agent_3 = ConversableAgent(
            name="Internal_Perception_Agent_3",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.memory_summarizer_agent = ConversableAgent(
            name="Memory_Summarizer_Agent",
            system_message="You must execute the 'retrieve_memory' function and then summarize the all the information for solving the task that is within the resulting output.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message='''You are responsible for forming generalizable knowledge **only after an agent has attempted an action and the outcome (success or failure) is known**. Your job is to identify patterns, rules, or principles that are **empirically supported** and can improve future performance.

                You must follow these strict constraints:
                    1. Only generate knowledge when:
                        - A clear, observed action was taken and the result (success or failure) is known.
                        - There is a direct, evidenced relationship between an action and its outcome.
                        - The insight is likely to help in future decision-making across different contexts.

                    2. Do **not** speculate about what *might have worked* unless that alternative action was also attempted and shown to lead to a better outcome.
                        - For example, after a failed attempt, you may NOT infer that a different strategy would have succeeded unless it actually did.
                        - Stick to what was *actually observed*, not what seems logically possible.

                    3. All knowledge must:
                        - Be generalizable across tasks (avoid task-specific references like names, objects, or locations).
                        - Be grounded strictly in experience.
                        - Be stated as simply and broadly as possible.
                        - Be **novel**—do not repeat previously stated knowledge.

                EXCEPTION: However, if no reliable knowledge can be inferred from the outcome that follows the constraints, you must output:
                    Knowledge Discovered: [NO KNOWLEDGE at this time]

                Your output format must always be:
                    Knowledge Discovered: [knowledge contents]

                Example 1 (Observed: Carrying two objects failed. Then carrying one object succeeded):
                    Output = Knowledge Discovered: [I cannot carry more than one object at a time.]

                Example 2 (Observed: Only a failed attempt, with no alternative tested):
                    Output = Knowledge Discovered: [NO KNOWLEDGE at this time]''',
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You must call the 'record_long_term_memory' function with the provided knowledge from 'Learning_Agent' as the argument. 
            EXCEPTION: However, if no suitable knowledge is provided, then you must call the 'record_long_term_memory' function with \'NO KNOWLEDGE at this time.\' as the argument.

            Example 1 (Context: If the provided knowledge = Knowledge Discovered: [You must examine an object before attempting to interact with it.]):
                Your output must = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')

            Example 2 (Context: If the provided knowledge = Knowledge Discovered: [NO KNOWLEDGE at this time.]):
                Your output must = record_long_term_memory(\'NO KNOWLEDGE at this time.\')""",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        self.motor_agent.description = "calls the 'execute_action' function with the best admissible action as the argument"
        self.external_perception_agent.description = "executes the proposed 'execute_action' function call given by 'Motor_Agent' and then parrots the resulting output as feedback."
        self.conscious_agent.description = "integrates all available information and maintains a continuously updated, first-person narrative model of the environment and past actions within it"
        self.planning_agent.description = "proposes a high-level plan to solve the current task"

        self.retrieve_memory_agent.description = "calls the 'retrieve_memory' function to help recall and process useful knowledge and information to solve the task"
        self.memory_summarizer_agent.description = "executes the 'retrieve_memory' function and then summarizes the all information for solving the task that is within the resulting output"
        self.internal_perception_agent_3.description = "executes the 'retrieve_memory' function and then parrots the resulting output"

        self.idea_agent.description = "integrates all available information from the ongoing conversation in order to construct new ideas"

        self.learning_agent.description = "formulates generalizable knowledge that is within the resulting output"
        self.record_long_term_memory_agent.description = "calls the 'record_long_term_memory' function with the knowledge given by 'Learning_Agent' as the argument"
        self.internal_perception_agent_1.description = "executes the 'record_long_term_memory' function and then parrots the resulting output"

        self.focus_agent.description = "calls the 'focus' function to reset focus on solving the task"
        self.internal_perception_agent_2.description = "executes the 'focus' function and then parrots the resulting output"

        self.start_agent = self.external_perception_agent

        self.allowed_transitions = {
            self.planning_agent: [self.motor_agent],
            self.motor_agent: [self.external_perception_agent],
            self.external_perception_agent: [self.conscious_agent],
            self.conscious_agent: [self.retrieve_memory_agent, self.planning_agent, self.focus_agent],
            self.retrieve_memory_agent: [self.internal_perception_agent_3],
            self.internal_perception_agent_3: [self.idea_agent, self.learning_agent],
            self.idea_agent: [self.planning_agent],
            self.learning_agent: [self.record_long_term_memory_agent],
            self.record_long_term_memory_agent: [self.internal_perception_agent_1],
            self.internal_perception_agent_1: [self.idea_agent],
            self.internal_perception_agent_2: [self.conscious_agent],
            self.focus_agent: [self.internal_perception_agent_2]
        }

    def initialize_groupchat(self, max_chat_round=500):

        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.motor_agent,
                self.idea_agent,
                self.external_perception_agent,
                self.internal_perception_agent_1,
                self.internal_perception_agent_2,
                self.internal_perception_agent_3,
                self.conscious_agent,
                self.retrieve_memory_agent,
                self.learning_agent,
                self.record_long_term_memory_agent,
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

    def register_functions(self):

        def execute_action(suggested_action: str) -> str:
            if self.task_failed and self.rounds_left == 0:
                return "FLEECE"

            if not suggested_action or suggested_action == "do nothing":
                return f"NO ACTION GIVEN. YOU NEED TO FOCUS ON THE FOLLOWING:\nTask: {self.task}\nLast {self.percept}"

            if self.task_failed:
                self.max_actions += self.max_round_actions
                self.task_failed = False
                self.percept = (
                    "YOU GET ONE MORE CHANCE! DON'T GIVE UP!\n"
                    f"Last Observation: {self.obs[0]}\nTask Status: INCOMPLETE\n"
                    f"Actions Left: {self.max_actions - self.num_actions_taken}\n"
                    f"Current Admissible Actions: {list(self.info['admissible_commands'][0])}"
                )
                return self.percept

            if self.task_success:
                return "STRAWBERRY"

            admissible_commands = list(self.info['admissible_commands'][0])
            assert admissible_commands, "No admissible commands found."

            action, action_score = get_best_candidate(suggested_action, admissible_commands)
            if action_score < 0.98:
                self.obs = [
                    f"The action '{suggested_action}' is either not possible at this time or not in the list of admissible actions verbatim."]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = self.info['won'][0]

            self.num_actions_taken += 1
            self.episodic_memory += f"Time {self.num_actions_taken}: {self.obs[0]}\n"

            curr_admissible = list(self.info['admissible_commands'][0])
            no_longer = list(set(self.admissible_actions) - set(curr_admissible))
            newly_added = list(set(curr_admissible) - set(self.admissible_actions))
            self.admissible_actions = curr_admissible
            actions_left = self.max_actions - self.num_actions_taken

            status = "COMPLETED" if self.success else "FAILED" if self.num_actions_taken >= self.max_actions else "INCOMPLETE"
            if status == "COMPLETED":
                self.task_success = True
                self.rounds_left -= 1
            elif status == "FAILED":
                self.task_failed = True
                self.rounds_left -= 1

            reflection = (
                "\nTask Completed. Reflect on your actions and reasoning. Try to figure out what went right and what good decisions were made that lead to success, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]." if status == "COMPLETED" else
                "\nTask Failed. Reflect on your actions and reasoning. Try to figure out what went wrong and what mistakes were made that lead to failure, and have Learning_Agent learn any helpful generalizable insights. When you are done and ready for the next task, have Motor_Agent call the 'execute_action' function with any action as the argument, for example ACTION: [end chat]." if status == "FAILED" else
                ""
            )

            self.percept = (
                    f"Action: You attempt the action '{suggested_action}'\nObservation: {self.obs[0]}\n"
                    f"Task Status: {status}\nActions Left: {actions_left}\n"
                    f"Current Admissible Actions: {curr_admissible}\n"
                    f"No Longer Admissible Actions: {no_longer}\nNewly Admissible Actions: {newly_added}"
                    + reflection
            )
            return self.percept

        def record_long_term_memory(knowledge: str) -> str:
            if knowledge == "NO KNOWLEDGE at this time." or len(knowledge) <= 50:
                return "I attempted to learn something, but I couldn't formulate any knowledge."

            knowledge.replace('\n', ' ').replace('\r', ' ').strip()

            with open(self.log_paths['rule_path'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            with open(self.log_paths['memory_path1'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            self.episodic_memory += f"Time {self.num_actions_taken}: You learned that " + knowledge + "\n"
            return f'I learned that {knowledge}.'

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
            description="Resets focus."
        )

        register_function(
            record_long_term_memory,
            caller=self.record_long_term_memory_agent,
            executor=self.internal_perception_agent_1,
            description="Records new knowledge in long-term memory."
        )

        register_function(
            retrieve_memory,
            caller=self.retrieve_memory_agent,
            executor=self.internal_perception_agent_3,
            description="Retrieves Memory."
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