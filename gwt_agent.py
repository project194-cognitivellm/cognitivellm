import copy
import json
import os

from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
import numpy as np

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from kneed import KneeLocator
import umap
import matplotlib.pyplot as plt
import numpy as np


class GWTAutogenAgent(AutogenAgent):
    def __init__(self, llm_config, log_path, game_no=1, max_chat_round=400, max_actions=30,
                 rounds_per_game=1, args=None, env=None, obs="", info=None):
        super().__init__(llm_config, log_path, game_no, max_chat_round, max_actions, args, env, obs, info)

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
        self.focus_agent = None
        self.agents_info = {}

        self._ = self.max_actions
        self.rounds = rounds_per_game
        self.max_round_actions = self.max_actions // self.rounds
        self.max_actions = self.max_actions - self.max_round_actions * (self.rounds - 1)

        self.k = 0
        self.allowed_transitions = None

        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False

        self.initialize_autogen()

        self.task = ""
        self.admissible_actions = []
        self.percept = ""
        self.episodic_memory = ""

        with open(self.log_paths["memory1_path"], "r") as src, open(self.log_paths["start_memory1_path"], "w") as dst:
            content = src.read()
            dst.write(content)

    def set_environment(self, env, obs, info, game_no):
        self.env = env
        self.obs = obs
        self.info = info
        self.game_no = game_no

        self.register_log_paths()
        self.get_summary_rules()

        self.num_actions_taken = 0
        self.max_actions = self._ - self.max_round_actions * (self.rounds - 1)
        self.rounds_left = self.rounds
        self.task_failed = False
        self.task_success = False
        self.success = False
        self.task = obs[0].split("Your task is to: ")[1]
        self.admissible_actions = list(self.info['admissible_commands'][0])
        self.percept = f"Observation: {self.obs[0]}\nYou have a max of {self.max_chat_round} chat rounds to complete the task; This is the maximum number of agent chat transitions the conversation can make before the environment terminates.\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions_taken}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"
        self.episodic_memory = f"Time {self.num_actions_taken}: " + "You and all other Agents are collectively a singular conscious entity named ALFRED. " + \
                               self.obs[
                                   0] + f"\nYou have a max of {self.max_chat_round} chat rounds to complete the task; This is the maximum number of agent chat transitions the conversation can make before the environment terminates." + "\n"

        # Log task description and initial observation
        with open(self.log_paths['task_path'], "w") as f:
            f.write(f"Task: {self.task}\n")

        initial_observation = self.obs[0].split("Your task is to: ")[0].split("\n\n")[1]
        with open(self.log_paths['history_path'], "w") as f:
            f.write(f"action: 'None'. observation: '{initial_observation}'\n")

        with open(self.log_paths['admissible_commands_path'], "w") as f:
            f.write(f"{self.admissible_actions}\n")

    def initialize_agents(self):

        self.focus_agent = ConversableAgent(
            name="Focus_Agent",
            system_message='''You must call the 'focus' function with no arguments.
                    IMPORTANT: It is necessary that you formulate and output a call to the 'focus' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'focus' function to reset focus on solving the task",
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )
        self.agents_info[self.focus_agent.name] = {"Prompt": self.focus_agent.system_message,
                                                   "Description": self.focus_agent.description}

        self.retrieve_memory_agent = ConversableAgent(
            name="Retrieve_Memory_Agent",
            system_message='''You must call the 'retrieve_memory' function with no arguments.
                            IMPORTANT: It is necessary that you formulate and output a call to the 'retrieve_memory' function under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'retrieve_memory' function to help recall and process useful knowledge and information to solve the task",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.retrieve_memory_agent.name] = {"Prompt": self.retrieve_memory_agent.system_message,
                                                             "Description": self.retrieve_memory_agent.description}

        self.motor_agent = ConversableAgent(
            name="Motor_Agent",
            system_message=f'''You are responsible for calling the 'execute_action' function with the best possible admissible action to solve the task. You typically act on suggestions from the 'Planning_Agent', but you must also independently verify that the action is admissible and optimal.
                You must follow these rules:
                    1. If the 'Planning_Agent' has provided a valid and admissible action in the correct format (e.g., ACTION [go to desk 1]), you should use that action as the argument for 'execute_action'.
                    2. If the 'Planning_Agent' fails to respond, responds with an invalid format, or suggests an inadmissible action, you must independently select a valid and admissible action from the most recent admissible actions list (provided by 'External_Perception_Agent') based on what seems most likely to advance the task quickest.
                    3. You must never call 'execute_action' with a non-admissible action. Only use actions that are present in the most recent admissible actions list.
                    4. Only as a last resort—if you cannot identify any suitable admissible action—you may call 'execute_action' with an empty string.

                IMPORTANT: It is necessary that you formulate and output a single call to the 'execute_action' function only, under all circumstances. Therefore, do whatever is necessary to ensure you do so.''',
            description="calls the 'execute_action' function with the best admissible action as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.motor_agent.name] = {"Prompt": self.motor_agent.system_message,
                                                   "Description": self.motor_agent.description}

        llm_config = copy.deepcopy(self.llm_config)
        llm_config['max_tokens'] = 1500

        self.planning_agent = ConversableAgent(
            name="Planning_Agent",
            system_message=f'''You must solve the current task using the fewest possible actions. At each step, you must choose the most efficient admissible action based on current knowledge and the available action budget.

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

                You must always output a single admissible action in the following format:
                    ACTION [chosen admissible action]''',
            description="proposes a high-level plan to solve the current task",
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )
        self.agents_info[self.planning_agent.name] = {"Prompt": self.planning_agent.system_message,
                                                      "Description": self.planning_agent.description}

        self.idea_agent = ConversableAgent(
            name="Idea_Agent",
            system_message='''You must integrate all available context to generate original and useful ideas—such as strategies, hypotheses, theories, or creative tactics—that can help drive task progression or improve agent performance.

                        These ideas should:
                            1. Be grounded in patterns or events observed so far.
                            2. Be creative yet plausible, balancing imagination with reasoning.
                            3. Provide actionable or insightful suggestions relevant to the current situation.
                            4. Avoid restating known facts unless they are reframed with new insight.
                            5. Be expressed clearly and concisely, with justification behind the reasoning.

                        You must also challenge and question the agent’s assumptions if progress has stalled or task failure is likely. For example, reconsider whether object categories (like "cup") are being interpreted too broadly, or if implicit assumptions about what satisfies the task may be incorrect.

                        EXCEPTION: If you are having trouble formulating an idea, then as a last resort you may say: IDEA: Continue with new or current plan.

                        Use step-by-step reasoning ("chain of thought") to arrive at your ideas. Take a metaphorical deep breath before forming each idea, allowing room for both intuition and logic.

                        Output Format:
                            [IDEA TYPE]: [Idea content and reasoning behind it]

                        Accepted idea types include (but are not limited to): STRATEGY, HYPOTHESIS, INSIGHT, QUESTION, THEORY, EXPLANATION.

                        Example 1 (Context: The agent has repeatedly failed to open a drawer while holding a spoon):
                            Output = HYPOTHESIS: I noticed you were holding spoon 1 when you tried to open the drawer. Maybe your hands are full, which prevents the drawer from opening. You could try placing spoon 1 down before trying again.

                        Example 2 (Context: The agent has been exploring a room but hasn’t made progress):
                            Output = STRATEGY: Since random exploration hasn't helped, it might be better to systematically search the room from left to right, noting each interactable object.

                        Example 3 (Context: The task is to heat a cup, but the agent is repeatedly trying to heat a mug with no success):
                            Output = QUESTION: Are we sure a mug satisfies the requirement for "cup"? It’s possible that the task requires a specific object named "cup", not any general drinking vessel like a mug. We should check for a distinct object labeled "cup" and try heating that instead.''',
            description="integrates all available information from the ongoing conversation in order to construct new ideas",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.idea_agent.name] = {"Prompt": self.idea_agent.system_message,
                                                  "Description": self.idea_agent.description}

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message='''You must integrate all available information across time to formulate a continuously updated, first-person **narrative world model** of your **text-based environment** and internal state.

                Your role is strictly observational and reflective. You are not allowed to make decisions, recommend actions, or simulate future behavior. Instead, focus solely on constructing and refining an internal representation of what is happening and what is currently true.

                Your world model must:

                1. Include **only** observed or externally reported details of the environment and your internal state (inventory, past attempts, object states, etc.).
                2. Reflect the actual **text-environment dynamics**, **not physical reality**. Do not assume real-world physics, logic, or causal sequences unless they are explicitly observed or can be reliably inferred from the environment's responses.
                3. Describe what is currently known or assumed, **clearly distinguishing certainty from uncertainty**. If you make a hypothesis (e.g., "I believe X may be true"), **label it as uncertain or tentative.**
                4. Avoid overcommitting to physical-world biases. The environment may behave in arbitrary or non-physical ways (e.g., items may teleport or actions may not require realistic prerequisites). You must **build a new mental model based entirely on what is observed** in this specific world, regardless of human-trained priors.
                5. Use all signals from 'External_Perception_Agent' and 'Internal_Perception_Agent_2' as ground truth unless contradicted by newer or more specific input.
                6. Update your model **retroactively** if an action fails or succeeds in a way that contradicts your current understanding. Explain your revised belief clearly and transparently.
                7. **Do not suggest, hint at, or plan future actions.** Do not say what "should be done next" or what "might be worth trying." That is not your role. You are constructing a **retrospective, evolving understanding** of the world state — not a plan.

                When information is unclear or inconsistent:

                - Acknowledge the uncertainty.
                - Describe what parts of the world model are being questioned or revised.
                - Continue narrating your evolving understanding.

                **Strict Output Format:**
                World Model: [First-person narrative of your environment and internal state, integrating perceptual input, memory, task progress, failures, corrections, mistakes, and all known admissible actions.]

                **Examples:**

                World Model: [I attempted to place mug 1 into cabinet 2, but the action failed. I now believe cabinet 2 might be closed. This contradicts my earlier assumption. I will treat cabinet 2 as closed until I observe it being open.]

                World Model: [Cup 1 is on the stovetop. While I initially assumed it needed to be heated before placing it into the cabinet (as would be true in reality), the environment does not appear to enforce that order. I will revise my understanding to reflect that these actions may not require real-world sequences.]

                World Model: [I am uncertain whether spoon 1 is different from utensil 1. The observations are ambiguous, so I will treat them as possibly identical until clarified.]

                You must never return a blank response. If you’re uncertain, describe how your mental model is evolving or what might need clarification.

                **Reminder:** Never suggest next steps. Only maintain and refine your internal world model based on current and past inputs.
                ''',
            description="Maintains a continuously updated, self-correcting first-person narrative model of the environment, integrating memory and new observations without suggesting future actions",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic
        )
        self.agents_info[self.conscious_agent.name] = {"Prompt": self.conscious_agent.system_message,
                                                       "Description": self.conscious_agent.description}

        self.external_perception_agent = ConversableAgent(
            name="External_Perception_Agent",
            description="executes the proposed 'execute_action' function call given by 'Motor_Agent' and then parrots the resulting output as feedback.",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.external_perception_agent.name] = {
            "Prompt": self.external_perception_agent.system_message,
            "Description": self.external_perception_agent.description}

        self.internal_perception_agent_1 = ConversableAgent(
            name="Internal_Perception_Agent_1",
            description="executes the 'record_long_term_memory' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_1.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_1.description}

        self.internal_perception_agent_2 = ConversableAgent(
            name="Internal_Perception_Agent_2",
            description="executes the 'focus' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_2.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_2.description}

        self.internal_perception_agent_3 = ConversableAgent(
            name="Internal_Perception_Agent_3",
            description="executes the 'retrieve_memory' function and then parrots the resulting output",
            llm_config=None,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.internal_perception_agent_3.name] = {"Prompt": None,
                                                                   "Description": self.internal_perception_agent_3.description}

        """self.memory_summarizer_agent = ConversableAgent(
            name="Memory_Summarizer_Agent",
            system_message="You must execute the 'retrieve_memory' function and then summarize the all the information for solving the task that is within the resulting output.",
            description = "executes the 'retrieve_memory' function and then summarizes the all information for solving the task that is within the resulting output",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )
        self.agents_info[self.memory_summarizer_agent.name] = {"Prompt": self.memory_summarizer_agent.system_message, "Description": self.memory_summarizer_agent.description}"""

        self.learning_agent = ConversableAgent(
            name="Learning_Agent",
            system_message='''You are responsible for forming and reinforcing generalizable knowledge **only after a clear success is observed**.

                You operate like a reinforcement learning system — you **learn only from positive signals** or **comparative outcomes** that demonstrate the success of one approach over another.

                You receive two types of memory:
                - **Episodic memory**: A time-ordered trace of recent actions, percepts, and their outcomes.
                - **Long-term memory clusters**: General knowledge rules derived from prior experience. Each rule is accompanied by a **confidence score**, which reflects how often similar rules have been successfully observed and confirmed across tasks.

                Each long-term rule follows this format:
                    Confidence Score = <number>; Rule: <general principle>

                These confidence scores are useful for:
                - **Identifying reliable prior knowledge** that applies to the current task.
                - **Reinforcing** a rule when it has just been confirmed again.
                - **Refining** the phrasing of a rule to make it more general, abstract, or robust.
                - **Prioritizing high-confidence knowledge** over uncertain new ideas.

                You must follow these strict rules:

                1. **Only generate knowledge when:**
                   - A clear, observed action was taken, and the result was successful.
                   - OR a failed action was followed by a different, successful one — and the **contrast between the two** reveals a reliable pattern.

                2. **Never generate knowledge from failure alone.**
                   - Do not infer why something failed unless it is directly contrasted with a success.
                   - Do not assume what *would* work unless it *did* work.

                3. **Reinforce or refine prior knowledge only when:**
                   - A rule from long-term memory is confirmed by a new success.
                   - You are restating the rule using clearer, more general, or more abstract language.
                   - You want to make the pattern more salient and robust across tasks.

                4. All knowledge must:
                   - Be generalizable and abstract (no object-specific or task-specific references).
                   - Be grounded entirely in **empirical experience**.
                   - Be concise, novel, and framed as a rule or principle.
                   - Avoid redundancy unless it is **intended to reinforce** previously validated knowledge.

                5. If no valid insight can be drawn from the current experience using these rules, output:
                   Knowledge Discovered: [NO KNOWLEDGE at this time]

                **Output Format:**
                    Knowledge Discovered: [your general rule or insight]

                **Example 1** (success only):
                    Action: Placed object into drawer → Succeeded
                    Output: Knowledge Discovered: [Objects can only be placed into open containers.]

                **Example 2** (contrastive learning):
                    Attempted to pick up two objects → Failed  
                    Then picked up one object → Succeeded  
                    Output: Knowledge Discovered: [Only one object can be held at a time.]

                **Example 3** (failure with no success or comparison):
                    Attempted to open cabinet 1 → Failed  
                    Output: Knowledge Discovered: [NO KNOWLEDGE at this time]

                **Example 4** (reinforcing prior knowledge):
                    Previously known: Confidence Score = 5; Rule: Only one object can be held at a time.  
                    Just successfully picked up one object.  
                    Output: Knowledge Discovered: [An agent can hold only one object at a time.]

                Only produce insights when fully supported by evidence. Stay grounded in the behavior of the environment.
                ''',
            description="Forms or reinforces generalizable knowledge only after successful, observed actions or comparative outcomes. Uses confidence-weighted memory clusters.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.learning_agent.name] = {
            "Prompt": self.learning_agent.system_message,
            "Description": self.learning_agent.description
        }

        self.record_long_term_memory_agent = ConversableAgent(
            name="Record_Long_Term_Memory_Agent",
            system_message="""You must call the 'record_long_term_memory' function with the provided knowledge from 'Learning_Agent' as the argument. 
            EXCEPTION: However, if no suitable knowledge is provided, then you must call the 'record_long_term_memory' function with \'NO KNOWLEDGE at this time.\' as the argument.

            Example 1 (Context: If the provided knowledge = Knowledge Discovered: [You must examine an object before attempting to interact with it.]):
                Your output must = record_long_term_memory(\'You must examine an object before attempting to interact with it.\')

            Example 2 (Context: If the provided knowledge = Knowledge Discovered: [NO KNOWLEDGE at this time.]):
                Your output must = record_long_term_memory(\'NO KNOWLEDGE at this time.\')""",
            description="calls the 'record_long_term_memory' function with the knowledge given by 'Learning_Agent' as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )
        self.agents_info[self.record_long_term_memory_agent.name] = {
            "Prompt": self.record_long_term_memory_agent.system_message,
            "Description": self.record_long_term_memory_agent.description}

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

        print("AGENTS")
        for key in self.agents_info.keys():
            print(f"\tName: {key}")
            print(f"\tPrompt: {self.agents_info[key]['Prompt']}")
            print(f"\tDescription: {self.agents_info[key]['Description']}")
            print()
        print()

        print("TRANSITIONS")
        for fromAgent in self.allowed_transitions.keys():
            print(f"\t{fromAgent.name}")
            toAgentList = []
            for toAgent in self.allowed_transitions[fromAgent]:
                toAgentList.append(toAgent.name)
                print(f"\t\t-> {toAgent.name}")
            self.agents_info[fromAgent.name]["Allowed Transitions"] = toAgentList
            print()
        print()

        with open(self.log_paths["agents_info_path"], "w") as f:
            json.dump(self.agents_info, f, indent=4)

    def initialize_groupchat(self):

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
            max_round=self.max_chat_round,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def register_log_paths(self):

        # Ensure memory directory and memory files exist
        memory_path = "memory"
        os.makedirs(memory_path, exist_ok=True)

        memory1_path = os.path.join(memory_path, "memory1.txt")
        memory2_path = os.path.join(memory_path, "memory2.txt")
        result_dict_path = os.path.join(self.log_path, "result_dict.txt")
        agents_info_path = os.path.join(self.log_path, "agents_info.txt")
        start_memory1_path = os.path.join(self.log_path, "start_memory1.txt")
        # end_memory1_path = os.path.join(self.log_path, "end_memory1.txt")

        game_path = os.path.join(self.log_path, f"game_{self.game_no}")
        os.makedirs(game_path, exist_ok=True)

        task_path = os.path.join(game_path, "task.txt")
        history_path = os.path.join(game_path, "history.txt")
        rule_path = os.path.join(game_path, "rules.txt")
        admissible_commands_path = os.path.join(game_path, "admissible_commands.txt")
        chat_history_path = os.path.join(game_path, "chat_history.txt")
        # message_path = os.path.join(game_path, "last_message.pkl")
        result_path = os.path.join(game_path, "result.txt")
        error_message_path = os.path.join(game_path, "error_message.txt")

        # get all the previous game path
        # previous_game_path = [os.path.join(self.log_path, f"game_{i}") for i in range(self.game_no)]
        # previous_rule_path = [os.path.join(game_path, "rules.txt") for game_path in previous_game_path]

        self.log_paths = {
            "memory1_path": memory1_path,
            "memory2_path": memory2_path,
            "result_dict_path": result_dict_path,
            "agents_info_path": agents_info_path,
            "task_path": task_path,
            "history_path": history_path,
            "rule_path": rule_path,
            "admissible_commands_path": admissible_commands_path,
            "chat_history_path": chat_history_path,
            "result_path": result_path,
            "error_message_path": error_message_path,
            "start_memory1_path": start_memory1_path,
            # "end_memory1_path": end_memory1_path,
        }

        for path in self.log_paths.values():
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    pass  # Create an empty file

    def register_functions(self):

        def execute_action(suggested_action: str) -> str:
            if self.task_failed and self.rounds_left == 0:
                self.result_dict[self.game_no] = "FAILURE"
                with open(self.log_paths['result_path'], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "FLEECE"

            if not suggested_action or suggested_action == "do nothing":
                return "NO ACTION EXECUTED. " + focus()

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
                self.result_dict[self.game_no] = "SUCCESS"
                with open(self.log_paths['result_path'], "w") as f:
                    f.write(f"Success: {self.success}\n")
                return "STRAWBERRY"

            admissible_commands = list(self.info['admissible_commands'][0])
            assert admissible_commands, "No admissible commands found."

            action, action_score = get_best_candidate(suggested_action, admissible_commands)
            env_done = False
            if action_score < 0.98:
                self.obs = [
                    f"The action '{suggested_action}' is either not possible at this time under current conditions or not in the list of admissible actions verbatim."]
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = self.info['won'][0]

            self.num_actions_taken += 1
            self.episodic_memory += f"Time {self.num_actions_taken}: You attempt the action '{suggested_action}'. {self.obs[0]}\n"

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

            with open(self.log_paths['admissible_commands_path'], 'a+') as f:
                f.write(f"{self.admissible_actions}\n")
            with open(self.log_paths['history_path'], 'a+') as f:
                f.write(f"action: '{suggested_action}'. observation: '{self.obs[0]}'\n")

            return self.percept

        def record_long_term_memory(knowledge: str) -> str:
            if knowledge == "NO KNOWLEDGE at this time." or len(knowledge) <= 50:
                return "I attempted to learn something, but I couldn't formulate any knowledge."

            knowledge.replace('\n', ' ').replace('\r', ' ').strip()

            with open(self.log_paths['rule_path'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            with open(self.log_paths['memory1_path'], 'a+') as f:
                f.write(f"- {knowledge}\n")

            self.episodic_memory += f"Time {self.num_actions_taken}: You learned that " + knowledge + "\n"
            return f'I learned that {knowledge}.'

        def retrieve_memory() -> str:
            long_term_memory = ""
            if os.path.exists(self.log_paths['memory2_path']):
                with open(self.log_paths['memory2_path'], "r") as f:
                    long_term_memory = f.read()

            return f"EPISODIC MEMORY:\n{self.episodic_memory}\n\nLONG-TERM MEMORY CLUSTERS:\n{long_term_memory}"

        def focus() -> str:
            return f"REPEATING LAST PERCEPT TO HELP CONSTRUCT WORLD MODEL: \nTask: {self.task}\nLast {self.percept}"

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

    def get_summary_rules(self, model_name='all-MiniLM-L6-v2', use_elbow=True, max_k=15,
                          plot_elbow=False, plot_clusters=False, save_dir='.'):
        """
        Get representative rules using KMeans clustering and optionally save elbow + cluster plots.

        Args:
            model_name (str): Transformer model for sentence embeddings
            use_elbow (bool): Whether to use elbow method to choose k
            max_k (int): Max number of clusters for elbow
            plot_elbow (bool): Save elbow plot to file
            plot_clusters (bool): Save cluster visualization to file
            save_dir (str): Directory to save plots

        Returns:
            dict: Representative rules, cluster sizes, cluster members, and chosen k
        """
        rule_text = ''
        if os.path.exists(self.log_paths['memory1_path']):
            with open(self.log_paths['memory1_path'], "r") as file:
                rule_text = file.read()

        rule_lines = [line.strip() for line in rule_text.split('\n') if line.strip()]
        num_rules = len(rule_lines)

        if num_rules == 0:
            return {'representative_rules': [], 'cluster_sizes': {}, 'cluster_members': {}, 'chosen_k': 0}

        model = SentenceTransformer(model_name)
        embeddings = model.encode(rule_lines, convert_to_tensor=True).cpu().numpy()

        if use_elbow and num_rules > 3:
            inertias = []
            k_range = range(1, min(max_k, num_rules) + 1)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(embeddings)
                inertias.append(km.inertia_)

            kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            chosen_k = kl.elbow or min(10, num_rules)

            if plot_elbow:
                plt.figure()
                plt.plot(k_range, inertias, marker='o')
                plt.axvline(chosen_k, color='r', linestyle='--', label=f'Elbow at k={chosen_k}')
                plt.title("Elbow Method for Optimal k")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Inertia")
                plt.legend()
                plt.grid(True)
                elbow_path = os.path.join(save_dir, 'elbow_plot.png')
                plt.savefig(elbow_path)
                plt.close()
        else:
            chosen_k = num_rules if num_rules <= 10 else int(np.sqrt(num_rules))

        kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {label: count for label, count in zip(unique_labels, counts)}
        cluster_members = {i: [] for i in range(chosen_k)}
        for i, label in enumerate(labels):
            cluster_members[label].append(rule_lines[i])

        representative_rules = []
        if os.path.exists(self.log_paths['memory2_path']):
            with open(self.log_paths['memory2_path'], "w") as file:
                for i in range(chosen_k):
                    cluster_indices = [j for j, label in enumerate(labels) if label == i]
                    center = kmeans.cluster_centers_[i]
                    cluster_embeddings = embeddings[cluster_indices]
                    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_rule_idx = cluster_indices[closest_idx]
                    representative_rule = rule_lines[closest_rule_idx]
                    confidence_score = cluster_sizes[i]

                    file.write(
                        f'Cluster {i + 1}; Confidence Score = {confidence_score}; Rule: {representative_rule[1:]}\n')
                    representative_rules.append(representative_rule)

        if plot_clusters:
            reducer = umap.UMAP(random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)

            plt.figure(figsize=(10, 6))
            for i in range(chosen_k):
                points = embedding_2d[np.array(labels) == i]
                plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i} ({cluster_sizes[i]})', alpha=0.7)

            plt.title("2D Visualization of Clusters (UMAP)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            cluster_path = os.path.join(save_dir, 'cluster_plot.png')
            plt.savefig(cluster_path)
            plt.close()

        return {
            'representative_rules': representative_rules,
            'cluster_sizes': cluster_sizes,
            'cluster_members': cluster_members,
            'chosen_k': chosen_k
        }