import os
import re
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent
#from transformers import pipeline

class GWTAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path=None, max_actions=50):
        super().__init__(env, obs, info, llm_config, log_path, max_actions)

        self.planning_agent = None
        self.executor_agent = None
        self.imagination_agent = None
        self.echo_agent = None
        self.echo_agent2 = None
        self.conscious_agent = None
        self.memory_retrieval_agent = None

        self.narrative_state = ""
        #self.merge_pipeline = pipeline("text2text-generation", model="t5-small")

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
                "VERY IMPORTANT: So long as you are being queried, you have not yet successfully completed the task. "
                "Never assume you have successfully completed the task. Once you complete the task, the chat will end "
                "on its own. If you do not provide an action suggestion, you will fail the task."
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        def executor_agent_termination_msg(msg):
            return msg["name"] == "Executor_Agent" and msg["content"] is not None and msg["content"][:5] != "ECHO:"

        self.echo_agent = get_echo_agent(self.llm_config,
                                         additional_termination_criteria=[executor_agent_termination_msg])

        self.echo_agent2 = get_echo_agent(self.llm_config)

        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="You call the execute_action function with the proposed action as the argument. For "
                           "example, if the proposed action is ACTION[go to desk 1], you should output "
                           "execute_action(\"go to desk 1\"). You must include a call to the execute_action function "
                           "in your output, or you will fail the task. If no proposed action is given, choose a random admissible action as the argument.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.imagination_agent = ConversableAgent(
            name="Imagination_Agent",
            system_message=(
                "Your goal is to help Planning_Agent solve the given task by "
                "providing new ideas, theories, explanations, and hypotheses whenever Planning_Agent is confused or is proposing repetitive actions."
                "Example Output: (After taking spoon 1)"
                "\nI noticed we were holding spoon 1 when we tried to open the drawer. Maybe the reason we couldn't open the drawer is because our hands are full. We need to place the spoon 1 somewhere before attempting to open the drawer again"
                "VERY IMPORTANT: So long as you are being queried, you have not yet successfully completed the task. "
                "Never assume you have successfully completed the task. Once you complete the task, the chat will end "
                "on its own."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.conscious_agent = ConversableAgent(
            name="Conscious_Agent",
            system_message=(
                "You are Conscious_Agent. Your role is to integrate all available information from the ongoing conversation and maintain a continuously updated, first-person narrative model of your environment and your actions within it. This narrative should:"
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
            llm_config = self.llm_config,
            human_input_mode = "NEVER",
            is_termination_msg = lambda msg: False,
        )

        self.memory_retrieval_agent = ConversableAgent(
            name="Memory_Retrieval_Agent",
            system_message="You call the get_environment_model function with the proposed model update as the argument. "
                            "For example, if the update is Model Update: [I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.]"
                            ", you should output get_environment_model(\'I am in a room with drawers (1-5), cabinets (1-14), and countertops (1-3). My task is to find spoon 1 and place it in a drawer. I found spoon 1 on countertop 1 and attempted to put it into drawer 1, but I was unable to open that drawer. I am now deciding what to do next.\'). "
                            "You must include a call to the get_environment_model function in your output, or you will fail the task. If no model update is given, call get_environment_model with an empty string as the argument.",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.allowed_transitions = {
            self.planning_agent: [self.executor_agent, self.imagination_agent],
            self.executor_agent: [self.echo_agent],
            self.imagination_agent: [self.planning_agent],
            self.echo_agent: [self.conscious_agent],
            self.conscious_agent: [self.memory_retrieval_agent],
            self.memory_retrieval_agent: [self.echo_agent2],
            self.echo_agent2: [self.planning_agent, self.imagination_agent]
        }

        self.planning_agent.description = "generates plans and makes action decisions to solve the task"
        self.executor_agent.description = (
            "calls execute_action with the proposed action as the argument to perform the suggested action"
        )
        self.imagination_agent.description = (
            "helps Planning_Agent solve the given task using the least amount of actions by "
            "providing new ideas whenever Planning_Agent is confused or is proposing repetitive and inefficient actions."
        )
        self.echo_agent.description = "reports action execution results as feedback."
        self.echo_agent2.description = "reports environment model as feedback."
        self.conscious_agent.description = "integrates all available information from the ongoing conversation and maintains a continuously updated, first-person narrative model of the environment and actions within it"
        self.memory_retrieval_agent.description = "calls get_environment_model with the proposed model update as the argument"

        self.start_agent = self.echo_agent

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < .97 and action_score >= .8:
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
                return f"Observation: {self.obs[0]}\nTask Status: SUCCESS\nActions Left: {self.max_actions - self.num_actions}"
            elif self.num_actions >= self.max_actions:
                return f"Observation: {self.obs[0]}\nTask Status: FAILURE\nActions Left: {self.max_actions - self.num_actions}"
            else:
                return f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}\nCurrent Admissible Actions: {list(self.info['admissible_commands'][0])}"

        def get_environment_model(new_info: str) -> str:
            self.narrative_state += f"Step {self.num_actions}: " + new_info + "\n"
            return self.narrative_state

        register_function_lambda(
            {r'execute_action': execute_action},
            [self.echo_agent]
        )

        register_function_lambda(
            {r'get_environment_model': get_environment_model},
            [self.echo_agent2]
        )

    def initialize_groupchat(self, max_chat_round=2000):
        self.group_chat = GroupChat(
            agents=[
                self.planning_agent,
                self.executor_agent,
                self.imagination_agent,
                self.echo_agent,
                self.echo_agent2,
                self.conscious_agent,
                self.memory_retrieval_agent
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
