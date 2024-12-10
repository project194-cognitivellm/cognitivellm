import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic
from autogen_agent import AutogenAgent


class BaselineAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path=None):
        super().__init__(env, obs, info, llm_config, log_path)

    def initialize_agents(self):
        self.assistant_agent = ConversableAgent(
            name="Assistant_Agent",
            system_message=(
                "You generate plans and make action decisions to solve a task. You will receive feedback from "
                "other agents on the result of executing your action decisions. You must use this feedback when "
                "generating further plans and action decisions. "
                "The feedback includes the observations, the admissible commands, scores, and dones. "
                "Your output should have the following format: THOUGHT[contents of your plan.] ACTION[proposed action.] "
                "Your proposed action should be one of the admissible commands. "
                "Example 1: "
                "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1, "
                "a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                "Your task is to: look at bowl under the desklamp.] "
                "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                "ACTION [go to desk 1] "
                "Example 2 (After assistant finds the desklamp at desk 1, then goes to desk 2.): "
                "Feedback: [on the desk 2, you see a bowl 1, and a cd 3] "
                "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                "I'll go to the desklamp now.] ACTION [go to desk 1]"
            ),
            llm_config=self.llm_config,
            is_termination_msg=is_termination_msg_generic,
            human_input_mode="NEVER"
        )

        self.environment_proxy = ConversableAgent(
            name="Environment Proxy",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="You call the execute_action function with the proposed action as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.grounding_agent = ConversableAgent(
            name="Grounding_Agent",
            system_message=(
                "You provide general knowledge at the start of task when the chat begins and whenever the "
                "environment_proxy reports the same results three times in a row"
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg_generic,
        )

        self.allowed_transitions = {
            self.assistant_agent: [self.executor_agent],
            self.executor_agent: [self.environment_proxy],
            self.grounding_agent: [self.executor_agent, self.assistant_agent],
            self.environment_proxy: [self.assistant_agent, self.grounding_agent]
        }

        self.assistant_agent.description = "generates plans and makes action decisions to solve the task"
        self.executor_agent.description = (
            "calls execute_action with the proposed action as the argument to perform the suggested action"
        )
        self.environment_proxy.description = "reports action execution results as feedback."
        self.grounding_agent.description = (
            "provides general knowledge at the start of task when the chat begins and whenever the "
            "environment_proxy reports the same results three times in a row. If it is the start of the task, "
            "call assistant_agent to generate the first plan. If the task is completed, output SUCCESS."
        )

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.8:
                return f"Observation: action '{suggested_action}' is not admissible."

            self.obs, scores, dones, self.info = self.env.step([action])

            # time.sleep(1)
            if dones[0]:
                return f"Observation: {self.obs[0]} SUCCESS"
            else:
                return f"Observation: {self.obs[0]} IN_PROGRESS"

        register_function_lambda(execute_action, r"execute_action", self.executor_agent)

    def initialize_groupchat(self, max_chat_round=200):
        self.group_chat = GroupChat(
            agents=[
                self.assistant_agent,
                self.executor_agent,
                self.grounding_agent,
                self.environment_proxy
            ],
            messages=[],
            allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
            speaker_transitions_type="allowed",
            max_round=50,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
