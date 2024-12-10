import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent


class BaselineAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path=None, max_actions=50):
        super().__init__(env, obs, info, llm_config, log_path, max_actions)

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
                "Your Output: THOUGHT [Now I find a bowl 1. I need to take bowl 1 back to desk 1 with a desklamp. "
                "I'll take bowl 1 first.] ACTION [take bowl 1 from desk 2]. "
                "VERY IMPORTANT: So long as you are being queried, you have not yet successfully completed the task. "
                "Never assume you have successfully completed the task. Once you complete the task, the chat will end "
                "on its own."
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        self.echo_agent = get_echo_agent(self.llm_config)

        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="You call the execute_action function with the proposed action as the argument. For "
                           "example, if the proposed action if ACTION[go to desk 1], you should output "
                           "execute_action(\"go to desk 1\").",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.grounding_agent = ConversableAgent(
            name="Grounding_Agent",
            system_message=(
                "You provide general knowledge at the start of task when the chat begins and whenever the "
                "environment_proxy reports the same results three times in a row. "
                "VERY IMPORTANT: So long as you are being queried, you have not yet successfully completed the task. "
                "Never assume you have successfully completed the task. Once you complete the task, the chat will end "
                "on its own."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        self.allowed_transitions = {
            self.assistant_agent: [self.executor_agent],
            self.executor_agent: [self.echo_agent],
            self.grounding_agent: [self.assistant_agent],
            self.echo_agent: [self.assistant_agent, self.grounding_agent]
        }

        self.assistant_agent.description = "generates plans and makes action decisions to solve the task"
        self.executor_agent.description = (
            "calls execute_action with the proposed action as the argument to perform the suggested action"
        )
        self.grounding_agent.description = (
            "provides general knowledge at the start of task when the chat begins and whenever the "
            "environment_proxy reports the same results three times in a row."
        )
        # self.echo_agent.description = "reports action execution results as feedback."

        self.start_agent = self.echo_agent

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.8:
                return f"Observation: action '{suggested_action}' is not admissible."

            self.obs, scores, dones, self.info = self.env.step([action])

            # time.sleep(1)
            if dones[0]:
                return f"Observation: {self.obs[0]} SUCCESS"
            elif self.num_actions >= self.max_actions:
                return f"Observation: {self.obs[0]} FAILURE"
            else:
                return f"Observation: {self.obs[0]} IN_PROGRESS"

        register_function_lambda(
            {r'execute_action': execute_action},
            [self.echo_agent]
        )

    def initialize_groupchat(self, max_chat_round=2000):
        self.group_chat = GroupChat(
            agents=[
                self.assistant_agent,
                self.executor_agent,
                self.grounding_agent,
                self.echo_agent
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
