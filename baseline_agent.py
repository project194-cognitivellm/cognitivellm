import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu


class BaselineAutogenAgent:
    def __init__(self, env, obs, info, llm_config):
        self.env = env
        self.obs = obs
        self.info = info
        self.llm_config = llm_config

        self.initialize_agents()
        self.register_functions()

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
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
            human_input_mode="NEVER"
        )

        self.environment_proxy = ConversableAgent(
            name="Environment Proxy",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
        )

        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message="You call the execute_action function with the proposed action as the argument",
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
        )

        self.grounding_agent = ConversableAgent(
            name="Grounding_Agent",
            system_message=(
                "You provide general knowledge at the start of task when the chat begins and whenever the "
                "environment_proxy reports the same results three times in a row"
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: msg["content"] is not None and "SUCCESS" in msg["content"],
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

    def register_functions(self):
        register_function(
            self.execute_action,
            caller=self.executor_agent,
            executor=self.environment_proxy,
            name="execute_action",
            description="Call this function to execute the suggested action"
        )

    def get_best_candidate(self, reference_sentence, candidate_sentences):
        # Tokenize the reference sentence
        reference = [reference_sentence.split()]
        best_score = 0.0
        best_candidate = ""

        # Iterate through each candidate sentence and calculate the BLEU score
        for candidate_sentence in candidate_sentences:
            candidate = candidate_sentence.split()
            bleu_score = sentence_bleu(reference, candidate)

            # Update best score and best candidate if this candidate is better
            if bleu_score > best_score:
                best_score = bleu_score
                best_candidate = candidate_sentence

        return best_candidate

    def execute_action(self, suggested_action: str) -> str:
        assert len(list(self.info['admissible_commands'])) == 1
        admissible_commands = list(self.info['admissible_commands'][0])
        assert len(admissible_commands) > 0

        action = self.get_best_candidate(suggested_action, admissible_commands)
        self.obs, scores, dones, self.info = self.env.step([action])
        return self.obs[0], f"Admissible Commands: {admissible_commands}, Scores: {scores[0]}, Dones: {dones[0]}"

    def run_chat(self):
        if isinstance(self.obs, (list, tuple)):
            initial_message_content = self.obs[0]
        else:
            initial_message_content = self.obs

        chat_result = self.grounding_agent.initiate_chat(
            self.group_chat_manager,
            message={"role": "system", "content": initial_message_content},
            summary_method="reflection_with_llm"
        )

        return chat_result
