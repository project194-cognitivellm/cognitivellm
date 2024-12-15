import os
import pickle
import re
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu
from helpers import get_best_candidate, register_function_lambda, is_termination_msg_generic, get_echo_agent
from autogen_agent import AutogenAgent


class MemoryAutogenAgent(AutogenAgent):
    def __init__(self, env, obs, info, llm_config, log_path, game_no, max_actions=50, args=None):
        super().__init__(env, obs, info, llm_config, log_path, game_no, max_actions, args)

        self.assistant_agent = None
        self.executor_agent = None
        self.grounding_agent = None
        self.echo_agent = None
        self.lesson_recorder_agent = None
        self.long_term_memory_summarizer_agent = None

        # This will store the ongoing memory summary that the long_term_memory_summarizer updates.
        self.memory_summary = "Long-Term Memory Summary:\n"

        self.initialize_autogen()

    def initialize_agents(self):
        # Assistant Agent
        self.assistant_agent = ConversableAgent(
            name="Assistant_Agent",
            system_message=(
                "You generate plans and make action decisions to solve a task. You will receive feedback "
                "from other agents on the result of executing your action decisions. You must use this "
                "feedback when generating further plans and action decisions. "
                "You will also sometimes receive a long-term memory summary from the Echo_Agent, which records "
                "lessons learned from previous tasks and steps. Incorporate these lessons if they are relevant. "
                "Your output should have the following format: THOUGHT[...your reasoning...] ACTION[...your action...] "
                "Your proposed action should be one of the admissible commands. Never assume you have successfully "
                "completed the task until the environment ends. If you do not provide an action suggestion, you fail."
            ),
            llm_config=self.llm_config,
            is_termination_msg=lambda msg: False,
            human_input_mode="NEVER"
        )

        # Executor Agent
        def executor_agent_termination_msg(msg):
            return (
                    msg["name"] == "Executor_Agent"
                    and msg["content"] is not None
                    and msg["content"][:5] != "ECHO:"
            )

        self.executor_agent = ConversableAgent(
            name="Executor_Agent",
            system_message=(
                "You call the execute_action function with the proposed action as the argument. For example, "
                "if the proposed action is ACTION[go to desk 1], you should output "
                "execute_action(\"go to desk 1\"). You must include a call to the execute_action function "
                "in your output, or you will fail the task."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False,
        )

        # Echo Agent (same as before)
        self.echo_agent = get_echo_agent(
            self.llm_config,
            additional_termination_criteria=[executor_agent_termination_msg]
        )

        # Lesson Recorder Agent
        # The lesson_recorder should look at the most recent feedback and produce a new memory if needed.
        # If no new memory is to be recorded, output "No new memory".
        self.lesson_recorder_agent = ConversableAgent(
            name="Lesson_Recorder_Agent",
            system_message=(
                "You are the Lesson Recorder. Your role is to produce new, useful lessons based on recent observations and actions. "
                "You have access to the most recent results, observations, and possibly previous memory summaries that detail "
                "lessons learned in the past.\n\n"
                ""
                "Guidelines for producing a new lesson:\n"
                "1. **Only produce a new lesson when it adds clear value.** Don't produce trivial or overly repetitive lessons. "
                "If a similar lesson already exists, only generate a new one if it refines, contradicts, or meaningfully expands "
                "on an existing lesson.\n"
                "2. **Correctness and Utility:** If recent events show that a previously recorded lesson might be misguided, "
                "impractical, or incorrect, produce a new lesson that clarifies or corrects it. For example, if previously you learned "
                "to always approach a problem in a certain way, but now you have evidence that a different approach works better, "
                "record a new lesson that corrects or supersedes the old one.\n"
                "3. **Generalizability and Future Orientation:** Lessons should be applicable beyond the immediate scenario. "
                "Focus on strategies, principles, and best practices that will help solve similar tasks in the future. "
                "Avoid overly specific or ephemeral details.\n"
                "4. **Conciseness and Clarity:** Make the lesson a single, clear statement of advice or insight. Avoid long-winded "
                "descriptions or multiple unrelated points. A good lesson is short, to the point, and easy to understand.\n"
                "5. **Only One Lesson at a Time:** If you have multiple insights, choose the one that is the most important or "
                "widely applicable. If you need to store multiple insights, consider whether they can be combined into a single, "
                "more general principle.\n\n"
                ""
                "Your output should be one of the following:\n"
                "- If no new meaningful lesson can be drawn from the latest feedback (i.e., it doesn't add substantial new insights "
                "or correct previous lessons), simply output: 'No new memory'.\n"
                "- Otherwise, output one new lesson in the format: 'NEW MEMORY: <lesson>'\n\n"
                ""
                "Always follow output formatting exactly, ot your message will not be processed."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        # Long Term Memory Summarizer
        # It receives the new memory from the lesson_recorder_agent. It must integrate it into the existing summary.
        # Then it must call record_memory_summary("<integrated summary>") as its sole output.
        self.long_term_memory_summarizer_agent = ConversableAgent(
            name="Long_Term_Memory_Summarizer_Agent",
            system_message=(
                "You are the Long Term Memory Summarizer. Your purpose is to maintain a well-organized, growing set of lessons "
                "learned from past experiences. You have an existing long-term memory summary (which is a list of lessons) and "
                "you receive a new lesson from the Lesson_Recorder_Agent. Your task is to:\n\n"
                "1. Integrate the new lesson into the existing list of lessons. Each lesson is a single, self-contained statement "
                "of advice, insight, or strategy learned.\n"
                "2. If the new lesson contradicts or refines previously stored lessons, remove or update the old ones to avoid confusion.\n"
                "3. If multiple lessons can be combined or generalized into a single, broader lesson, do so. Avoid redundancy.\n"
                "4. Keep the lessons concise, actionable, and future-oriented. Avoid overly specific details that won't apply broadly. "
                "Ensure each lesson is understandable at a glance.\n"
                "6. If the Lesson_Recorder_Agent says 'No new memory', do not change the list. Simply re-record the same existing list.\n\n"
                "7. The updated summary must contain all relevant lessons (including integrated new insights) after your modifications.\n\n"
                ""
                "The final output must be a call to record_memory_summary(\"\"\"<updated summary>\"\"\"), where <updated summary> "
                "is represented as a list of strings representing the current set of lessons. For example:\n\n"
                "record_memory_summary(\"\"\"1. <lesson 1>\n2. <lesson 2>\n3. <lesson 3>\"\"\")\n\n"
                "Do not output anything else beyond the single function call. Your entire output should be the function call "
                "with the updated lessons as a list of strings. If you do not follow the output format exactly, your "
                "message will not be processed."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: False
        )

        self.assistant_agent.description = "generates plans and makes action decisions to solve the task"
        self.executor_agent.description = (
            "calls execute_action with the proposed action as the argument"
        )
        self.lesson_recorder_agent.description = (
            "records new lessons learned from the latest events"
        )
        self.long_term_memory_summarizer_agent.description = (
            "integrates new lessons into a long-term memory summary and records it"
        )
        self.echo_agent.description = "echoes function outputs and memory updates."

        # The start agent could be the echo_agent or assistant_agent depending on desired start.
        # Typically we start from the assistant_agent. But here, let's start from assistant_agent for a new round.
        self.start_agent = self.assistant_agent

    def register_functions(self):
        # Define execute_action
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            self.num_actions += 1

            action, action_score = get_best_candidate(suggested_action, admissible_commands)

            if action_score < 0.8:
                self.obs = [f"action '{suggested_action}' is not admissible."]
                self.success = False
            else:
                self.obs, scores, dones, self.info = self.env.step([action])
                self.success = dones[0]

            if self.success:
                return f"Observation: {self.obs[0]}\nTask Status: SUCCESS\nActions Left: {self.max_actions - self.num_actions}"
            elif self.num_actions >= self.max_actions:
                return f"Observation: {self.obs[0]}\nTask Status: FAILURE\nActions Left: {self.max_actions - self.num_actions}"
            else:
                return f"Observation: {self.obs[0]}\nTask Status: INCOMPLETE\nActions Left: {self.max_actions - self.num_actions}"

        # Define record_memory_summary
        def record_memory_summary(summary: str) -> str:
            # Store the summary in the agent's memory
            self.memory_summary = summary
            return self.memory_summary

        def get_recorded_memory():
            return self.memory_summary

        register_function_lambda(
            {r'execute_action': execute_action, r'record_memory_summary': record_memory_summary},
            [self.echo_agent]
        )

        register_function_lambda(
            {r'get_recorded_memory': get_recorded_memory},
            [self.long_term_memory_summarizer_agent],
            last_message_only=True, append_echo=True, echo_signifier="Previous Memory Summary:",
            search_for_tool_call=False
        )

    def initialize_groupchat(self, max_chat_round=2000):
        def state_transition(last_speaker, groupchat):
            messages = groupchat.messages

            with open(self.log_paths['message_path'], "wb") as f:
                pickle.dump(messages, f)

            if last_speaker is self.assistant_agent:
                next_speaker = self.executor_agent
            elif last_speaker is self.executor_agent:
                next_speaker = self.echo_agent
            elif last_speaker is self.echo_agent:
                if messages[-2]["name"] == "Executor_Agent":
                    next_speaker = self.lesson_recorder_agent
                elif messages[-2]["name"] == "Long_Term_Memory_Summarizer_Agent":
                    next_speaker = self.assistant_agent
                else:
                    raise ValueError(f"Unknown echo entry: {messages[-2]['name']}")
            elif last_speaker is self.lesson_recorder_agent:
                next_speaker = self.long_term_memory_summarizer_agent
            elif last_speaker is self.long_term_memory_summarizer_agent:
                next_speaker = self.echo_agent
            else:
                raise ValueError(f"Unknown speaker: {last_speaker}")

            return next_speaker

        self.group_chat = GroupChat(
            agents=[
                self.assistant_agent,
                self.executor_agent,
                self.echo_agent,
                self.lesson_recorder_agent,
                self.long_term_memory_summarizer_agent
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
