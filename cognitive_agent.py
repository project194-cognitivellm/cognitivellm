import os
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
from nltk.translate.bleu_score import sentence_bleu


def get_best_candidate(reference_sentence, candidate_sentences):
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


class CognitiveAutogenAgent:
    def __init__(self, env, obs, info, llm_config):
        self.env = env
        self.obs = obs
        self.info = info
        self.llm_config = llm_config

        # Initialize plan memory database
        self.plan_memory_database = {}

        self.initialize_agents()
        self.register_functions()

    def initialize_agents(self):
        # Planner Agent
        self.planner_agent = ConversableAgent(
            name="Planner_Agent",
            system_message=(
                "You propose a plan or sub-plan to accomplish the given task. "
                "You have access to the plan memory database, which stores the latest result for each proposed plan. "
                "Based on the plan memory, decide whether to stick to the same plan or propose a new one."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

        # Actor Agent
        self.actor_agent = ConversableAgent(
            name="Actor_Agent",
            system_message=(
                "You propose actions to try to accomplish the current plan provided by the Planner Agent. "
                "Your actions should be one of the admissible commands."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

        # Plan Result Evaluator Agent
        self.evaluator_agent = ConversableAgent(
            name="Evaluator_Agent",
            system_message=(
                "You evaluate the extent to which the plan has been accomplished based on the latest observation. "
                "Your evaluation could be: partially accomplished, accomplished, ran into an unexpected error, plan failed, etc. "
                "Your evaluation could also include any insights you've gathered from the observation relevant to the plan choice. "
                "Update the plan memory database with the latest result."
            ),
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )

        # Environment Proxy
        self.environment_proxy = ConversableAgent(
            name="Environment_Proxy",
            llm_config=False,
            human_input_mode="NEVER",
        )

        # Allowed transitions between agents
        self.allowed_transitions = {
            self.planner_agent: [self.actor_agent],
            self.actor_agent: [self.environment_proxy],
            self.environment_proxy: [self.evaluator_agent],
            self.evaluator_agent: [self.planner_agent],
        }

        # Agent descriptions
        self.planner_agent.description = "proposes plans to accomplish the task"
        self.actor_agent.description = "proposes actions to execute the current plan"
        self.evaluator_agent.description = "evaluates the progress of the plan based on observations"
        self.environment_proxy.description = "executes actions and returns observations"

        # Group Chat
        self.group_chat = GroupChat(
            agents=[
                self.planner_agent,
                self.actor_agent,
                self.evaluator_agent,
                self.environment_proxy
            ],
            messages=[],
            allowed_or_disallowed_speaker_transitions=self.allowed_transitions,
            speaker_transitions_type="allowed",
            max_round=100,
            send_introductions=True
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )

    def register_functions(self):
        # Define execute_action as a nested function
        def execute_action(suggested_action: str) -> str:
            assert len(list(self.info['admissible_commands'])) == 1
            admissible_commands = list(self.info['admissible_commands'][0])
            assert len(admissible_commands) > 0

            action = get_best_candidate(suggested_action, admissible_commands)
            self.obs, scores, dones, self.info = self.env.step([action])
            return self.obs[0], f"Admissible Commands: {admissible_commands}, Scores: {scores[0]}, Dones: {dones[0]}"

        # Register the execute_action function
        register_function(
            execute_action,
            caller=self.environment_proxy,
            executor=self.environment_proxy,
            name="execute_action",
            description="Execute the action in the environment and return the observation"
        )

        # Define update_plan_memory as a nested function
        def update_plan_memory(plan: str, evaluation: str):
            self.plan_memory_database[plan] = evaluation
            return f"Plan memory updated for plan: '{plan}' with evaluation: '{evaluation}'"

        # Register the update_plan_memory function
        register_function(
            update_plan_memory,
            caller=self.evaluator_agent,
            executor=self.evaluator_agent,
            name="update_plan_memory",
            description="Update the plan memory database with the evaluation of the plan"
        )

    def run_chat(self):
        if isinstance(self.obs, (list, tuple)):
            initial_message_content = self.obs[0]
        else:
            initial_message_content = self.obs

        # Start the chat with the Planner Agent proposing a plan
        chat_result = self.planner_agent.initiate_chat(
            self.group_chat_manager,
            message={"role": "system", "content": initial_message_content},
            summary_method="reflection_with_llm"
        )

        return chat_result
