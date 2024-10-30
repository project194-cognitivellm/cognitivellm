import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

from nltk.translate.bleu_score import sentence_bleu
from autogen import ConversableAgent, register_function, GroupChat, GroupChatManager
import os

# load config
config = generic.load_config()
env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)
print("Initialized Environment")

# interact
obs, info = env.reset()
print("Reset environment")
print(f"Admissible Commands: {info['admissible_commands'][0]}")

def get_best_candidate(reference_sentence, candidate_sentences):
    # Tokenize the reference sentence
    # print(f"Reference Sentence: {reference_sentence}")
    reference = [reference_sentence.split()]
    best_score = 0.0
    best_candidate = ""

    # Iterate through each candidate sentence and calculate the BLEU score
    for candidate_sentence in candidate_sentences:
        candidate = candidate_sentence.split()
        bleu_score = sentence_bleu(reference, candidate)
        # print(f"Candidate Sentence: {candidate_sentence}, BLEU: {bleu_score}")

        # Update best score and best candidate if this candidate is better
        if bleu_score > best_score:
            best_score = bleu_score
            best_candidate = candidate_sentence

    # print(f"Best Candidate: {best_candidate}, BLEU: {best_score}")
    return best_candidate

def execute_action(suggested_action: str) -> str:
    global info
    assert len(list(info['admissible_commands'])) == 1
    admissible_commands = list(info['admissible_commands'][0])
    assert len(admissible_commands) > 0
    action = get_best_candidate(suggested_action, admissible_commands)
    obs, scores, dones, info = env.step([action])
    return obs[0], f"Scores: {scores[0]}, Dones: {dones[0]}"

llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]}

assistant_agent = ConversableAgent(
    name="Assistant_Agent",
    system_message="You generate plans and make action decisions to solve a task. You will receive feedback from"
                   "other agents on the result of executing your action decisions. You must use this feedback when "
                   "generating further plans and action decisions."
                   "Your output should have the following format: THOUGHT[contents of your plan.] ACTION[proposed action.]"
                   "Example 1: "
                   "Task Description: [You are in the middle of a room. Looking quickly around you, you see a bed 1,"
                   " a desk 2, a desk 1, a safe 1, a drawer 2, a drawer 1, a shelf 3, a shelf 2, and a shelf 1. "
                   "Your task is to: look at bowl under the desklamp.]"
                   "Your Output: THOUGHT [First, I need to find a bowl. A bowl is more likely to appear in desk "
                   "(1-2), drawer (1-2), shelf (1-3), bed (1). Then I need to find and use a desklamp.] "
                   "ACTION [go to desk 1]"
                   "Example 2 (After assistant finds the desklamp at desk 1, then goes to desk 2.): "
                   "Feedback: [on the desk 2, you see a bowl 1, and a cd 3]"
                   "Your Output: THOUGHT [Now I find a bowl (1). I need to use the desklamp to look at the bowl. "
                   "I'll go to the desklamp now.] ACTION [go to desk 1]",
    llm_config=llm_config,
    is_termination_msg=lambda msg: msg["content"] is not None and "Task failed" in msg["content"],
    human_input_mode="NEVER"
)

print("Initialized assistant agent")

environment_proxy = ConversableAgent(
    name="Environment Proxy",
    llm_config=False,
    human_input_mode="NEVER"
)

print("Initialized environment proxy")

executor_agent = ConversableAgent(
    name="Executor_Agent",
    system_message="You call the execute_action function with the proposed action as the argument",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

print("Initialized executor agent")

grounding_agent = ConversableAgent(
    name="Grounding_Agent",
    system_message="You provide general knowledge at the start of task when the chat begins and whenever the "
                    "environment_proxy reports the same results three times in a row",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

print("Initialized grounding agent")

allowed_transitions = {
    assistant_agent: [executor_agent],
    executor_agent: [environment_proxy],
    grounding_agent: [executor_agent],
    environment_proxy: [assistant_agent, grounding_agent]
}

assistant_agent.description = "generates plans and makes action decisions to solve the task"
executor_agent.description = "calls execute_action with the proposed action as the argument to perform the suggested action"
environment_proxy.description = "reports action execution results as feedback."
grounding_agent.description = ("provides general knowledge at the start of task when the chat begins and whenever the "
                               "environment_proxy reports the same results three times in a row")

group_chat = GroupChat(
    agents=[assistant_agent, executor_agent, grounding_agent, environment_proxy],
    messages=[],
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed",
    max_round=20,
    send_introductions=True
)

print("Initialized group chat")

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

print("Initialized group chat manager")

register_function(
    execute_action,
    caller=executor_agent,
    executor=environment_proxy,
    name ="execute_function",
    description="Call this function to execute the suggested action"
)

print("Initialization complete; Starting Chat")

if isinstance(obs, (list, tuple)):
    initial_message_content = obs[0]
else:
    initial_message_content = obs

chat_result = grounding_agent.initiate_chat(
        group_chat_manager,
        message={"role": "system", "content": initial_message_content},
        summary_method="reflection_with_llm"
)

print("Finished Chat")
