import re
import copy

from typing import Dict, List, Tuple, Callable
from autogen import ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from sentence_transformers import SentenceTransformer, util


def parse_tool_call(tool_call_string):
    """
    Parse a tool call string to extract the tool call name and parameters.

    :param tool_call_string: A string in the format "function_name(parameters)"
    :return: A tuple (tool_call_name, parameters)
    """
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, tool_call_string)
    if match:
        tool_call_name = match.group(1)
        parameters = match.group(2)
        return tool_call_name, parameters
    else:
        raise ValueError("Invalid tool call string format")


# The transform must adhere to transform_messages.MessageTransform protocol.
class MessageToolCall:
    def __init__(self, tool_dict: Dict[str, Callable]):
        # PATTERN TO MATCH TOOL CALL NAME.
        self.tool_dict = tool_dict
        for _, tool in tool_dict.items():
            if not callable(tool):
                raise ValueError("The input must be a callable python function.")

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)

        for message in temp_messages:
            if isinstance(message["content"], str):
                print(f"checking message {message['content']} for tool call")
                for pattern, function in self.tool_dict.items():
                    print(f"checking against pattern {pattern}")
                    if re.match(pattern, message["content"]):
                        print(f"PASS")
                        tool_name, param = parse_tool_call(message["content"])
                        # result = tool_name(param)
                        message["content"] = function(param)  # TOOL CALL !!
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        for pattern, function in self.tool_dict.items():
                            if re.match(pattern, item["text"]):
                                item["text"] = function(message["content"])  # TOOL CALL !!
        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        for message, post_message in zip(pre_transform_messages, post_transform_messages):
            if message["content"] != post_message["content"]:
                return (f"Message content changed from \n\n {message['content']}\n\n to\n\n {post_message['content']}",
                        True)
        return "", False


def register_function_lambda(tool_dict: Dict[str, Callable], agents: List[ConversableAgent]):
    tool_handling = transform_messages.TransformMessages(
        transforms=[MessageToolCall(tool_dict)])
    for agent in agents:
        tool_handling.add_to_agent(agent)


sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_best_candidate(reference_sentence, candidate_sentences):
    # Compute embeddings
    target_embedding = sentence_transformer_model.encode(reference_sentence, convert_to_tensor=True)
    command_embeddings = sentence_transformer_model.encode(candidate_sentences, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(target_embedding, command_embeddings)

    # Find the most similar command
    most_similar_idx = similarities.argmax()
    most_similar_command = candidate_sentences[most_similar_idx]
    score = similarities.detach().cpu().numpy()[0, most_similar_idx]

    return most_similar_command, score


def is_termination_msg_generic(msg):
    return msg["content"] is not None and ("SUCCESS" in msg["content"] or "FAILURE" in msg["content"])
