import re
import copy
import ast

from typing import Dict, List, Tuple, Callable
from autogen import ConversableAgent
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from sentence_transformers import SentenceTransformer, util


def parse_tool_call(tool_call_string: str) -> Tuple[str, Tuple]:
    """
    Parse a tool call string to extract the tool call name and parameters.
    For example: "function_name('param1', 42)" -> ("function_name", ("param1", 42))
    """
    pattern = r"(\w+)\((.*?)\)"
    match = re.match(pattern, tool_call_string)
    if match:
        tool_call_name = match.group(1)
        parameters = match.group(2).strip()
        if parameters:
            # Safely evaluate parameters
            try:
                # Wrap parameters so ast.literal_eval interprets them as a tuple
                args = ast.literal_eval(f"({parameters},)")
            except SyntaxError:
                # If something is malformed, treat parameters as a single string
                args = (parameters,)
        else:
            args = ()
        return tool_call_name, args
    else:
        raise ValueError(f"Invalid tool call string format: {tool_call_string}")


class MessageToolCall:
    def __init__(self, tool_dict: Dict[str, Callable]):
        # Ensure all values in tool_dict are callable.
        self.tool_dict = tool_dict
        for _, tool in tool_dict.items():
            if not callable(tool):
                raise ValueError("All tools must be callable functions.")

    def _transform_text_content(self, text: str) -> str:
        """
        For a given text string, find and replace all occurrences of tool calls
        defined in self.tool_dict.
        """
        # For each tool_name, repeatedly find and replace all calls
        for tool_name, func in self.tool_dict.items():
            # Build a pattern that matches this specific tool call
            # Note: The non-greedy .*? is used to match minimal parameters
            # While still allowing multiple calls.
            pattern = rf"{re.escape(tool_name)}\((.*?)\)"

            match = re.search(pattern, text)
            if not match:
                # No more occurrences of this tool
                continue
            # Extract the full matched substring
            full_call_str = text[match.start():match.end()]
            print(f"Match: {pattern, full_call_str}")
            # Parse it
            parsed_tool_name, args = parse_tool_call(full_call_str)
            if parsed_tool_name == tool_name:
                result = func(*args)
                return f"ECHO: {result}"
            else:
                # If somehow parsing didn't match the tool_name, break to avoid infinite loop
                raise ValueError(f"Tool name mismatch: {parsed_tool_name} != {tool_name}")
        return text

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)

        message = temp_messages[-1]
        # If content is a simple string
        if isinstance(message["content"], str):
            message["content"] = self._transform_text_content(message["content"])

        # If content is a list, iterate over text-type items
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item.get("type") == "text" and isinstance(item["text"], str):
                    item["text"] = self._transform_text_content(item["text"])

        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        # Compare pre and post transformation messages for changes.
        for message, post_message in zip(pre_transform_messages, post_transform_messages):
            if message["content"] != post_message["content"]:
                return "Function call triggered", True
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
