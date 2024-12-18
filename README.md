# SallCo: Specialized Agentic Large Language Models for Cognitive Functions

## Table of Contents
- [Description](#description)
  - [SallCo](#sallco)
  - [ALFWorld](#alfworld)
- [Installation](#installation)
  - [Ubuntu](#ubuntu)
  - [Windows (via WSL)](#windows-via-wsl)
  - [Lambda Compatibility](#lambda-compatibility)
- [Quick Start](#quick-start)
  - [Agent Types](#agent-types)
  - [Optional Arguments](#optional-arguments)
  - [Output and Logging](#output-and-logging)
- [Additional Features](#additional-features)
  - [Resume Chat](#resume-chat)
  - [Loop Transition](#loop-transition)


## Description

### SallCo
SallCo is a framework for creating specialized agentic large language models for cognitive functions based on AutoGen.

### ALFWorld
**ALFWorld** is a text-basedgym environment, where we use to test SallCo. The initial observation includes the environment state and a **Task Description**. Subsequent observations only include the environment state. AutoGen can interact with `env.step()` to receive new observations.

**Scripts**:
1. `alfworld_random_agent.py`: Run a random agent on the ALFWorld environment.
2. `interactive_alfworld.py`: Interactively run ALFWorld environment steps.

## Installation

### Ubuntu
1. Install ALFWorld and test run autogen
    ```sh
    conda create -n cogllm python=3.9
    conda activate cogllm
    pip install 'alfworld[full]'
    export ALFWORLD_DATA=<storage_path>
    alfworld-download
    alfworld-download --extra
    ```

2. Install Autogen.
   ```sh
   pip install autogen-agentchat~=0.2
   pip install nltk
   pip install flaml[automl]
   ```

### Windows
Be sure to use WSL in Windows to install the `jericho` package. Ideally, if in Windows, should use WSL to install conda, if not done so already. If using windows, please be sure to run the following commands.


```sh
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install python3-dev
sudo apt-get install python3-pip
sudo apt-get install zlib1g-dev
sudo apt-get install libssl-dev libffi-dev
pip3 install cython
```
Then, follow the ubuntu installation instructions.


### Lambda Compatibility
In order to make our code compatible with Lambda models, I needed to make a few changes to the base code 
which worked with OpenAI models. Here are the changes:

1. Update llm_config as follows:
```sh
API_KEY = os.environ.get("LAMBDA_API_KEY")
BASE_URL = "https://api.lambdalabs.com/v1"
MODEL = "llama3.1-70b-instruct-berkeley"

llm_config = {
    "timeout": 1000,
    "cache_seed": None,
    "max_tokens": 300,
    "config_list": [{"model": MODEL, "api_key": API_KEY, "base_url": BASE_URL}]}
```
2. export the API key
```sh
export LAMBDA_API_KEY=<your_api_key>
```

3. When registering tools, instead of using the library's register_function method, I used a custom method defined in
`helpers.py`. Also, instead of registering the function to the agent which calls it, I register all functions to an
Echo Agent which states the output of the executed function. See baseline_agent.py and gwt_agent.py for examples.

You can run the code with the following command (`--baseline` can be replaced with `--gwt` to use gwt_agent.py):
```sh
python run_autogen_eval_loop --baseline configs/eval_config.yaml
```



## Quick Start
You can try SallCo with the following command:
```sh
python run_autogen_eval_loop.py --gwt_rule --long_term_memory configs/eval_config.yaml --start_game_no 0 --log_path runs/rule_long_term
```

### Agent Types
You can choose from the following agent types:
- `--baseline`: Baseline agent from AUTOGEN paper
- `--gwt`: SallCo with guidance agent
- `--gwt_rule`: SallCo with rule agent
- `--gwt_rule_simplified`: SallCo with simplified rule agent

### Optional Arguments
- `--long_term_memory`: Enable long-term memory feature
  - The agent will summarize and learn task-relevant guidance or task-irrelevant rules in each game
  - Guidance/rules are saved to `guidance.txt` / `rule.txt`
  - Previous games' guidances or rules will influence current game decisions
- `--log_path <path>`: Specify custom log directory
  - Defaults to timestamp-based naming if not provided
- `--start_game_no <number>`: Set starting game number
  - Defaults to 1 if not provided

### Output and Logging
- Metrics are saved to:
  - Specified log path
  - Weights & Biases (default project: "cogllm")
- Error messages from LLM agents are recorded in the log file


## Additional Features
### Resume Chat
Resume chat from the last message. Group chat sometimes fails due to some errors like server error. We cannot count it as a failure and we need to resume the chat. Check the following code for details.
```sh
max_num_of_resume = 3
if chat_result is None:
    
    for i in range(max_num_of_resume):
        if os.path.exists(message_path):
            with open(message_path, "rb") as f:
                last_message = pickle.load(f)   
            
        chat_result, error_message = agent.resume_chat(last_message)
        
        if chat_result is not None:
            break
```

### Loop Transition
When A talks to B, B talks to C. But sometimes, B will need to talk to A again. For example, command evaluation agent finds there are no addmissible command which can complete the task. Then he needs to tell back to previous agent, let him replan. So there is a loop transition. Check the following code for details.
```sh
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    # record the last message
    # last message is a Dict. use pickle to save it.
    with open(message_path, "wb") as f:
        pickle.dump(messages, f)
    
    if last_speaker is self.memory_agent:
        
        next_speaker = self.retrive_memory_agent
    elif last_speaker is self.retrive_memory_agent:
        next_speaker = self.task_agent
    elif last_speaker is self.task_agent:
        next_speaker = self.command_evaluation_agent
    elif last_speaker is self.command_evaluation_agent:
        if "Task_Agent" in messages[-1]["content"]:
            next_speaker = self.task_agent
        else:
            next_speaker = self.executor_agent
    elif last_speaker is self.executor_agent:
        next_speaker = self.memory_agent

    tool_call_flag = "tool_calls" in messages[-1]
    if tool_call_flag:
        return last_speaker
    else:
        return next_speaker


# Group Chat
self.group_chat = GroupChat(
    agents=[
        self.memory_agent,
        self.retrive_memory_agent,
        self.task_agent,
        self.command_evaluation_agent,
        self.executor_agent,
    ],
    messages=[],
    speaker_selection_method=state_transition,
    max_round=200,
    send_introductions=True
)
```


