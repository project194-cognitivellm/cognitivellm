# CognitiveLLM


## Table of Contents

- [Installation](#installation)
- [Description](#description)


## Installation

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
### Ubuntu
1. Install ALFWorld and test run autogen
    ```sh
    conda create -n cogllm python=3.9
    conda activate cogllm
    pip install 'alfworld[full]'
    export ALFWORLD_DATA=<storage_path>
    alfworld-download
    alfworld-download --extra
    python run_autogen configs/base_config.yaml
    ```

2. Install Autogen.
   ```sh
   pip install autogen-agentchat~=0.2
   pip install nltk
   pip install flaml[automl]
   ```

## Description
### ALFWorld
ALFWorld is a gym environment. The initial observation includes the environment state and **Task Description**. Following observations only include environment states. We can let AutoGen call the *env.step()* function and get the observation.

Scripts:
1. `alfworld_random_agent.py`: Runs random agent on ALFWorld environment.
2. `interactive_alfworld.py`: Interactive mode to run ALFWorld environment.

### AutoGen
We apply AutoGen to the ALFWorld environment, mimicking the conversation pattern and prompts from the original AutoGen paper.

Scripts:
1. `run_autogen.py`: Runs a single trajectory of an AutoGen agent on ALFWorld environment.


### Evaluation
Evaluation script for ALFWorld. It is controlled by user.
```sh
python alfworld_eval.py configs/eval_config.yaml 
```

Evaluation script for AutoGen on ALFWorld.
```sh
python run_autogen.py configs/eval_config.yaml 
```

### Loop GWT
Loop GWT script for AutoGen on ALFWorld.
```sh
python loop_gwt_run_autogen_eval.py configs/eval_config.yaml 
```

Key features:
1. Resume chat from the last message. If the chat fails, it will try 3 times. Check the following code for details.
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

2. Loop transition. When A talks to B, B talks to C. But sometimes, B will need to talk to A again. For example, command evaluation agent finds there are no addmissible command which can complete the task. Then he needs to tell back to previous agent, let him replan. So there is a loop transition. Check the following code for details.
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

