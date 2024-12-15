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

2. When registering tools, instead of using the library's register_function method, I used a custom method defined in
`helpers.py`. Also, instead of registering the function to the agent which calls it, I register all functions to an
Echo Agent which states the output of the executed function. See baseline_agent.py and gwt_agent.py for examples.

You can run the code with the following command (`--baseline` can be replaced with `--gwt` to use gwt_agent.py):
```sh
python run_autogen_eval_loop --baseline configs/eval_config.yaml
```

### Loop GWT (Old)
Loop GWT script for AutoGen on ALFWorld.
```sh
python run_autogen_eval_loop.py configs/eval_config.yaml 
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

### Long Term Guidance
This is a new feature that I added to the code. In every game, the agent will summarize and learn the guidance of the task. It will be saved in the `guidance.txt` file. During this game, the guidance will be used to guide the agent's action. Long term guidance means that the guidance learned in previous games will be used to guide the agent's action in current game. Use `--long_term_guidance` to enable this feature. 

To run the code with long term guidance, use the following command:

```sh
python run_autogen_eval_loop --gwt --long_term_guidance configs/eval_config.yaml
```


### Lambda
export LAMBDA_API_KEY="secret_cog-llm_4ac4efd85b9a4552904bd5b4630c60a4.Al0KM1fEEmrokHGcJQrlQyW4SY8OuCvU"
export ALFWORLD_DATA=

