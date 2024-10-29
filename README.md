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
