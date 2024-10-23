# CognitiveLLM


## Table of Contents

- [Installation](#installation)
- [Description](#description)


## Installation

1. Install ALFWorld. Specific your storage_path
    ```sh
    conda create -n cogllm python=3.9
    conda activate cogllm
    pip install alfworld[full]
    export ALFWORLD_DATA=<storage_path>
    alfworld-download
    alfworld-download --extra
    python alfworld_test.py configs/base_config.yaml
    ```
2. AutoGen?
    

## Description
### ALFWorld
ALFWorld is a gym environment. The initial observation includes the environment state and **Task Description**. Following observations only include environment states. We can let AutoGen call the *env.step()* function and get the observation. See details in alfworld_test.py.