# ALFWorld Project Log

## Environment Setup
```bash
export ALFWORLD_DATA=/home/yixiao/yixiao/course/cs294/project/cognitivellm/alfworld_data
```

## Task Analysis

### Failed Tasks (1-75)
Total Failed: 54 tasks
```python
[
    1, 2, 4, 5, 6, 10, 11, 13, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 28, 30, 
    31, 33, 34, 35, 36, 38, 39, 42, 43, 44, 45, 48, 49, 50, 51, 53, 54, 55, 56, 
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 73
]
```


### Output Organization
Each experiment run is saved in a timestamped directory under `runs/`:

```
runs/
├── YYYYMMDD_HHMMSS/
│   ├── game_0/
│   │   ├── memory.txt
│   │   ├── admissible_commands.txt
│   │   ├── chat_history.txt
│   │   ├── task_info.txt
│   │   └── result.txt
│   ├── game_1/
│   │   ├── memory.txt
│   │   ├── admissible_commands.txt
│   │   ├── chat_history.txt
│   │   ├── task_info.txt
│   │   └── result.txt
│   ├── result_list.txt
│   └── ...
└── ...
```
