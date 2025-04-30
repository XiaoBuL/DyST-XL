## DyST-XL: Dynamic Layout Planning and Content Control for Compositional Text-to-Video Generation
Concatenate the compositional text to be analyzed after the text in [template.txt](template.txt), input it into the LLM for dynamic layout planning, and then extract the output to obtain xxx.yaml.

The information and format that need to be extracted are as follows:
```
base_prompt: str
if_change: bool
(initial_split_ratio&final_split_ratio) or split_ratio: list[list[float]]
global_prompt: str
local_prompt: list[str]
```

To generate video, input the config path and run the command:
```
python cli_demo.py --config_pth xxx.yaml
```