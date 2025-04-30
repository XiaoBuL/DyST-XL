#  DyST-XL: Dynamic Layout Planning and Content Control for Compositional Text-to-Video Generation
Official implementation of the paper "[DyST-XL: Dynamic Layout Planning and Content Control for Compositional Text-to-Video Generation]([https://arxiv.org/abs/2312.03805](https://arxiv.org/abs/2504.15032))".


## Usage
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


## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@article{he2025dyst,
  title={DyST-XL: Dynamic Layout Planning and Content Control for Compositional Text-to-Video Generation},
  author={He, Weijie and Liu, Mushui and Yu, Yunlong and Wang, Zhao and Wu, Chao},
  journal={arXiv preprint arXiv:2504.15032},
  year={2025}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at lms@zju.edu.cn.
