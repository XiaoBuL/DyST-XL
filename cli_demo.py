"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""
import logging
import argparse

import torch
from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
)

from diffusers.utils import export_to_video
logging.basicConfig(level=logging.INFO)
from model import DyST_XL, DualAttenionProcessor
import os
import yaml

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (1360, 768),
    "cogvideox1.5-5b": (1360, 768),

    # cogvideox-*
    "cogvideox-5b-i2v": (720, 480),
    "cogvideox-5b": (720, 480),
    "cogvideox-2b": (720, 480),
}

def attention_mask_generation(size, split_ratio, prompt_index, text_len=226):
    n, h, w = size
    if isinstance(split_ratio, list):
        split_ratio = [(split_ratio[0] * h).round().long(), (split_ratio[1] * h).round().long()]
    else:
        split_ratio = (split_ratio * h).round().long()
    t2t = torch.zeros(text_len)
    t2v = torch.zeros(text_len, n*h*w, dtype=bool)
    v2v = torch.zeros(n,h,w)
    
    t2v[:prompt_index[0][0]] = True
    for i, index in enumerate(prompt_index):
        mask = torch.zeros(n, h, w, dtype=bool)
        if isinstance(split_ratio, list):
            initial_split, final_split = split_ratio[0][i], split_ratio[1][i]
            t = torch.linspace(0, 1, n).view(-1, 1)
            splits = ((1 - t) * initial_split + t * final_split).round().long()
            for j, split in enumerate(splits):
                mask[j, split[0]:split[2], split[1]:split[3]] = True
                if v2v[j, split[0]:split[2], split[1]:split[3]].max()!=0:
                    v2v[j, split[0]:split[2], split[1]:split[3]] = v2v[j, split[0]:split[2], split[1]:split[3]].max()
                else:
                    v2v[j, split[0]:split[2], split[1]:split[3]] = i+1
        else:
            split = split_ratio[i]
            mask[:, split[0]:split[2], split[1]:split[3]] = True
            if v2v[:, split[0]:split[2], split[1]:split[3]].max()!=0:
                v2v[:, split[0]:split[2], split[1]:split[3]] = v2v[:, split[0]:split[2], split[1]:split[3]].max()
            else:
                v2v[:, split[0]:split[2], split[1]:split[3]] = i+1
        t2v[index[0]:index[1]] = mask.flatten().unsqueeze(0)
        t2t[index[0]:index[1]] = i+1
    t2t = torch.eq(t2t.unsqueeze(1), t2t.unsqueeze(0))
    v2v = v2v.flatten()
    v2v = torch.eq(v2v.unsqueeze(1), v2v.unsqueeze(0))
    attention_mask = torch.cat([torch.cat([t2t, t2v], 1), torch.cat([t2v.t(), v2v], 1)], 0)

    return attention_mask
    
def information_load(args):
    with open(args.config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
        prompt = [config['global_prompt']] + config['local_prompt']
        if config['if_change']:
            split_ratio = [torch.tensor(config['initial_split_ratio']), torch.tensor(config['final_split_ratio'])]
        else:
            split_ratio = torch.tensor(config['split_ratio'])
        return prompt, split_ratio


def load_model(args, dtype):
    model_name = args.model_path.split("/")[-1].lower()
    width, height = RESOLUTION_MAP[model_name]
    n, h, w = (args.num_frames-1)//4+1, height//16, width//16
    logging.info(f"\033[1mUsing default resolution (weight, height) for {model_name}\033[0m")

    pipe = DyST_XL().from_pretrained(args.model_path, torch_dtype=dtype)

    for i, block in enumerate(pipe.transformer.transformer_blocks):
        if i < args.stop_block:
            block.attn1.processor = DualAttenionProcessor(base_ratio=args.base_ratio, stop_step=args.stop_step, size=[n, h, w])
        else:
            break

    if 'CogVideoX-2b' in args.model_path:
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_sequential_cpu_offload()
    return pipe, height, width

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, default="Nestled in a sun-drenched glade, an elderly man with a face etched with the wisdom of years sits on a rustic bench, his voice a gentle stream of tales. Dressed in a faded denim shirt and worn khakis, his hands dance with the rhythm of storytelling. Beside him, a little girl with pigtails and a dress as bright as a summer's day leans in, her eyes sparkling with curiosity. Above, on a branch of an ancient oak, a cat with a sleek coat and eyes as green as the leaves listens, its tail curled in contentment, as the story weaves its magic.")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="Path of the pre-trained model use"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=8, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--base_ratio", type=float, default=0.8)
    parser.add_argument("--stop_step", type=int, default=10)
    parser.add_argument("--stop_block", type=int, default=10)
    parser.add_argument("--noise_weight", type=float, default=0.95)
    parser.add_argument("--noise_step", type=int, default=10)
    parser.add_argument("--config_pth", type=str, default='xxx.yaml')
    parser.add_argument("--output_dir", type=str, default='output')

    args = parser.parse_args()

    dtype = torch.float16 if 'CogVideoX-2b' in args.model_path else torch.bfloat16
    pipe, height, width = load_model(args, dtype)
    n, h, w = (args.num_frames-1)//4+1, height//16, width//16
    prompt, split_ratio = information_load(args)
    output_path = os.path.join(args.output_dir, args.config_pth.split('.')[0]+'.mp4')
    print(output_path)
    prompt_len = pipe.tokenizer(prompt[1:], padding="max_length", max_length=226, truncation=True, add_special_tokens=True, return_tensors="pt")['attention_mask'].sum(dim=1)
    init_index = 0
    prompt_index = []
    for i,l in enumerate(prompt_len):
        prompt_index.append([init_index, init_index+l.item()-1])
        init_index = init_index+l.item()-1
    attention_mask = attention_mask_generation([n, h, w], split_ratio, prompt_index)
    DualAttenionProcessor.attn_mask = attention_mask.to(distributed_state.device)
    prompt = [prompt[0], " ".join(prompt[1:])]
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        if i < args.stop_block:
            block.attn1.processor.step = 0
        else:
            break
    video_generate = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_videos_per_prompt=args.num_videos_per_prompt,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        use_dynamic_cfg=True,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator().manual_seed(args.seed),
        noise_weight=args.noise_weight,
        noise_step=args.noise_step,
        split_ratio=split_ratio
    ).frames[0]
    export_to_video(video_generate, output_path, fps=args.fps)
