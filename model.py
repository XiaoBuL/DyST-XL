import torch
from diffusers.models.attention_processor import Attention
from typing import Optional
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.cogvideo.pipeline_cogvideox import replace_example_docstring, retrieve_timesteps, EXAMPLE_DOC_STRING, CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
import math
from transformers import T5EncoderModel, T5Tokenizer



class DyST_XL(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler]
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        if max_sequence_length is not None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        else:            
            text_inputs = self.tokenizer(
                prompt,
                padding=False,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds


    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        noise_weight: float = 0.95,
        noise_step: int = 10,
        split_ratio: Optional[Union[List, torch.Tensor]] = None
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if split_ratio is not None:
            if isinstance(split_ratio, List):
                initial_split_ratio = split_ratio[0]
                final_split_ratio = split_ratio[1]
                initial_split_ratio = (initial_split_ratio * latents.shape[-2]).round().long()
                final_split_ratio = (final_split_ratio * latents.shape[-2]).round().long()
                step = torch.linspace(0, 1, latents.shape[1]).view(-1, 1)
            else:
                split_ratio = (split_ratio * latents.shape[-2]).round().long()


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                if i < noise_step:
                    if isinstance(split_ratio, List):
                        latents_motion = latents.clone()
                        for initial_split, final_split in zip(initial_split_ratio, final_split_ratio):
                            splits = ((1 - step) * initial_split + step * final_split).round().long()
                            box_noise = latents[:, 0, :, splits[0, 0]:splits[0, 2], splits[0, 1]:splits[0, 3]].clone()
                            for j in range(latents.shape[1]):
                                latents_motion[:, j, :, splits[j, 0]:splits[j, 2], splits[j, 1]:splits[j, 3]] = torch.nn.functional.interpolate(box_noise, size=(splits[j, 2]-splits[j, 0], splits[j, 3]-splits[j, 1]))
                        latents = noise_weight * latents + (1-noise_weight) * latents_motion
                    else:
                        for split in split_ratio:
                            latents[:, :, :, split[0]:split[2], split[1]:split[3]] = latents[:, :, :, split[0]:split[2], split[1]:split[3]]*noise_weight+(1-noise_weight)*latents[:, :1, :, split[0]:split[2], split[1]:split[3]]

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)


                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

class DualAttenionProcessor:
    attn_mask = None
    def __init__(self, base_ratio=1.0, stop_step=0, size=[13,30,45]):
        self.base_ratio = base_ratio
        self.step = 0
        self.stop_step = stop_step
        self.size = size
        self.index = torch.tensor([0, 2, 1, 3], dtype=torch.long).cuda()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, text_seq_length = encoder_hidden_states.size(0), encoder_hidden_states.size(1)
        n, h, w = self.size

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                
        # query, shift_query = query.chunk(2)
        # key, shift_key = key.chunk(2)
        # value, shift_value = value.chunk(2)
        # uncond_query, query = query.chunk(2)
        # uncond_key, key = key.chunk(2)
        # uncond_value, value = value.chunk(2)
        if self.step < self.stop_step:
            query_global, query_local = query[[0, 2]], query[[1, 3]]
            key_global, key_local = key[[0, 2]], key[[1, 3]]
            value_global, value_local = value[[0, 2]], value[[1, 3]]
            
            hidden_states_global = F.scaled_dot_product_attention(
                query_global, key_global, value_global, dropout_p=0.0, is_causal=False
            )
            hidden_states_local = F.scaled_dot_product_attention(
                query_local, key_local, value_local, attn_mask=self.attn_mask, dropout_p=0.0, is_causal=False
            )
            self.step += 1

            hidden_states_global = hidden_states_global.transpose(1, 2).reshape(batch_size//2, -1, attn.heads * head_dim)
            hidden_states_global = attn.to_out[0](hidden_states_global)
            hidden_states_global = attn.to_out[1](hidden_states_global)
            encoder_hidden_states_global, hidden_states_global = hidden_states_global.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            
            hidden_states_local = hidden_states_local.transpose(1, 2).reshape(batch_size//2, -1, attn.heads * head_dim)
            hidden_states_local = attn.to_out[0](hidden_states_local)
            hidden_states_local = attn.to_out[1](hidden_states_local)
            encoder_hidden_states_local, hidden_states_local = hidden_states_local.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            
            encoder_hidden_states = torch.cat([encoder_hidden_states_global, encoder_hidden_states_local], 0)
            encoder_hidden_states = encoder_hidden_states.index_select(dim=0, index=self.index)
            
            hidden_states = self.base_ratio * hidden_states_global + (1 - self.base_ratio) * hidden_states_local
            hidden_states = torch.cat([hidden_states, hidden_states], 0)
            hidden_states = hidden_states.index_select(dim=0, index=self.index)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
                
            self.step += 1

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
        
        
        
        return hidden_states, encoder_hidden_states