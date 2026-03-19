# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
LoopOFT Framework

A VLA framework that supports Teacher Forcing with dual VLM:
  - Student: Full VLM (Vision + Language)
  - Teacher: Text-only LLM (frozen)

The student learns by matching the teacher's hidden states while predicting actions.
Key Features:
  - Qwen3-VL as backbone
  - Teacher Forcing via hidden state matching
  - L1 regression for action prediction
  
"""

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("LoopOFT")
class LoopOFT(baseframework):
    """
    LoopOFT: VLA with Teacher Forcing support.

    Components:
      - Student VLM: Full Vision-Language Model
      - Teacher LLM: Text-only LLM (frozen)
      - Action Head: MLP for action prediction

    Training:
      - Student predicts actions via VLM + Action Head
      - Student matches Teacher's hidden states (MSE loss)
      - Final loss = action_loss + teacher_loss * weight

    Inference:
      - Only use Student VLM + Action Head
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        
        self.loop_token_id = self.qwen_vl_interface.loop_token_id

        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()
        
        self._init_halting_projector(self.config.framework.qwenvl.num_loop)

        self.use_teacher_llm = self.config.framework.use_teacher_llm
        self.computation_penalty_weight = self.config.framework.get("computation_penalty_weight", 0.0)
        self.halting_entropy_weight = self.config.framework.get("halting_entropy_weight", 0.0)
        self.initial_teacher_loss_weight = config.framework.get("teacher_loss_weight", 0.5)
        self.teacher_loss_weight = self.initial_teacher_loss_weight
        self.teacher_loss_decay_steps = config.framework.get("teacher_loss_decay_steps", 1000)
        # Loop effectiveness loss parameters
        self.use_loop_effectiveness_loss = config.framework.get("use_loop_effectiveness_loss", False)
        self.loop_effectiveness_k = config.framework.get("loop_effectiveness_k", 1.0)
        self.loop_effectiveness_gamma = config.framework.get("loop_effectiveness_gamma", 0.0)
        self._optimizer_step = 0
        self._gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps", 1)
        self._forward_step_count = 0

    def _init_halting_projector(self, Tmax):
        lm = self.qwen_vl_interface.model.language_model

        if not hasattr(lm, "halting_projector"):
            return

        module = lm.halting_projector
        last_linear = None

        # ===== 1. 初始化权重（缩小scale，防止sigmoid饱和）=====
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # 🔥关键：缩小
                nn.init.zeros_(m.bias)
                last_linear = m

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # ===== 2. 初始化bias（step-aware + 随机）=====
        if last_linear is not None:
            # 参数可以调
            base = -2.0        # 初始更偏向继续
            step_scale = 0.4   # 每步增加停止概率
            noise_std = 0.1    # 小随机扰动

            # 如果 bias 是标量（shared head）
            if last_linear.bias.numel() == 1:
                b = base + (Tmax / 2) * step_scale
                b += torch.randn(1).item() * noise_std
                nn.init.constant_(last_linear.bias, b)

            # 如果 bias 是 per-step（少见但更强）
            else:
                bias = []
                for t in range(Tmax):
                    b = base + step_scale * t
                    b += torch.randn(1).item() * noise_std
                    bias.append(b)

                last_linear.bias.data = torch.tensor(
                    bias, device=last_linear.bias.device, dtype=last_linear.bias.dtype
                )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            loop_token = torch.full((input_ids.shape[0], 1), self.loop_token_id, dtype=input_ids.dtype, device=input_ids.device)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_token], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)
    
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            student_hidden_states = torch.stack(qwenvl_outputs.hidden_states,dim=1)
            halting_scores = torch.stack(qwenvl_outputs.halting_scores, dim=1).squeeze(-1)
            remaining_scores = torch.stack(qwenvl_outputs.remaining_scores, dim=1).squeeze(-1)

        # mask = remaining_scores < self.qwen_vl_interface.stop_threshold     # （B,L）
        # first_indices = mask.int().argmax(dim=1)
        # has_any = mask.any(dim=1)                          # [B]
        # first_indices = torch.where(
        #     has_any, 
        #     first_indices, 
        #     torch.full_like(first_indices, self.qwen_vl_interface.num_loop-1)
        # )
        # halting_scores *= (
        #     torch.arange(halting_scores.shape[1], device=halting_scores.device)
        #     <= first_indices.unsqueeze(1)
        # )

        # TODO： get halting layer output for last_hidden
        tmp_hidden_states = student_hidden_states[:,5::self.qwen_vl_interface.num_preserved_layers]
        assert tmp_hidden_states.shape[1] == self.qwen_vl_interface.num_loop

        print(f"{remaining_scores=}")

        # remaining score entropy loss
        p = remaining_scores.clamp(1e-6, 1.0)
        entropy = -(p * torch.log(p)).sum(dim=1)
        entropy_loss = -entropy.mean() * self.halting_entropy_weight


        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            
            # Process each layer's hidden state through action head and compute weighted loss
            actions = torch.tensor(
                np.array(actions), device=tmp_hidden_states.device, dtype=torch.float32
            )
            actions_target = actions[:, -(self.future_action_window_size+1):, :]
            
            # Compute loss for each layer's output
            action_losses = []
            for i in range(tmp_hidden_states.shape[1]):
                layer_hidden = tmp_hidden_states[:, i, :, :]  # [B, S, H]
                action_queries = self._gather_action_token_embeddings(layer_hidden, input_ids, action_token_id=self.action_token_id)
                pred_actions = self.action_model.predict_action(action_queries)
                # Use reduction='none' to keep batch dimension (already set in constructor)
                layer_loss = self.l1_loss(pred_actions, actions_target)
                # Take mean over action dimensions but keep batch dimension
                layer_loss = layer_loss.mean(dim=[1, 2])  # [B]
                action_losses.append(layer_loss)
            
            # Stack losses and apply remaining_scores weighting
            action_losses = torch.stack(action_losses, dim=1)  # [B, num_loop]
            action_loss = (action_losses * remaining_scores).sum(dim=1).mean()
            
            # Loop effectiveness loss
            loop_effectiveness_loss = 0.0
            if self.use_loop_effectiveness_loss and action_losses.shape[1] > 1:
                # Compute improvement: previous loss - current loss
                improvement = action_losses[:, :-1] - action_losses[:, 1:]  # [B, num_loop-1]
                improvement = improvement.detach()  # Detach to avoid gradient flow
                
                # Compute w_t = sigmoid(k * (I_t - gamma))
                k = self.loop_effectiveness_k
                gamma = self.loop_effectiveness_gamma
                w_t = torch.sigmoid(k * (improvement - gamma))  # [B, num_loop-1]
                
                # Compute target_stop = 1 - w_t
                target_stop = 1 - w_t  # [B, num_loop-1]
                
                # Compute cross entropy between remaining_scores and target_stop
                # Take the first (num_loop-1) elements of remaining_scores
                remaining_scores_truncated = remaining_scores[:, :action_losses.shape[1]-1]  # [B, num_loop-1]
                
                # Cross entropy loss
                loop_effectiveness_loss = F.binary_cross_entropy(
                    remaining_scores_truncated, 
                    target_stop, 
                    reduction='mean'
                )

        teacher_loss = 0.0
        if self.use_teacher_llm and self.training:
            self._forward_step_count += 1
            if self._forward_step_count % self._gradient_accumulation_steps == 0:
                self._optimizer_step += 1
                if self._optimizer_step >= self.teacher_loss_decay_steps:
                    self.teacher_loss_weight = 0.0

            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.inference_mode():
                    teacher_outputs = self.qwen_vl_interface.teacher_model(
                        **qwen_inputs,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teacher_hidden_states = torch.stack(teacher_outputs.hidden_states, dim=1)

                teacher_loss = self.mse_loss(student_hidden_states, teacher_hidden_states)
                weighted_teacher_loss = self.teacher_loss_weight * teacher_loss
                
            total_loss = action_loss + weighted_teacher_loss + entropy_loss + loop_effectiveness_loss

        else:
            total_loss = action_loss + entropy_loss + loop_effectiveness_loss

        return {
            "action_loss": action_loss,
            "teacher_loss": teacher_loss,
            "entropy_loss": entropy_loss,
            "loop_effectiveness_loss": loop_effectiveness_loss,
            "teacher_loss_weight": self.teacher_loss_weight,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        # 添加 loop token
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            loop_token = torch.full((input_ids.shape[0], 1), self.loop_token_id, dtype=input_ids.dtype, device=input_ids.device)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_token], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 处理 loop 相关的 hidden states
            student_hidden_states = torch.stack(qwenvl_outputs.hidden_states, dim=1)
            
            # 推理时只使用最后一个 hidden state
            tmp_hidden_states = student_hidden_states[:,5::self.qwen_vl_interface.num_preserved_layers]
            student_last_hidden = tmp_hidden_states[:, -1, :, :]  # [B, S, H]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(student_last_hidden, input_ids, action_token_id=self.action_token_id)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        action_token_id=None,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        action_queries = last_hidden.gather(dim=1, index=expanded_index)
        return action_queries


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/starvla_train_libero_loopoft.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    
    breakpoint()
    model = LoopOFT(config=cfg)
    print("✅ LoopOFT model created successfully")
