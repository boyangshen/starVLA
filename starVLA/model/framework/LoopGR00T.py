# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
LoopGR00T Framework

A VLA framework that combines LoopOFT's loop mechanism with GR00T's flow-matching action head:
  - Student: Full VLM (Vision + Language) with loop mechanism
  - Teacher: Text-only LLM (frozen)
  - Action Head: Flow-matching for continuous action prediction

Key Features:
  - Qwen3-VL as backbone
  - Teacher Forcing via hidden state matching
  - Flow-matching for action prediction
  - Loop mechanism for iterative refinement
  
"""

from tqdm import tqdm
from typing import List, Optional, Tuple, Union
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
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("LoopGR00T")
class LoopGR00T(baseframework):
    """
    LoopGR00T: VLA with Loop mechanism and Flow-matching action head.

    Components:
      - Student VLM: Full Vision-Language Model with loop mechanism
      - Teacher LLM: Text-only LLM (frozen)
      - Action Head: Flow-matching for continuous action prediction

    Training:
      - Student predicts actions via VLM + Flow-matching Action Head
      - Student matches Teacher's hidden states (MSE loss)
      - Final loss = action_loss + teacher_loss * weight

    Inference:
      - Only use Student VLM + Flow-matching Action Head
      - Use loop mechanism for iterative refinement
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        
        # Align dimensions for action model
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # Action token configuration
        self.action_token = "🔍"
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        
        # 3 loop tokens with fixed IDs
        self.loop_token_ids = [140000, 140001, 140002]

        self.mse_loss = nn.MSELoss()
        
        self._init_halting_projector()

        self.use_teacher_llm = self.config.framework.use_teacher_llm
        self.initial_teacher_loss_weight = config.framework.get("teacher_loss_weight", 0.5)
        self.teacher_loss_weight = self.initial_teacher_loss_weight
        self.teacher_loss_decay_steps = config.framework.get("teacher_loss_decay_steps", 1000)
        # Entropy loss parameters
        self.use_entropy_loss = config.framework.get("use_entropy_loss", True)
        self.halting_entropy_weight = self.config.framework.get("halting_entropy_weight", 0.0)
        # Loop effectiveness loss parameters
        self.use_loop_effectiveness_loss = config.framework.get("use_loop_effectiveness_loss", False)
        self.loop_effectiveness_weight = config.framework.get("loop_effectiveness_weight", 1.0)

        # Cosine similarity loss parameter
        self.use_cosine_loss = config.framework.get("use_cosine_loss", False)
        self.cosine_loss_weight = config.framework.get("cosine_loss_weight", 0.2)

        # Loss update counter parameters
        self.max_loss_updates = config.framework.get("max_loss_updates", 10000)
        self._loss_update_count = 0

        self._optimizer_step = 0
        self._gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps", 1)
        self._forward_step_count = 0

    def _init_halting_projector(self):
        lm = self.qwen_vl_interface.model.language_model

        if not hasattr(lm, "halting_projector"):
            return

        module = lm.halting_projector

        # ===== 1. 随机初始化 =====
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化，产生正负权重
                nn.init.xavier_uniform_(m.weight)
                # 偏置初始化为小的随机值
                nn.init.normal_(m.bias, mean=0.0, std=0.01)

            elif isinstance(m, nn.LayerNorm):
                # LayerNorm权重初始化为1，保证输出范围
                nn.init.ones_(m.weight)
                # 偏置初始化为0
                nn.init.zeros_(m.bias)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        
        # Add action tokens to instructions
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            # Add loop tokens
            loop_tokens = torch.tensor([self.loop_token_ids], dtype=input_ids.dtype, device=input_ids.device).expand(input_ids.shape[0], -1)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_tokens], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 3), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)
            # Create action token mask
            action_token_mask = (qwen_inputs["input_ids"] == self.action_token_id)
            qwen_inputs["action_token_mask"] = action_token_mask

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            student_hidden_states = torch.stack(qwenvl_outputs.hidden_states, dim=1)
            halting_scores = torch.stack(qwenvl_outputs.halting_scores, dim=1).squeeze(-1)
            remaining_scores = torch.stack(qwenvl_outputs.remaining_scores, dim=1).squeeze(-1)

        # Get loop hidden states
        tmp_hidden_states = student_hidden_states[:, 5::self.qwen_vl_interface.num_preserved_layers]
        assert tmp_hidden_states.shape[1] == self.qwen_vl_interface.num_loop

        # Update loss update counter
        self._forward_step_count += 1
        if self._forward_step_count % self._gradient_accumulation_steps == 0:
            self._loss_update_count += 1

        # Check if max loss updates reached
        loss_updates_reached = self._loss_update_count >= self.max_loss_updates

        # remaining score entropy loss
        entropy_loss = 0.0
        if self.use_entropy_loss and not loss_updates_reached:
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
            
            # Process state if present
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=tmp_hidden_states.device, dtype=torch.float32
                )
            
            # Compute loss for each layer's output
            action_losses = []
            for i in range(tmp_hidden_states.shape[1]):
                layer_hidden = tmp_hidden_states[:, i, :, :]  # [B, S, H]
                action_queries = self._gather_action_token_embeddings(layer_hidden, input_ids, action_token_id=self.action_token_id)
                
                # Compute action loss using flow matching
                state_repeated = state_tensor.repeat(1, 1, 1) if state_tensor is not None else None
                layer_loss = self.action_model(action_queries, actions_target, state_repeated)
                action_losses.append(layer_loss)
            
            # Stack losses and compute mean
            action_losses = torch.stack(action_losses, dim=1)  # [B, num_loop]
            action_loss = action_losses.mean()

            # Loop effectiveness loss
            loop_effectiveness_loss = 0.0
            if self.use_loop_effectiveness_loss and action_losses.shape[1] > 1:

                # goodness = -action_losses, then normalize with z-score (per sample)
                goodness = -action_losses.clone().detach()
                mean_val = goodness.mean(dim=1, keepdim=True)
                std_val = goodness.std(dim=1, keepdim=True)
                goodness = (goodness - mean_val) / (std_val + 1e-8)  # z-score normalization per sample

                target_remaining_scores = F.softmax(goodness / 0.5, dim=1)
                
                # Compute KL divergence between target_remaining_scores and remaining_scores
                p = target_remaining_scores.clamp(1e-8, 1.0)  # target distribution
                q = remaining_scores.clamp(1e-8, 1.0)         # predicted distribution
                kl_div = (p * torch.log(p / q)).sum(dim=1).mean()  # [B, num_loop] -> scalar
                loop_effectiveness_loss = kl_div * self.loop_effectiveness_weight

            # cosine similarity loss
            cosine_loss = 0.0
            if self.use_cosine_loss and not loss_updates_reached and hasattr(qwenvl_outputs, 'latent_logits_list') and qwenvl_outputs.latent_logits_list:
                latent = torch.stack(qwenvl_outputs.latent_logits_list, dim=1)  # [B, L, D]
                if latent.shape[1] > 1:
                    # Normalize latent vectors
                    latent_norm = F.normalize(latent, dim=-1)  # [B, L, D]
                    
                    # Compute pairwise cosine similarity using matrix multiplication
                    similarity_matrix = torch.matmul(latent_norm, latent_norm.transpose(1, 2))
                    
                    # Create a mask to exclude diagonal (self-similarity) and lower triangle (duplicates)
                    L = latent.shape[1]
                    mask = torch.triu(torch.ones(L, L, device=latent.device), diagonal=1).bool()
                    
                    # Apply mask and compute mean similarity
                    cosine_sim = similarity_matrix[:, mask]  # [B, L*(L-1)/2]
                    # Compute loss (encourage diversity, discourage high similarity)
                    cosine_loss = cosine_sim.mean()

        teacher_loss = 0.0
        if self.use_teacher_llm and self.training:
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
                
            total_loss = action_loss + weighted_teacher_loss + entropy_loss + loop_effectiveness_loss + cosine_loss

        else:
            total_loss = action_loss + entropy_loss + loop_effectiveness_loss + cosine_loss * self.cosine_loss_weight

        return {
            "action_loss": action_loss,
            "teacher_loss": teacher_loss,
            "entropy_loss": entropy_loss,
            "loop_effectiveness_loss": loop_effectiveness_loss,
            "cosine_loss": cosine_loss,
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
        state = [example["state"] for example in examples] if "state" in examples[0] else None
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Add action tokens to instructions
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        # Add loop tokens
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            loop_tokens = torch.tensor([self.loop_token_ids], dtype=input_ids.dtype, device=input_ids.device).expand(input_ids.shape[0], -1)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_tokens], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 3), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)

            # Create action token mask
            action_token_mask = torch.zeros_like(qwen_inputs["input_ids"], dtype=torch.bool)
            if self.action_token_id is not None:
                action_token_mask = (qwen_inputs["input_ids"] == self.action_token_id)
            qwen_inputs["action_token_mask"] = action_token_mask
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Process loop-related hidden states
            student_hidden_states = torch.stack(qwenvl_outputs.hidden_states, dim=1)
            
            # Get remaining scores
            remaining_scores = torch.stack(qwenvl_outputs.remaining_scores, dim=1).squeeze(-1)
            
            # Ensure remaining_scores has at least 2 dimensions (B, L)
            if remaining_scores.dim() == 1:
                remaining_scores = remaining_scores.unsqueeze(0)
            
            # Get loop hidden states
            tmp_hidden_states = student_hidden_states[:, 5::self.qwen_vl_interface.num_preserved_layers]
            
            # Use fixed loop index for inference
            fixed_loop_index = 2  # Default to the last loop iteration
            
            # Select hidden state based on fixed index
            student_last_hidden = tmp_hidden_states[:, fixed_loop_index, :, :]  # [B, S, H]

        # Process state if present
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(student_last_hidden.device, dtype=student_last_hidden.dtype)
        
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(student_last_hidden, input_ids, action_token_id=self.action_token_id)
            pred_actions = self.action_model.predict_action(action_queries, state_tensor)

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
    parser.add_argument("--config_yaml", type=str, default="./examples/Robocasa_tabletop/train_files/qwen3oft2b_robocasa.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    
    # Add loop-specific configuration
    cfg.framework.use_teacher_llm = True
    cfg.framework.teacher_loss_weight = 0.5
    cfg.framework.teacher_loss_decay_steps = 1000
    cfg.framework.use_entropy_loss = True
    cfg.framework.halting_entropy_weight = 0.01
    cfg.framework.use_loop_effectiveness_loss = True
    cfg.framework.loop_effectiveness_weight = 1.0
    cfg.framework.use_cosine_loss = True
    cfg.framework.cosine_loss_weight = 0.2
    
    model = LoopGR00T(config=cfg)
    print("✅ LoopGR00T model created successfully")
