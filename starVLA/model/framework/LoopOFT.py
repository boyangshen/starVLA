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

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.use_teacher_llm = self.config.framework.use_teacher_llm
        self.initial_teacher_loss_weight = config.framework.get("teacher_loss_weight", 0.5)
        self.teacher_loss_weight = self.initial_teacher_loss_weight
        self.teacher_loss_decay_steps = config.framework.get("teacher_loss_decay_steps", 1000)
        self._optimizer_step = 0
        self._gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps", 1)
        self._forward_step_count = 0

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
            qwen_inputs["input_ids"] = torch.cat([loop_token, input_ids], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([ones, attention_mask], dim=1)
    
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            student_hidden_states = qwenvl_outputs.hidden_states

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
            pred_actions = self.action_model.predict_action(action_queries)

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size+1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        teacher_loss = 0.0
        if self.use_teacher_llm and self.training:
            self._forward_step_count += 1
            if self._forward_step_count % self._gradient_accumulation_steps == 0:
                self._optimizer_step += 1
                self.teacher_loss_weight = max(
                    0.0, 
                    self.initial_teacher_loss_weight * (1 - self._optimizer_step / self.teacher_loss_decay_steps)
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.inference_mode():
                    teacher_outputs = self.qwen_vl_interface.teacher_model(
                        **qwen_inputs,
                        output_attentions=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    teacher_hidden_states = teacher_outputs.hidden_states
                
                teacher_loss = self.mse_loss(student_hidden_states, teacher_hidden_states.detach())
                total_loss = action_loss + self.teacher_loss_weight * teacher_loss
        else:
            total_loss = action_loss

            return {
                "action_loss": action_loss,
                "teacher_loss": teacher_loss,
                "teacher_loss_weight": self.teacher_loss_weight,
                "total_loss": total_loss,
            }

        return {"action_loss": action_loss}



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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
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
