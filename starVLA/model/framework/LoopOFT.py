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

from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from scipy.stats import gaussian_kde
from scipy.integrate import quad

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
        
        # 3 loop tokens with fixed IDs
        self.loop_token_ids = [140000, 140001, 140002]

        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()
        
        self._init_halting_projector()

        self.use_teacher_llm = self.config.framework.use_teacher_llm
        self.halting_entropy_weight = self.config.framework.get("halting_entropy_weight", 0.0)
        self.initial_teacher_loss_weight = config.framework.get("teacher_loss_weight", 0.5)
        self.teacher_loss_weight = self.initial_teacher_loss_weight
        self.teacher_loss_decay_steps = config.framework.get("teacher_loss_decay_steps", 1000)
        # Entropy loss parameters
        self.use_entropy_loss = config.framework.get("use_entropy_loss", True)
        self.halting_entropy_weight = self.config.framework.get("halting_entropy_weight", 0.0)
        # Loop effectiveness loss parameters
        self.use_loop_effectiveness_loss = config.framework.get("use_loop_effectiveness_loss", False)
        self.loop_effectiveness_weight = config.framework.get("loop_effectiveness_weight", 1.0)


        # self.kde_distribution_transformer = KDEDistributionTransformer(
        #     num_bins=config.framework.qwenvl.get("num_loop", 6)
        # )
        # Cosine similarity loss parameter
        self.use_cosine_loss = config.framework.get("use_cosine_loss", False)
        self.cosine_loss_weight = config.framework.get("cosine_loss_weight", 0.2)


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
        
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            loop_tokens = torch.tensor([self.loop_token_ids], dtype=input_ids.dtype, device=input_ids.device).expand(input_ids.shape[0], -1)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_tokens], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 3), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)
            action_token_mask = (qwen_inputs["input_ids"] == self.action_token_id)
            qwen_inputs["action_token_mask"] = action_token_mask

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

        # TODO： get halting layer output for last_hidden
        tmp_hidden_states = student_hidden_states[:,5::self.qwen_vl_interface.num_preserved_layers]
        assert tmp_hidden_states.shape[1] == self.qwen_vl_interface.num_loop

        print(f"{remaining_scores=}")

        # remaining score entropy loss
        entropy_loss = 0.0
        if self.use_entropy_loss:
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
            
            # Stack losses and apply remaining_scores weighting (remaining_scores is already normalized [B, num_loop])
            action_losses = torch.stack(action_losses, dim=1)  # [B, num_loop]
            # action_loss = (action_losses * remaining_scores).sum(dim=1).mean()

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

                # self.kde_distribution_transformer.device = goodness.device
                # target_remaining_scores = self.kde_distribution_transformer.get_kde_density(goodness)
                # target_remaining_scores = torch.from_numpy(target_remaining_scores).to(goodness.device, dtype=goodness.dtype)
                
                # Compute KL divergence between target_remaining_scores and remaining_scores
                # KL(P||Q) = sum(P * log(P/Q))
                p = target_remaining_scores.clamp(1e-8, 1.0)  # target distribution
                q = remaining_scores.clamp(1e-8, 1.0)         # predicted distribution
                kl_div = (p * torch.log(p / q)).sum(dim=1).mean()  # [B, num_loop] -> scalar
                loop_effectiveness_loss = kl_div * self.loop_effectiveness_weight

            # cosine similarity loss
            cosine_loss = 0.0
            if self.use_cosine_loss and hasattr(qwenvl_outputs, 'latent_logits_list') and qwenvl_outputs.latent_logits_list:
                latent = torch.stack(qwenvl_outputs.latent_logits_list, dim=1)  # [B, L, D]
                if latent.shape[1] > 1:
                    # Normalize latent vectors
                    latent_norm = F.normalize(latent, dim=-1)  # [B, L, D]
                    
                    # Compute pairwise cosine similarity using matrix multiplication
                    # similarity matrix shape: [B, L, L]
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
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        # 添加 3 个 loop token
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is not None:
            loop_tokens = torch.tensor([self.loop_token_ids], dtype=input_ids.dtype, device=input_ids.device).expand(input_ids.shape[0], -1)
            qwen_inputs["input_ids"] = torch.cat([input_ids, loop_tokens], dim=1)
            if "attention_mask" in qwen_inputs:
                attention_mask = qwen_inputs["attention_mask"]
                ones = torch.ones((attention_mask.shape[0], 3), dtype=attention_mask.dtype, device=attention_mask.device)
                qwen_inputs["attention_mask"] = torch.cat([attention_mask, ones], dim=1)

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
            
            # 处理 loop 相关的 hidden states
            student_hidden_states = torch.stack(qwenvl_outputs.hidden_states, dim=1)
            logger.info(f"student_hidden_states shape: {student_hidden_states.shape}")

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


class KDEDistributionTransformer:
    """
    KDE 分布转换器
    
    每个 batch 独立拟合 KDE，然后离散化为概率分布
    输入形状: [batch, loop_id]
    输出形状: [batch, loop_id] (保持不变)
    """
    
    def __init__(
        self,
        num_bins: int = 6,
        bandwidth: float = 0.5,
        bin_method: str = 'std',
        epsilon: float = 1e-10,
    ):
        """
        Args:
            num_bins: 离散化的区间数量
            bandwidth: KDE 带宽
            bin_method: bin 选择方法 ('std', 'range', 'percentile', 'iqr', 'adaptive')
            epsilon: 数值稳定性常数
        """
        self.num_bins = num_bins
        self.bandwidth = bandwidth
        self.bin_method = bin_method
        self.epsilon = epsilon
        
        # 每个 batch 的 bins 和 target 分布
        self.row_bins = None
        self.row_target_dist = None
        
        # 每个 batch 的 KDE 对象
        self.row_kdes = None
    
    def _select_bins(self, data: np.ndarray) -> np.ndarray:
        """根据数据自适应选择 bins"""
        data = np.array(data)
        data_min = data.min()
        data_max = data.max()
        data_mean = data.mean()
        data_std = data.std()
        data_range = data_max - data_min
        
        if self.bin_method == 'std':
            std_factor = 2.0
            lower = data_mean - data_std * std_factor
            upper = data_mean + data_std * std_factor
        elif self.bin_method == 'range':
            padding_factor = 0.2
            lower = data_min - data_range * padding_factor
            upper = data_max + data_range * padding_factor
        elif self.bin_method == 'percentile':
            lower = np.percentile(data, 5)
            upper = np.percentile(data, 95)
            margin = (upper - lower) * 0.1
            lower -= margin
            upper += margin
        elif self.bin_method == 'iqr':
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            iqr_factor = 1.5
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
        elif self.bin_method == 'adaptive':
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            bins = np.percentile(data, percentiles)
            data_range = bins[-1] - bins[0]
            bins[0] -= data_range * 0.1
            bins[-1] += data_range * 0.1
            return bins
        else:
            raise ValueError(f"Unknown bin_method: {self.bin_method}")
        
        return np.linspace(lower, upper, self.num_bins + 1)
    
    def _kde_to_discrete_prob(
        self,
        kde: gaussian_kde,
        bins: np.ndarray
    ) -> np.ndarray:
        """将 KDE 连续分布离散化为概率分布"""
        probabilities = []
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            
            try:
                prob, _ = quad(lambda x: kde.evaluate(x)[0], lower, upper)
            except:
                mid = (lower + upper) / 2
                width = upper - lower
                prob = kde.evaluate(mid)[0] * width
            
            probabilities.append(prob)
        
        probabilities = np.array(probabilities)
        
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            probabilities = np.ones(self.num_bins) / self.num_bins
        
        return probabilities
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'KDEDistributionTransformer':
        """
        拟合 KDE 并计算 target 分布
        
        每个 batch 独立拟合 KDE
        
        Args:
            data: 输入数据，形状 [batch, loop_id]
        
        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        batch_size = data.shape[0]
        self.row_bins = []
        self.row_target_dist = []
        self.row_kdes = []
        
        for i in range(batch_size):
            row_data = data[i]  # 形状 [loop_id]
            bins = self._select_bins(row_data)
            kde = gaussian_kde(row_data, bw_method=self.bandwidth)
            target_dist = self._kde_to_discrete_prob(kde, bins)
            
            self.row_bins.append(bins)
            self.row_target_dist.append(target_dist)
            self.row_kdes.append(kde)
        
        return self
    
    def transform(
        self,
        data: np.ndarray,
        return_probs: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        将输入数据转换为概率分布
        
        Args:
            data: 输入数据，形状 [batch, loop_id]
            return_probs: 是否同时返回概率值
        
        Returns:
            概率分布，形状 [batch, loop_id]
        """
        original_shape = data.shape
        batch_size = original_shape[0]
        loop_id = original_shape[1]
        
        if self.row_bins is None:
            self.fit(data)
        
        probs = np.zeros(original_shape)
        bin_indices = np.zeros(original_shape, dtype=int)
        
        for i in range(batch_size):
            row_data = data[i]
            bins = self.row_bins[i]
            target_dist = self.row_target_dist[i]
            
            for j in range(loop_id):
                val = row_data[j]
                bin_idx = np.searchsorted(bins, val, side='right') - 1
                bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
                bin_indices[i, j] = bin_idx
                probs[i, j] = target_dist[bin_idx]
            
            # 行内归一化
            row_sum = probs[i].sum()
            probs[i] = probs[i] / (row_sum + self.epsilon)
        
        if return_probs:
            return probs, bin_indices.astype(float)
        else:
            return probs
    
    def fit_transform(
        self,
        data: np.ndarray,
        return_probs: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """拟合并转换"""
        self.fit(data)
        return self.transform(data, return_probs)
    
    def get_bins(self) -> List[np.ndarray]:
        """获取每个 batch 的 bins"""
        return self.row_bins
    
    def get_target_distribution(self) -> List[np.ndarray]:
        """获取每个 batch 的 target 分布"""
        return self.row_target_dist
    
    def get_kde_density(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        获取 KDE 概率密度（不做离散化）
        
        对每个 batch，使用对应的 KDE 对象计算概率密度
        
        Args:
            data: 输入数据，形状 [batch, loop_id]
        
        Returns:
            概率密度，形状 [batch, loop_id]
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if self.row_kdes is None:
            self.fit(data)
        
        original_shape = data.shape
        batch_size = original_shape[0]
        loop_id = original_shape[1]
        
        densities = np.zeros(original_shape)
        
        for i in range(batch_size):
            row_data = data[i]
            kde = self.row_kdes[i]
            
            # 计算每个数据点的概率密度
            for j in range(loop_id):
                val = row_data[j]
                densities[i, j] = kde.evaluate(val)[0]
        
        return densities
    
    def get_kde_density_grid(
        self,
        batch_idx: int,
        x_grid: np.ndarray
    ) -> np.ndarray:
        """
        获取指定 batch 的 KDE 概率密度网格
        
        Args:
            batch_idx: batch 索引
            x_grid: 评估点网格，形状 [n_points]
        
        Returns:
            概率密度，形状 [n_points]
        """
        if self.row_kdes is None:
            raise ValueError("请先调用 fit() 方法")
        
        if batch_idx >= len(self.row_kdes):
            raise ValueError(f"batch_idx {batch_idx} 超出范围")
        
        kde = self.row_kdes[batch_idx]
        densities = kde.evaluate(x_grid)
        
        return densities


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


