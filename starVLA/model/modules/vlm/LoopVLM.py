import torch
import torch.nn as nn
from typing import Optional, List
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file

from starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel
from starVLA.model.modules.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig


class _LoopVLM_Interface(nn.Module):
    """
    LoopVLM VLM Interface.
    
    实现说明:
    - 加载 Vision-Language Model
    - 提供统一的 forward / generate / build_qwenvl_inputs 接口
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        初始化 LoopVLM 接口。
        
        Args:
            config: OmegaConf 配置对象
        """
        super().__init__()

        # 从配置中获取模型路径
        vlm_config = config.framework.get("qwenvl", {})
        vlm_path = vlm_config.get("vlm_path", None)
        base_vlm = vlm_config.get("base_vlm", "path/to/your/LoopVLM")
        pretrained_weights_path = vlm_config.get("pretrained_weights_path", None)
        
        # 优先级: vlm_path > base_vlm
        model_id = vlm_path if vlm_path else base_vlm
        
        self.use_teacher_llm = config.framework.get("use_teacher_llm", False)
        
        # Loop 相关参数
        self.loop_token_id = vlm_config.get("loop_token_id", 140000)
        self.as_student = vlm_config.get("as_student", False)
        self.num_preserved_layers = vlm_config.get("num_preserved_layers", 24)
        self.num_loop = vlm_config.get("num_loop", 1)
        self.stop_threshold = vlm_config.get("stop_threshold", 0.5)
        
        self.model_config = Qwen3VLConfig.from_pretrained(model_id)
        self.model_config.use_teacher_llm = self.use_teacher_llm
        self.model_config.as_student = self.as_student
        self.model_config.num_preserved_layers = self.num_preserved_layers
        self.model_config.num_loop = self.num_loop
        self.model_config.stop_threshold = self.stop_threshold
        
        self.model = Qwen3VLForConditionalGeneration(self.model_config)
        
        pretrained_weights_path = vlm_config.get("pretrained_weights_path", None)
        if pretrained_weights_path:
            state_dict = load_file(pretrained_weights_path)
            self.model.load_state_dict(state_dict, strict=False)
        
        # 设置 hidden_size 兼容属性
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        
        # 保存 config（必须在调用 _init_teacher_model 之前）
        self.config = config
        
        # Teacher Model: 只在训练模式时初始化
        # 根据 config 框架配置中的 is_training 标志决定是否初始化
        self.is_training_mode = config.framework.get("is_training", True)
        if self.use_teacher_llm and self.is_training_mode:
            self._init_teacher_model()
        else:
            self.teacher_model = None
        
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _init_teacher_model(self):
        """初始化 teacher_model（用于 teacher forcing）。"""
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            return
        
        self.teacher_model = None
        
        self.teacher_model = Qwen3VLForConditionalGeneration(self.model_config)
        self.teacher_model.requires_grad_(False)
        
        # 加载权重
        pretrained_weights_path = self.config.framework.get("qwenvl", {}).get("pretrained_weights_path", None)
        if pretrained_weights_path:
            state_dict = load_file(pretrained_weights_path)
            missing, unex = self.teacher_model.load_state_dict(state_dict, strict=False)        
        print(f"✅ Initialized teacher_model (frozen) for teacher forcing")
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播。
        
        Args:
            input_ids: token ids [B, T]
            attention_mask: attention mask [B, T]
            pixel_values: 图像张量
            labels: 训练标签
            output_hidden_states: 必须为 True
            return_dict: 返回 dict 格式
            
        Returns:
            CausalLMOutputWithPast 包含 hidden_states
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return outputs


    def generate(
        self,
        **kwargs,
    ):
        """
        自回归生成接口。
        
        Args:
            **kwargs: 传给 model.generate()
            
        Returns:
            生成结果
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images: List, instructions: List[str], solutions=None, **kwargs):
        """
        构建模型输入（参考 QWen3.py 实现）。
        
        Args:
            images: List[PIL.Image] 或 List[List[PIL.Image]]，多视角图像
            instructions: List[str]，文本指令
            solutions: 可选的解决方案文本（用于训练）
            
        Returns:
            dict: 包含 input_ids, attention_mask, pixel_values 等
        """
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        
        for imgs, instruction in zip(images, instructions):
            if isinstance(imgs, list):
                content = [{"type": "image", "image": img} for img in imgs]
            else:
                content = [{"type": "image", "image": imgs}]

            if hasattr(self.config, 'datasets') and hasattr(self.config.datasets, 'vla_data'):
                cot_prompt = getattr(self.config.datasets.vla_data, 'CoT_prompt', None)
                if cot_prompt:
                    prompt = cot_prompt.replace("{instruction}", instruction)
                else:
                    prompt = instruction
            else:
                prompt = instruction
            
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        return batch_inputs.to(device)

