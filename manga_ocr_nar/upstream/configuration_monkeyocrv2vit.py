from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig
# Remove Qwen3Config import as it causes error on older transformers/python versions
# from transformers.models.qwen3 import Qwen3Config 
#from transformers import Qwen2_5_VLProcessor, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


class MonkeyOCRv2VisionConfig(PretrainedConfig):
    model_type = "MonkeyOCRv2VisionTransformer"

    def __init__(
        self,
        embed_dim: int = 1536,  # vision encoder embed size
        hidden_size: int = 1536,  # after merger hidden size
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        vision_attn_implementation="flash_attention_2",  # "eager","sdpa","flash_attention_2"
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,  # ve causal forward
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.vision_attn_implementation = vision_attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


# Commented out Processor definition to avoid dependencies on Qwen2_5_VLProcessor
# class MonkeyOCRv2Processor(Qwen2_5_VLProcessor):
#     attributes = ["image_processor", "tokenizer"]
#     def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
#         super().__init__(image_processor, tokenizer, chat_template=chat_template)


# AutoProcessor.register("MonkeyOCRv2VisionTransformer", MonkeyOCRv2Processor)
CONFIG_MAPPING.register("MonkeyOCRv2VisionTransformer", MonkeyOCRv2VisionConfig)
