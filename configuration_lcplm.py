# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0
from typing import Optional, Union
from transformers import PretrainedConfig


class LcPlmConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality."""
    model_type = "lc_plm"

    def __init__(
            self,
            # From original MambaConfig
            d_model: int = 1536,
            d_intermediate: int = 0,
            n_layer: int = 48,
            vocab_size: int = 33,
            ssm_cfg: Optional[dict] = None,
            attn_layer_idx = None,
            attn_cfg = None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = True,
            residual_in_fp32: bool = True,
            fused_add_norm: bool = True,
            pad_vocab_size_multiple: int = 8,
            layer: str = "Mamba2",

            # Used in init_weights
            initializer_cfg: Optional[dict] = None,

            # BiMamba-specific params
            bidirectional: bool = True,
            bidirectional_strategy: Union[str, None] = "add",
            bidirectional_weight_tie: bool = True,
            tie_embeddings: bool = True,
            pad_token_id: int = -100,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.attn_layer_idx = attn_layer_idx
        self.attn_cfg = attn_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.layer = layer
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.tie_embeddings = tie_embeddings
        self.pad_token_id = pad_token_id
