import torch
from transformers.models.mistral.modeling_mistral import MistralMLP, MistralSdpaAttention, MistralDecoderLayer, \
    MistralModel, MistralForCausalLM, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from model_utils import build_basis_collection, Coefficient
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ShareMistralMLP(MistralMLP):
    def __init__(self, config, share_gate, share_up, share_down, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.share_gate = share_gate
        self.share_up = share_up
        self.share_down = share_down
        if share_gate:
            self.gate_proj = Coefficient(self.intermediate_size, config.num_basis_gate)
        if share_up:
            self.up_proj = Coefficient(self.intermediate_size, config.num_basis_up)
        if share_down:
            self.down_proj = Coefficient(self.hidden_size, config.num_basis_down)

    def forward(self, hidden_state, **kwargs):
        if self.share_gate:
            gate = self.act_fn(self.gate_proj(kwargs['gate_basis'][str(self.layer_idx)](hidden_state)))
        else:
            gate = self.act_fn(self.gate_proj(hidden_state))

        if self.share_up:
            up = self.up_proj(kwargs['up_basis'][str(self.layer_idx)](hidden_state))
        else:
            up = self.up_proj(hidden_state)

        hidden_state = gate * up
        if self.share_down:
            down = self.down_proj(kwargs['down_basis'][str(self.layer_idx)](hidden_state))
        else:
            down = self.down_proj(hidden_state)
        return down


class ShareMistralSdpaAttention(MistralSdpaAttention):
    def __init__(self, config, layer_idx, share_k, share_q, share_v, share_o):
        super().__init__(config, layer_idx)
        self.share_k = share_k
        self.share_q = share_q
        self.share_v = share_v
        self.share_o = share_o
        if share_q:
            self.q_proj = Coefficient(self.num_heads * self.head_dim, config.num_basis_q)
        if share_k:
            self.k_proj = Coefficient(self.num_key_value_heads * self.head_dim, config.num_basis_k)
        if share_v:
            self.v_proj = Coefficient(self.num_key_value_heads * self.head_dim, config.num_basis_v)
        if share_o:
            self.o_proj = Coefficient(self.hidden_size, config.num_basis_o)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
    ):
        if output_attentions:
            raise NotImplementedError

        bsz, q_len, _ = hidden_states.size()

        if self.share_k:
            key_states = self.k_proj(kwargs['k_basis'][str(self.layer_idx)](hidden_states))
        else:
            key_states = self.k_proj(hidden_states)

        if self.share_q:
            query_states = self.q_proj(kwargs['q_basis'][str(self.layer_idx)](hidden_states))
        else:
            query_states = self.q_proj(hidden_states)

        if self.share_v:
            value_states = self.v_proj(kwargs['v_basis'][str(self.layer_idx)](hidden_states))
        else:
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        if self.share_o:
            attn_output = kwargs['o_basis'][str(self.layer_idx)](attn_output)
            attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class ShareMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        super().__init__(config, layer_idx)
        share_k = self._in_group(config.k_groups, layer_idx)
        share_q = self._in_group(config.q_groups, layer_idx)
        share_v = self._in_group(config.v_groups, layer_idx)
        share_o = self._in_group(config.o_groups, layer_idx)
        share_up = self._in_group(config.up_groups, layer_idx)
        share_down = self._in_group(config.down_groups, layer_idx)
        share_gate = self._in_group(config.gate_groups, layer_idx)
        self.self_attn = ShareMistralSdpaAttention(config, layer_idx, share_k, share_q, share_v, share_o)
        self.mlp = ShareMistralMLP(config, share_gate, share_up, share_down, layer_idx)

    @staticmethod
    def _in_group(groups, layer_idx):
        return any(layer_idx in group for group in groups)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ShareMistralModel(MistralModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "num_basis_k"):
            self.k_basis = build_basis_collection(config.k_groups, config.num_basis_k, config.hidden_size)
        else:
            self.k_basis = None
        if hasattr(config, "num_basis_q"):
            self.q_basis = build_basis_collection(config.q_groups, config.num_basis_q, config.hidden_size)
        else:
            self.q_basis = None
        if hasattr(config, "num_basis_v"):
            self.v_basis = build_basis_collection(config.v_groups, config.num_basis_v, config.hidden_size)
        else:
            self.v_basis = None
        if hasattr(config, "num_basis_o"):
            self.o_basis = build_basis_collection(config.o_groups, config.num_basis_o, config.hidden_size)
        else:
            self.o_basis = None
        if hasattr(config, "num_basis_gate"):
            self.gate_basis = build_basis_collection(config.gate_groups, config.num_basis_gate, config.hidden_size)
        else:
            self.gate_basis = None
        if hasattr(config, "num_basis_up"):
            self.up_basis = build_basis_collection(config.up_groups, config.num_basis_up, config.hidden_size)
        else:
            self.up_basis = None
        if hasattr(config, "num_basis_down"):
            self.down_basis = build_basis_collection(config.down_groups, config.num_basis_down,
                                                     config.intermediate_size)
        else:
            self.down_basis = None

        self.layers = torch.nn.ModuleList(
            [ShareMistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
                # layer_outputs = self._gradient_checkpointing_func(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     causal_mask,
                #     position_ids,
                #     past_key_values,
                #     output_attentions,
                #     use_cache,
                #     cache_position,
                # )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    k_basis=self.k_basis,
                    q_basis=self.q_basis,
                    v_basis=self.v_basis,
                    o_basis=self.o_basis,
                    gate_basis=self.gate_basis,
                    up_basis=self.up_basis,
                    down_basis=self.down_basis,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShareMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ShareMistralModel(config)
        self.config = config
