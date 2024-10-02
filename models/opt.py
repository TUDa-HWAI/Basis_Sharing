import torch
import torch.nn as nn
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTDecoder, OPTModel, OPTForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import logging
from model_utils import build_basis_collection, Coefficient

logger = logging.get_logger(__name__)


class ShareOPTAttention(OPTAttention):
    def __init__(self, config, layer_idx, share_k, share_q, share_v, share_o):
        super().__init__(config)
        self.share_k = share_k
        self.share_q = share_q
        self.share_v = share_v
        self.share_o = share_o
        self.layer_idx = layer_idx

        if share_k:
            self.k_proj = Coefficient(self.embed_dim, config.num_basis_k, bias=self.enable_bias)
        if share_q:
            self.q_proj = Coefficient(self.embed_dim, config.num_basis_q, bias=self.enable_bias)
        if share_v:
            self.v_proj = Coefficient(self.embed_dim, config.num_basis_v, bias=self.enable_bias)

        if share_o:
            self.out_proj = Coefficient(self.embed_dim, config.num_basis_o, bias=self.enable_bias)

    def forward(
            self,
            hidden_states,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions: bool = False,
            **kwargs,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if self.share_q:
            query_states = self.q_proj(kwargs["q_basis"][str(self.layer_idx)](hidden_states)) * self.scaling

        # get query proj
        else:
            query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            if self.share_k:
                key_states = self._shape(self.k_proj(kwargs['k_basis'][str(self.layer_idx)](hidden_states)), -1, bsz)
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            if self.share_v:
                value_states = self._shape(self.v_proj(kwargs['v_basis'][str(self.layer_idx)](hidden_states)), -1, bsz)
            else:
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            if self.share_k:
                key_states = self._shape(self.k_proj(kwargs['k_basis'][str(self.layer_idx)](hidden_states)), -1, bsz)
            else:
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            if self.share_v:
                value_states = self._shape(self.v_proj(kwargs['v_basis'][str(self.layer_idx)](hidden_states)), -1, bsz)
            else:
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        if self.share_o:
            attn_output = kwargs['o_basis'][str(self.layer_idx)](attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class ShareOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        # share_attn = self._in_group(config.attn_groups, layer_idx)
        share_k = self._in_group(config.k_groups, layer_idx)
        share_q = self._in_group(config.q_groups, layer_idx)
        share_v = self._in_group(config.v_groups, layer_idx)
        share_o = self._in_group(config.o_groups, layer_idx)
        self.share_up = self._in_group(config.up_groups, layer_idx)
        self.share_down = self._in_group(config.down_groups, layer_idx)
        self.layer_idx = layer_idx

        self.self_attn = ShareOPTAttention(config, layer_idx, share_k, share_q, share_v, share_o)

        if self.share_up:
            self.fc1 = Coefficient(config.ffn_dim, config.num_basis_up, bias=config.enable_bias)

        if self.share_down:
            self.fc2 = Coefficient(self.embed_dim, config.num_basis_down, bias=config.enable_bias)

    @staticmethod
    def _in_group(groups, layer_idx):
        return any(layer_idx in group for group in groups)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.share_up:
            hidden_states = kwargs['up_basis'][str(self.layer_idx)](hidden_states)
        hidden_states = self.fc1(hidden_states)

        hidden_states = self.activation_fn(hidden_states)

        if self.share_down:
            hidden_states = kwargs['down_basis'][str(self.layer_idx)](hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ShareOPTDecoder(OPTDecoder):
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

        # if hasattr(config, "num_basis_attn"):
        #     self.attn_basis = build_basis_collection(config.attn_groups, config.num_basis_attn, config.hidden_size)
        # else:
        #     self.attn_basis = None
        if hasattr(config, "num_basis_o"):
            self.o_basis = build_basis_collection(config.o_groups, config.num_basis_o, config.hidden_size)
        else:
            self.o_basis = None
        if hasattr(config, "num_basis_up"):
            self.up_basis = build_basis_collection(config.up_groups, config.num_basis_up, config.hidden_size)
        else:
            self.up_basis = None
        if hasattr(config, "num_basis_down"):
            self.down_basis = build_basis_collection(config.down_groups, config.num_basis_down,
                                                     config.ffn_dim)
        else:
            self.down_basis = None

        self.layers = nn.ModuleList([ShareOPTDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
                # layer_outputs = self._gradient_checkpointing_func(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     causal_attention_mask,
                #     head_mask[idx] if head_mask is not None else None,
                #     None,
                #     output_attentions,
                #     use_cache,
                # )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    k_basis=self.k_basis,
                    q_basis=self.q_basis,
                    v_basis=self.v_basis,
                    o_basis=self.o_basis,
                    up_basis=self.up_basis,
                    down_basis=self.down_basis,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ShareOPTModel(OPTModel):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = ShareOPTDecoder(config)


class ShareOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ShareOPTModel(config)
        self.config = config
