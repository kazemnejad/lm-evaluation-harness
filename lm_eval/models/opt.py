import os
import random
from typing import Optional, List, Union, Tuple

import accelerate
import huggingface_hub
import torch
import torch.nn.functional as F
import transformers
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import OPTForCausalLM, OPTModel, OPTConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from transformers.models.opt.modeling_opt import (
    OPTDecoder,
    OPTDecoderLayer,
    OPTLearnedPositionalEmbedding,
    make_positions,
)

from lm_eval.models.gpt2 import HFLM


class CustomOPTLearnedPositionalEmbedding(OPTLearnedPositionalEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 1):
        super(OPTLearnedPositionalEmbedding, self).__init__(
            num_embeddings, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings
        self.has_logged = False

    def forward(self, attention_mask: Tensor, positions: Optional[Tensor] = None):
        if positions is None:
            attention_mask = attention_mask.long()
            positions = make_positions(attention_mask, self.padding_idx)
        else:
            positions = torch.where(
                attention_mask.bool(), positions, self.padding_idx
            ).long()

        if not self.has_logged:
            print("-------> positions:")
            print(positions)
            self.has_logged = True

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class CustomOPTDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig):
        super(OPTDecoder, self).__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )

        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        if self.padding_idx is not None:
            num_embeddings = config.max_position_embeddings + 2

        self.embed_positions = CustomOPTLearnedPositionalEmbedding(
            num_embeddings, config.hidden_size, self.padding_idx
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        self.layer_norm = None
        self.layers = nn.ModuleList(
            [OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
            )

        positions = self.embed_positions(attention_mask, positions=position_ids)[
            :, past_key_values_length:, :
        ]

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + positions

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    transformers.logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomOPTModel(OPTModel):
    def __init__(self, config):
        super(OPTModel, self).__init__(config)
        self.decoder = CustomOPTDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


class CustomOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super(OPTForCausalLM, self).__init__(config)

        self.model = CustomOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OPT(HFLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super(HFLM, self).__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            revision=revision,
            subfolder=subfolder,
        )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode(
                "hello\n\nhello", add_special_tokens=False
            ) == [42891, 50118, 50118, 42891], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

        self.pretrained = pretrained
        self.has_logged = False

        self._load_opt_model()


    def _load_opt_model(self):
        weights_path = huggingface_hub.snapshot_download(self.pretrained)
        files = os.listdir(weights_path)
        weights_path = (
            os.path.join(weights_path, "pytorch_model.bin")
            if "pytorch_model.bin" in files
            else weights_path
        )
        print("OPT weights path", weights_path)

        config = transformers.AutoConfig.from_pretrained(self.pretrained)

        # Initializes an empty shell with the model. This is instant and does not take any RAM.
        with accelerate.init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(config)
        # Initialize the model under the previous context manager breaks the tied weights.
        model.tie_weights()

        # Infer device map automatically
        device_map = accelerate.infer_auto_device_map(
            model.model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16"
        )

        if "decoder.embed_tokens" in device_map and "decoder.layers.0" in device_map:
            embed_tokens_device = device_map["decoder.embed_tokens"]
            num_gpus = torch.cuda.device_count()
            num_layers = config.num_hidden_layers

            # Evenly distribute layers between gpus
            if num_gpus > 0:
                bs = ((num_layers - 1) // num_gpus) + 1
                for k, i in enumerate(range(0, num_layers - 1, bs)):
                    for j in range(i, min(i + bs, num_layers - 1)):
                        device_map[f"decoder.layers.{j}"] = k

            # Last layer needs to be on the same device as the token embedding
            # because the vocabulary projection layer has its weight tied to the token embedding
            device_map[f"decoder.layers.{num_layers - 1}"] = embed_tokens_device

        print(f"Device map for loading {self.pretrained}")
        print(device_map)

        accelerate.load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            dtype="float16",
        )
        model.tie_weights()

        self.gpt2 = model
        self.gpt2.eval()


class OPTWithPhaseShift(OPT):
    def __init__(self, phase_shift=0, **kwargs):
        super().__init__(**kwargs)
        self.phase_shift = int(phase_shift)

    def _load_opt_model(self):
        weights_path = huggingface_hub.snapshot_download(self.pretrained)
        files = os.listdir(weights_path)
        weights_path = (
            os.path.join(weights_path, "pytorch_model.bin")
            if "pytorch_model.bin" in files
            else weights_path
        )
        print("OPT weights path:", weights_path)

        config = transformers.AutoConfig.from_pretrained(self.pretrained)

        # Initializes an empty shell with the model. This is instant and does not take any RAM.
        with accelerate.init_empty_weights():
            model = CustomOPTForCausalLM(config)
        # Initialize the model under the previous context manager breaks the tied weights.
        model.tie_weights()

        # Infer device map automatically
        device_map = accelerate.infer_auto_device_map(
            model.model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16"
        )

        if "decoder.embed_tokens" in device_map and "decoder.layers.0" in device_map:
            embed_tokens_device = device_map["decoder.embed_tokens"]
            num_gpus = torch.cuda.device_count()
            num_layers = config.num_hidden_layers

            # Evenly distribute layers between gpus
            if num_gpus > 0:
                bs = ((num_layers - 1) // num_gpus) + 1
                for k, i in enumerate(range(0, num_layers - 1, bs)):
                    for j in range(i, min(i + bs, num_layers - 1)):
                        device_map[f"decoder.layers.{j}"] = k

            # Last layer needs to be on the same device as the token embedding
            # because the vocabulary projection layer has its weight tied to the token embedding
            device_map[f"decoder.layers.{num_layers-1}"] = embed_tokens_device

        print(f"Device map for loading {self.pretrained}")
        print(device_map)

        accelerate.load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            dtype="float16",
        )
        model.tie_weights()

        self.gpt2 = model
        self.gpt2.eval()

        print(type(self.gpt2))

    def _model_call(self, inps: torch.Tensor):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if self.phase_shift != 0:
                batch_size, seq_len = inps.shape
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=inps.device
                )
                position_ids = (
                    position_ids.unsqueeze(0).view(-1, seq_len).repeat(batch_size, 1)
                )
                position_ids += self.phase_shift
                position_ids = position_ids.to(inps.device)
            else:
                position_ids = None

            if not self.has_logged:
                print("-------> inputs:")
                print(self.tokenizer.batch_decode(inps)[0])
                self.has_logged = True

            return self.gpt2(input_ids=inps, position_ids=position_ids)[0][:, :, :50257]
