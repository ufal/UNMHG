# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py
# that is licensed under Apache-2.0 license, i.e.

# Copyright 2023 The Salesforce Authors and The HuggingFace Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BLIP-2 model extended to handle articles with multiple images and text-only articles"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import (
    Blip2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Blip2Config,
    Blip2QFormerConfig,
    Blip2VisionConfig,
)

from transformers.models.blip_2.modeling_blip_2 import (
    Blip2ForConditionalGenerationModelOutput,
    BLIP_2_INPUTS_DOCSTRING,
    BLIP_2_START_DOCSTRING,
)

from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)


class UMLSUM_Blip2(Blip2ForConditionalGeneration):
    def _preprocess_accelerate(self):
        hf_device_map = self.hf_device_map
        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True

    def forward(
        self,
        images_per_text,
        input_ids,
        pixel_embeds=None,
        pixel_values=None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # Need to add because of PEFT
        inputs_embeds=None,
        decoder_inputs_embeds=None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is not None or pixel_embeds is not None:
            # step 1: forward the images (if present) through the vision encoder,
            # ---- Allows processing more than 1 image per piece of text, or no images at all ----
            max_imgs_per_text = max(images_per_text)

            if pixel_embeds is not None:
                image_embeds = pixel_embeds
                vision_outputs = None
            else:
                # ---- To get image embeddings of shape: (batch_size, seq_len, hidden_size) ----
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                image_embeds = vision_outputs[0]

            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # ---- Concatenate images corresponding to the same text, i.e., reshape so that images from the same article are together ----
            query_output = query_outputs[0]
            query_output = torch.split(query_output, images_per_text, dim=0)
            query_output = [q_img.view(-1, q_img.shape[-1]) for q_img in query_output]

            query_output = nn.utils.rnn.pad_sequence(
                query_output, batch_first=True, padding_value=0.0
            )
            # ---- Size is now: batch_size x (max(images_per_text)*32) x config.qformer_config.hidden_size ----

            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.language_projection(query_output)

            language_attention_mask = (
                torch.tensor(
                    [
                        _inum * [1] + (max_imgs_per_text - _inum) * [0]
                        for _inum in images_per_text
                    ]
                )
                .view(-1, max_imgs_per_text)
                .long()
                .to(language_model_inputs.device)
            )
            language_attention_mask = torch.repeat_interleave(
                language_attention_mask, self.config.num_query_tokens, dim=1
            ).to(language_model_inputs.device)
        else:
            vision_outputs = None
            query_outputs = None

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None or pixel_embeds is not None:
            inputs_embeds = torch.cat(
                [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
                dim=1,
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if pixel_values is not None or pixel_embeds is not None:
            attention_mask = torch.cat(
                [
                    language_attention_mask,
                    attention_mask.to(language_attention_mask.device),
                ],
                dim=1,
            )

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # ---- We compute the loss here since we need to take into account the sequence length of the query embeddings ----
            if labels is not None:
                logits = logits[:, -labels.size(1) :, :]
                # ---- Shift so that only tokens < n are used to predict n ----
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size),
                    shift_labels.view(-1),
                )
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]],
        images_per_text,
        pixel_embeds=None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Further override adds the ability to handle more than 1 image for a particular piece of text.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            images_per_text (`torch.LongTensor`):
                Number of input images per text sequence.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        if pixel_values is not None or pixel_embeds is not None:
            # ---- Allow more than 1 image per piece of text ----
            max_imgs_per_text = max(images_per_text)

            if pixel_embeds is not None:
                image_embeds = pixel_embeds
            else:
                image_embeds = self.vision_model(
                    pixel_values, return_dict=True
                ).last_hidden_state

            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state

            # ---- Concatenate images corresponding to the same text, i.e., reshape so that images from the same article are together ----
            query_output = torch.split(query_output, images_per_text, dim=0)
            query_output = [q_img.view(-1, q_img.shape[-1]) for q_img in query_output]

            query_output = nn.utils.rnn.pad_sequence(
                query_output, batch_first=True, padding_value=0.0
            )
            # ---- Size is now: batch_size x (max(images_per_text)*32) x config.qformer_config.hidden_size ----

            language_model_inputs = self.language_projection(query_output)
            language_attention_mask = (
                torch.tensor(
                    [
                        _inum * [1] + (max_imgs_per_text - _inum) * [0]
                        for _inum in images_per_text
                    ]
                )
                .view(-1, max_imgs_per_text)
                .long()
                .to(language_model_inputs.device)
            )
            language_attention_mask = torch.repeat_interleave(
                language_attention_mask, self.config.num_query_tokens, dim=1
            ).to(language_model_inputs.device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if pixel_values is not None or pixel_embeds is not None:
            attention_mask = torch.cat(
                [language_attention_mask, attention_mask.to(attention_mask.device)],
                dim=1,
            )

        # ---- Concatenate query embeddings with prompt embeddings ----
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None or pixel_embeds is not None:
            inputs_embeds = torch.cat(
                [language_model_inputs, inputs_embeds.to(attention_mask.device)], dim=1
            )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
