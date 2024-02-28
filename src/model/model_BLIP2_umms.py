from collections import defaultdict
import csv
import json
import os
import unicodedata
import math
from typing import Union
import sys

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd

from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from modeling_umms_blip2 import UMLSUM_Blip2
from torchmetrics.text.rouge import ROUGEScore
from peft import get_peft_config, get_peft_model, LoraConfig


MAX_TGT_LEN = 128
MAX_SRC_LEN = 1024


class UMMSTransformerBLIP2(pl.LightningModule):
    def __init__(
        self,
        pre_trained_ckpt: str = "",
        append_task_id=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self._create_model()

        self.validation_step_outputs = defaultdict(list)

        self.rougescore = ROUGEScore(rouge_keys="rougeL")

        self.automatic_optimization = False

    def _create_model(self):
        self.model = UMLSUM_Blip2.from_pretrained(
            self.hparams.pre_trained_ckpt, torch_dtype=torch.float16
        )

        self.processor = AutoProcessor.from_pretrained(self.hparams.pre_trained_ckpt)

        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q", "v", "q_proj", "v_proj", "dense"],
            layers_to_transform=list(range(100)),
            layers_pattern=["anguage_model.*layer", "former.*layer"],
        )

        self.model = get_peft_model(self.model, peft_config)

    def forward(
        self,
        pixel_values,
        pixel_embeds,
        images_per_text,
        input_ids,
        attention_mask,
        labels,
        decoder_attention_mask,
    ):
        _out = self.model.forward(
            pixel_values=pixel_values,
            pixel_embeds=pixel_embeds,
            images_per_text=images_per_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        self.log("Loss", _out["loss"])
        return _out["loss"]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # Assumes we train with CombinedLoader
        for _name, _batch in batch.items():
            input_ids = _batch["src_ids"]
            attention_mask = _batch["src_mask"]

            pixel_values = (
                _batch["pixel_values"] if "pixel_values" in _batch.keys() else None
            )
            pixel_embeds = (
                _batch["pixel_embeds"] if "pixel_embeds" in _batch.keys() else None
            )
            images_per_text = (
                _batch["images_per_text"]
                if "images_per_text" in _batch.keys()
                else None
            )

            labels = _batch["tgt_ids"]
            tgt_mask = _batch["tgt_mask"]

            # Compute loss
            _loss = self.forward(
                pixel_values=pixel_values,
                pixel_embeds=pixel_embeds,
                images_per_text=images_per_text,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=tgt_mask,
            )

            for pg in opt.param_groups:
                # Assumes there is a single param group
                self.log("learning_rate", pg["lr"], on_step=True, on_epoch=False)

            self.manual_backward(_loss)

        if (batch_idx + 1) % 20 == 0:
            self.clip_gradients(
                opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt.step()
            opt.zero_grad()

        return _loss

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["src_ids"]
        attention_mask = batch["src_mask"]

        pixel_values = batch["pixel_values"] if "pixel_values" in batch.keys() else None
        pixel_embeds = batch["pixel_embeds"] if "pixel_embeds" in batch.keys() else None
        images_per_text = (
            batch["images_per_text"] if "images_per_text" in batch.keys() else None
        )

        # Generate the summary using the text and visual features
        txt_summary_tokens = self.model.generate(
            pixel_values=pixel_values,
            pixel_embeds=pixel_embeds,
            images_per_text=images_per_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            max_new_tokens=MAX_TGT_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )

        predicted_sent = self.processor.tokenizer.batch_decode(
            txt_summary_tokens, skip_special_tokens=True
        )

        if "pixel_values" not in batch.keys() and "pixel_embeds" not in batch.keys():
            return {"hyp": predicted_sent}

        _predicted_sents, _predicted_frames = [], []
        _img_id_predicted = 0

        for _p_sent in predicted_sent:
            try:
                _sent, _frame = _p_sent.split(" img_ind_")
                _frame = int(_frame)
            except:
                _sent = _p_sent
                _frame = 0
            else:
                _img_id_predicted += 1
            _predicted_sents.append(_sent)
            _predicted_frames.append(_frame)

        return {
            "hyp": _predicted_sents,
            "frame_ids": _predicted_frames,
            "visual_predicted": _img_id_predicted,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.prediction_step(batch, batch_idx, dataloader_idx)

        _out = {
            "hyp": predictions["hyp"],
            "ref": [sent for sent in batch["tgt"]],
        }

        # Text-only
        if "pixel_values" not in batch.keys() and "pixel_embeds" not in batch.keys():
            self.validation_step_outputs[dataloader_idx].append(_out)
            return _out

        _out["selected_cosine_sim"] = [
            _scores[_pid] if _pid < len(_scores) else _scores[0]
            for _pid, _scores in zip(predictions["frame_ids"], batch["raw_cos_sim"])
        ]
        _out["is_top_one"] = [
            np.argmax(_scores) == _pid
            for _pid, _scores in zip(predictions["frame_ids"], batch["raw_cos_sim"])
        ]
        _out["frame_ids"] = [
            _pid if _pid < len(_scores) else 0
            for _pid, _scores in zip(predictions["frame_ids"], batch["raw_cos_sim"])
        ]
        _out["frame_refs"] = [np.argmax(_scores) for _scores in batch["raw_cos_sim"]]
        _out["is_img_predicted"] = predictions["visual_predicted"] / len(batch["tgt"])

        self.validation_step_outputs[dataloader_idx].append(_out)
        return _out

    def on_validation_epoch_end(self, *arg, **kwargs):
        _rougescores = []
        _cossim = []
        for _id, _list in self.validation_step_outputs.items():
            # Text similarity metrics
            predictions = [sent for _out in _list for sent in _out["hyp"]]
            refs = [sent for _out in _list for sent in _out["ref"]]

            df = pd.DataFrame()

            self.rougescore.update(preds=predictions, target=refs)
            rougescore = self.rougescore.compute()
            self.log(
                "ROUGEL_SCORE_F: " + str(_id),
                torch.round(100 * rougescore["rougeL_fmeasure"], decimals=2),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            _rougescores.append(rougescore["rougeL_fmeasure"])
            self.rougescore.reset()

            if "selected_cosine_sim" in _list[0].keys():
                # Image/Frame selection scores
                cosine_sims = [
                    score for _out in _list for score in _out["selected_cosine_sim"]
                ]
                self.log(
                    "AVG_COS_SIM: " + str(_id),
                    round(np.mean(cosine_sims), 2),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

                _is_top_one = [score for _out in _list for score in _out["is_top_one"]]
                self.log(
                    "FRAME_TOP_1_ACCURACY: " + str(_id),
                    round(100 * sum(_is_top_one) / len(_is_top_one), 2),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

                _is_img_predicted = [100 * _out["is_img_predicted"] for _out in _list]
                self.log(
                    "PERCENT_IMG_PREDICTED: " + str(_id),
                    round(np.mean(_is_img_predicted), 2),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

                df["FRAME_PRED"] = [
                    _fid for _out in _list for _fid in _out["frame_ids"]
                ]
                df["FRAME_REF"] = [
                    _fid for _out in _list for _fid in _out["frame_refs"]
                ]

            # Write predictions, this simple version requires that a single GPU (*no DDP*) is used
            if self.trainer.state.fn != "fit":
                df["PRED"] = predictions
                df["REF"] = refs
                df.to_csv(
                    os.path.join(
                        self.logger.log_dir,
                        self.trainer.state.fn + "_dl_" + str(_id) + ".tsv",
                    ),
                    sep="\t",
                    index=False,
                    quoting=csv.QUOTE_NONE,
                )

        self.log(
            "ROUGEL_SCORE_F_MIN",
            min(_rougescores),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.validation_step_outputs = defaultdict(list)

    def configure_optimizers(self):
        # Based on fine-tuning params in BLIP2 paper: https://arxiv.org/pdf/2301.12597.pdf
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.999),
            weight_decay=0.05,
        )
