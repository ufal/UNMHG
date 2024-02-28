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
import torch_optimizer as optim

from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from torchmetrics.text.rouge import ROUGEScore
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer


MAX_TGT_LEN = 128
VISUAL_FEATURES_SIZE = 1024


class UMMSTransformerT5(pl.LightningModule):
    def __init__(
        self,
        pre_trained_ckpt: str = "",
        append_task_id=False,
        visual_weight=1,
        is_single_task=False,
        is_text_only=False,
        start_with_img=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self._create_model()

        self.CE_loss = torch.nn.CrossEntropyLoss(
            weight=self.loss_weights, reduction="none"
        )
        self.validation_step_outputs = defaultdict(list)

        self.rougescore = ROUGEScore(rouge_keys="rougeL")

    def _create_model(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.pre_trained_ckpt,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pre_trained_ckpt)
        _old_len = len(self.tokenizer)
        # Add the new tokens that will be used to select frames
        new_tokens = [f"img_ind_{_ind}" for _ind in range(351)]
        if self.hparams.append_task_id:
            new_tokens.extend(["t+v->t+i", "t+i->t+i", "t->t"])

        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Modify the loss function to put more weight towards frame/image indices
        visual_indices = torch.tensor([_old_len + _ind for _ind in range(351)]).long()
        loss_weights = torch.ones(len(self.tokenizer))
        loss_weights[visual_indices] = self.hparams.visual_weight
        self.loss_weights = loss_weights

        if not self.hparams.is_text_only:
            # Additional, vision-related parameters
            self.vision_projection = torch.nn.Linear(
                in_features=VISUAL_FEATURES_SIZE,  # using CLIP large features
                out_features=self.model.config.d_model,
                bias=False,
            )

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
    ):
        _out = self.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        self.log("Inner CE loss", _out["loss"])
        return _out["logits"]

    def training_step(self, batch, batch_idx):
        _losses = []
        # Assumes we train with CombinedLoader
        for _name, _batch in batch.items():
            src_tokens = _batch["src_ids"]
            src_mask = _batch["src_mask"]

            src_token_embeds = self.model.encoder.embed_tokens(src_tokens)
            if "visual_features" in _batch.keys():
                visual_features = self.vision_projection(_batch["visual_features"])
                if _batch["task_type"] == "t+v->t+i":
                    _positions = visual_features.shape[1]
                    p_enc = Summer(PositionalEncodingPermute1D(_positions))
                    visual_features = p_enc(visual_features)

                src_token_embeds = torch.cat([visual_features, src_token_embeds], dim=1)
                src_mask = torch.cat([_batch["visual_mask"], src_mask], dim=1)

            assert src_token_embeds.shape[:2] == src_mask.shape

            tgt_tokens = _batch["tgt_ids"]
            tgt_mask = _batch["tgt_mask"]
            tgt_probs_for_loss = _batch["tgt_probs_for_loss"]

            # Compute text summary loss
            logits = self.forward(
                inputs_embeds=src_token_embeds,
                attention_mask=src_mask,
                labels=tgt_tokens,
            )
            _loss = self.CE_loss(
                logits.permute(0, 2, 1), tgt_probs_for_loss.permute(0, 2, 1)
            )
            _loss = torch.mean(
                torch.where(tgt_mask == 1, _loss, torch.zeros_like(_loss))
            )
            if self.hparams.is_single_task:
                _alpha = 1.0
            else:
                _alpha = 1 / 3
            _losses.append(_alpha * _loss)

        _all_task_loss = sum(_losses)
        self.log("CE loss AVG", _all_task_loss)
        return _all_task_loss

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        src_tokens = batch["src_ids"]
        src_mask = batch["src_mask"]
        src_token_embeds = self.model.encoder.embed_tokens(src_tokens)
        if "visual_features" in batch.keys():
            visual_features = self.vision_projection(batch["visual_features"])
            if batch["task_type"] == "t+v->t+i":
                _positions = visual_features.shape[1]
                p_enc = Summer(PositionalEncodingPermute1D(_positions))
                visual_features = p_enc(visual_features)

            src_token_embeds = torch.cat([visual_features, src_token_embeds], dim=1)
            src_mask = torch.cat([batch["visual_mask"], src_mask], dim=1)

        assert src_token_embeds.shape[:2] == src_mask.shape
        # Generate the summary using the text and visual features
        txt_summary_tokens = self.model.generate(
            inputs_embeds=src_token_embeds,
            attention_mask=src_mask,
            num_beams=4,
            max_new_tokens=MAX_TGT_LEN,
            repetition_penalty=2.5,
            length_penalty=1.0,
        )
        predicted_sent = self.tokenizer.batch_decode(
            txt_summary_tokens, skip_special_tokens=True
        )

        if "visual_features" not in batch.keys():
            return {"hyp": predicted_sent}

        _predicted_sents, _predicted_frames = [], []
        _img_id_predicted = 0

        for _p_sent in predicted_sent:
            if self.hparams.start_with_img:
                if _p_sent.startswith("img_ind_"):
                    try:
                        _tmp_sent = _p_sent[len("img_ind_") :]
                        _frame = int(_tmp_sent.split()[0])
                        _sent = " ".join(_tmp_sent.split()[1:])
                    except:
                        _sent = _p_sent
                        _frame = 0
                    else:
                        _img_id_predicted += 1
                else:
                    _sent = _p_sent
                    _frame = 0

            else:
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
            "ref": batch["tgt"],
        }

        # Text-only
        if "raw_cos_sim" not in batch.keys():
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

        # Min score for checkpointing/early stopping
        self.log(
            "ROUGEL_SCORE_F_MIN",
            min(_rougescores),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.validation_step_outputs = defaultdict(list)

    def configure_optimizers(self):
        # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor
        return optim.Adafactor(
            self.model.parameters(),
            lr=1e-3,
            eps2=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer.step(closure=optimizer_closure)
        for pg in optimizer.param_groups:
            # Assumes there is a single param group
            self.log("learning_rate", pg["lr"], on_step=True, on_epoch=False)
