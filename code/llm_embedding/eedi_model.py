from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.file_utils import ModelOutput


@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    nce_loss: Optional[Tensor] = None
    distill_loss: Optional[Tensor] = None


def get_base_model(cfg):
    config = AutoConfig.from_pretrained(cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code)
    config.use_cache = False

    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModel.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    model.config.pretraining_tp = 1

    # LoRA ---
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # CAUSAL_LM, FEATURE_EXTRACTION
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


class BiEncoderModel(nn.Module):
    def __init__(self, cfg, base_model, accelerator):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.use_distillation = cfg.model.use_distillation

        self.sub_batch_size = cfg.train_params.sub_batch_size

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.sentence_pooling_method = cfg.model.sentence_pooling_method

        self.accelerator = accelerator
        self.negatives_cross_device = cfg.model.negatives_cross_device  # accelerator.use_distributed
        if self.negatives_cross_device:
            assert accelerator.use_distributed, "Distributed training is required for negatives_cross_device"

        self.world_size = accelerator.num_processes
        self.process_rank = accelerator.process_index

        accelerator.print(f"negatives_cross_device: {self.negatives_cross_device}")
        accelerator.print(f"world_size: {self.world_size}")
        accelerator.print(f"process_rank: {self.process_rank}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "last":
            return self.last_token_pool(hidden_state, mask)

    def encode(self, features):
        if self.sub_batch_size is not None and self.sub_batch_size > 0:
            all_p_reps = []
            for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                end_inx = min(i + self.sub_batch_size, len(features["attention_mask"]))
                sub_features = {k: v[i:end_inx] for k, v in features.items()}
                last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                p_reps = self.sentence_embedding(last_hidden_state, sub_features["attention_mask"])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
        else:
            last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
            all_p_reps = self.sentence_embedding(last_hidden_state, features["attention_mask"])

        return all_p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps, temperature=0.02):
        x = q_reps.unsqueeze(1)
        y = p_reps.unsqueeze(0)

        sim = nn.CosineSimilarity(dim=-1)(x, y) / temperature
        return sim

    def forward(self, queries, contents, teacher_scores, temperature=0.02):
        q_reps = self.encode(queries)
        p_reps = self.encode(contents)
        # self.accelerator.print(f"before dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

        if self.negatives_cross_device:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            # self.accelerator.print(f"after dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

        group_size = p_reps.size(0) // q_reps.size(0)  # group_size = 1
        scores = self.compute_similarity(q_reps, p_reps, temperature)
        scores = scores.view(q_reps.size(0), -1)  # (n, m)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * group_size
        nce_loss = self.compute_loss(scores, target)

        # compute distillation loss
        if self.use_distillation:
            distill_loss = self.compute_distillation_loss(scores, teacher_scores)
            loss = 0.5 * nce_loss + 0.5 * distill_loss
        else:
            distill_loss = torch.tensor(0.0, device=scores.device)
            loss = nce_loss

        # print(f"loss: {loss}, nce_loss: {nce_loss}, distill_loss: {distill_loss}")

        return EmbedderOutput(loss=loss, scores=scores, nce_loss=nce_loss, distill_loss=distill_loss)

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def compute_distillation_loss(self, scores, teacher_scores, teacher_temperature=1.25):
        # print(f"scores: {scores.shape}, teacher_scores: {teacher_scores.shape}")
        # print(f"scores: {scores}")
        # print(f"teacher_scores: {teacher_scores}")
        teacher_targets = F.softmax(teacher_scores.detach() / teacher_temperature, dim=-1)  # (n, m)
        # print(f"teacher_targets: {teacher_targets}")
        loss = -torch.mean(torch.sum(torch.log_softmax(scores, dim=-1) * teacher_targets, dim=-1))
        return loss

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
