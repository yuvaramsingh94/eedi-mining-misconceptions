from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.file_utils import ModelOutput


@dataclass
class RankerOutput(ModelOutput):
    yes_logit_mean: Optional[Tensor] = None
    no_logit_mean: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    distillation_loss: Optional[Tensor] = None
    ce_loss: Optional[Tensor] = None


def get_base_model(cfg):
    config = AutoConfig.from_pretrained(cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code)
    config.use_cache = False

    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
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
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


class EediRanker(nn.Module):
    def __init__(self, cfg, base_model, tokenizer):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.group_size = cfg.train_params.per_device_train_group_size

        self.yes_loc = tokenizer("Yes", add_special_tokens=False)["input_ids"][-1]
        self.no_loc = tokenizer("No", add_special_tokens=False)["input_ids"][-1]

        print(f">> EediRanker: Yes token id: {self.yes_loc}")
        print(f">> EediRanker: No token id: {self.no_loc}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def encode_with_sub_batch(self, input_ids, attention_mask):
        sub_batch_size = 8
        all_logits = []
        yes_logit_means = []
        no_logit_means = []
        for i in range(0, len(input_ids), sub_batch_size):
            end_inx = min(i + sub_batch_size, len(input_ids))
            sub_input_ids = input_ids[i:end_inx]
            sub_attention_mask = attention_mask[i:end_inx]
            outputs = self.model(input_ids=sub_input_ids, attention_mask=sub_attention_mask, output_hidden_states=True)
            sub_score_yes = outputs.logits[:, -1, self.yes_loc]  # [bs]
            sub_score_no = outputs.logits[:, -1, self.no_loc]  # [bs]
            sub_logits = sub_score_yes - sub_score_no  # bs

            all_logits.append(sub_logits)
            yes_logit_means.append(sub_score_yes.abs().mean())
            no_logit_means.append(sub_score_no.abs().mean())

        all_logits = torch.cat(all_logits, 0)
        yes_logit_mean = torch.stack(yes_logit_means).mean()
        no_logit_mean = torch.stack(no_logit_means).mean()

        return all_logits.contiguous(), yes_logit_mean, no_logit_mean

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        score_yes = outputs.logits[:, -1, self.yes_loc]  # [bs]
        score_no = outputs.logits[:, -1, self.no_loc]  # [bs]
        all_logits = score_yes - score_no  # bs

        yes_logit_mean = score_yes.abs().mean()
        no_logit_mean = score_no.abs().mean()

        return all_logits.contiguous(), yes_logit_mean, no_logit_mean

    def forward(self, input_ids, attention_mask, labels=None, teacher_logits=None, ce_mask=None, teacher_mask=None, temperature=1.0, **kwargs):
        logits, yes_logit_mean, no_logit_mean = self.encode(input_ids, attention_mask)

        loss = None
        distillation_loss = None
        ce_loss = None

        if labels is not None:
            logits = logits.reshape(-1, self.group_size)
            labels = labels.to(logits.device).reshape(-1)  # bs
            ce_mask = ce_mask.to(logits.device).reshape(-1)
            ce_loss = (self.loss_fn(logits, labels) * ce_mask).mean()

            # distillation ----
            if teacher_logits is not None:
                teacher_targets = teacher_logits.reshape(-1, self.group_size)
                teacher_targets = torch.softmax(teacher_targets.detach() / temperature, dim=-1)
                distillation_loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=-1) * teacher_targets, dim=-1))
                loss = 0.5 * ce_loss + 0.5 * distillation_loss  # alpha = 0.5, beta = 0.5
            else:
                loss = ce_loss

        return RankerOutput(
            loss=loss,
            logits=logits,
            distillation_loss=distillation_loss,
            ce_loss=ce_loss,
            yes_logit_mean=yes_logit_mean,
            no_logit_mean=no_logit_mean,
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
