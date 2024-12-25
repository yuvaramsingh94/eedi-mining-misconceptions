import json
import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
from accelerate.logging import get_logger
from llm_reasoner.eedi_dataset import MathDataset
from llm_reasoner.eedi_loader import TextCollator, show_batch
from llm_reasoner.eedi_optimizer import get_optimizer
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils.train_utils import AverageMeter, as_minutes, download_kaggle_dataset, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run

logger = get_logger(__name__)


torch._dynamo.config.optimize_ddp = False


def to_list(t):
    return t.float().cpu().tolist()


def run_evaluation(accelerator, model, valid_dl):
    model.eval()
    all_losses = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for _, batch in enumerate(valid_dl):
        with torch.no_grad():
            outputs = model(**batch)

        batch_losses = accelerator.gather_for_metrics(outputs.loss)
        batch_losses = to_list(batch_losses)
        all_losses.extend(batch_losses)

        progress_bar.update(1)
    progress_bar.close()

    # --
    eval_dict = dict()
    eval_dict["valid_loss"] = np.mean(all_losses)

    return eval_dict


@hydra.main(version_base=None, config_path="../conf/llm_reasoner", config_name="conf_reasoner_14b")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.local_rank = accelerator.process_index

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    print_line()
    accelerator.print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    # download data in main process
    with accelerator.main_process_first():
        input_dir = download_kaggle_dataset(cfg.dataset.comp_dataset)
        fold_dir = download_kaggle_dataset(cfg.dataset.fold_dataset)
        cot_dir = download_kaggle_dataset(cfg.dataset.cot_dataset)
    accelerator.wait_for_everyone()

    df = pd.read_csv(os.path.join(input_dir, "pair.csv")).rename(columns={"QuestionId": "query_id"})
    df["QuestionId"] = df["query_id"].apply(lambda x: x.split("_")[0])
    df["QuestionId"] = df["QuestionId"].astype(int)

    # custom train-valid split
    fold_df = pd.read_parquet(os.path.join(fold_dir, "folds.parquet"))  # .rename(columns={"QuestionId": "query_id"})
    df = pd.merge(df, fold_df, on="QuestionId", how="left")
    df["kfold"] = df["kfold"].fillna(99).astype(int)
    accelerator.print(f"# of folds: {df['kfold'].nunique()}")
    accelerator.print("Fold distribution:")
    accelerator.print(df["kfold"].value_counts())

    train_folds = cfg.train_folds

    if cfg.full_fit:
        train_df = df.copy()
        valid_df = df[df["kfold"] == 0].copy()
        valid_df = valid_df.sample(min(1024, len(valid_df))).copy()
    else:
        train_df = df[df["kfold"].isin(train_folds)].copy()
        valid_df = df[df["kfold"] == 0].copy()
        valid_df = valid_df.sample(min(1024, len(valid_df))).copy()

    train_df = train_df.drop(columns=["kfold"]).reset_index(drop=True)
    valid_df = valid_df.drop(columns=["kfold"]).reset_index(drop=True)

    print(f"# of train: {train_df.shape[0]}")
    print(f"# of valid: {valid_df.shape[0]}")
    # -----

    if cfg.debug:
        n = min(1024, len(train_df))
        n = min(n, len(valid_df))
        train_df = train_df.head(n).reset_index(drop=True)
        valid_df = valid_df.head(n).reset_index(drop=True)

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    print_line()

    accelerator.wait_for_everyone()

    explanation_df = pd.read_csv(os.path.join(cot_dir, "synthetic.csv"))
    accelerator.print(f"shape of explanation data: {explanation_df.shape}")

    accelerator.print(f"shape of train before merge: {train_df.shape}")
    train_df = train_df.merge(explanation_df[["query_id", "Explanation"]], on="query_id", how="inner")
    accelerator.print(f"shape of train after merge: {train_df.shape}")

    accelerator.print(f"shape of validation before merge: {valid_df.shape}")
    valid_df = valid_df.merge(explanation_df[["query_id", "Explanation"]], on="query_id", how="inner")
    accelerator.print(f"shape of validation after merge: {valid_df.shape}")

    dataset_creator = MathDataset(cfg)
    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer
    valid_ds = valid_ds.sort("length")
    accelerator.print("len of valid_ds", len(valid_ds))

    data_collator = TextCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,  # True,
        collate_fn=data_collator,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()
    accelerator.print("Show training batch")
    for bid, b in enumerate(train_dl):
        accelerator.print(f"Batch {bid}")
        show_batch(b, tokenizer, task="training", print_fn=accelerator.print)
        if bid > 8:
            break
    print_line()

    accelerator.print("Show validation (first batch)")

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task="validation", print_fn=accelerator.print)

    print_line()

    accelerator.print("Show validation (last batch)")
    for b in valid_dl:
        pass
    show_batch(b, tokenizer, task="validation", print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()

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
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    model.config.pretraining_tp = 1

    if cfg.model.use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # lora ---
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            use_dora=cfg.model.lora.use_dora,
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=cfg_dict["model"]["lora"]["target_modules"],
            modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.config.use_cache = False
    accelerator.wait_for_everyone()

    if cfg.model.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model)

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # ------- training setup --------------------------------------------------------------#
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(f"STEP: {current_iteration+1:5}/{num_training_steps:5}. " f"LR: {get_lr(optimizer):.4f}. " f"Loss: {loss_meter.avg:.4f}. ")

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)
                    accelerator.log({"total_grad_l2_norm": round(grad_norm.item(), 5)}, step=current_iteration)

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl)

                lb = eval_response["valid_loss"]

                print_line()
                et = as_minutes(time.time() - start_time)
                accelerator.print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                print_line()
                accelerator.print(f">>> Current LB (valid loss) = {round(lb, 4)}")

                print_line()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    if cfg.save_model:
                        unwrapped_model = accelerator.unwrap_model(model)

                        unwrapped_model.save_pretrained(
                            f"{cfg.outputs.model_dir}/last",
                            state_dict=accelerator.get_state_dict(model),
                            save_function=accelerator.save,
                        )

                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()
    # --

    # --- save at the end ------------------------------------------------------------#
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.save_model:
            accelerator.print("Saving model at the end...")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f"{cfg.outputs.model_dir}/last")
            tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
