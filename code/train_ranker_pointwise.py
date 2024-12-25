import json
import os
import random

import hydra
import pandas as pd
import torch
from accelerate.logging import get_logger
from datasets import Dataset
from omegaconf import OmegaConf
from ranker_pointwise.ranker_dataset import RankerDataset
from ranker_pointwise.ranker_loader import RankerCollator, RankerCollatorTrain, show_batch
from ranker_pointwise.ranker_model import EediRanker, get_base_model
from ranker_pointwise.ranker_optimizer import get_optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.train_utils import (
    AverageMeter,
    add_fs_examples,
    download_kaggle_dataset,
    eedi_process_df,
    get_custom_cosine_schedule_with_warmup,
    get_lr,
    setup_training_run,
    train_valid_split,
)

logger = get_logger(__name__)

torch._dynamo.config.optimize_ddp = False


def format_example(row):
    example = f"Question: {row['QuestionText']}\nAnswer:{row['CorrectAnswerText']}\nMisconception Answer: {row['InCorrectAnswerText']}"
    return example


@hydra.main(version_base=None, config_path="../conf/ranker_pointwise", config_name="conf_pointwise_14b")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.local_rank = accelerator.process_index
    rng = random.Random(cfg.seed)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    print_line()
    accelerator.print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    # download data in main process
    with accelerator.main_process_first():
        input_dir = download_kaggle_dataset(cfg.dataset.input_dataset)
        comp_dir = download_kaggle_dataset(cfg.dataset.comp_dataset)
        label_dir = download_kaggle_dataset(cfg.dataset.label_dataset)
    accelerator.wait_for_everyone()

    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(input_dir, "valid.parquet"))

    if cfg.full_fit:
        valid_df = pd.read_parquet(os.path.join(input_dir, "valid_ff.parquet"))
        train_df = pd.concat([train_df, valid_df]).reset_index(drop=True)
        valid_df = valid_df.head(128).reset_index(drop=True)

    train_df["content_id"] = train_df["content_id"].astype(str)
    valid_df["content_id"] = valid_df["content_id"].astype(str)

    # add combined_id ---
    train_df["combined_id"] = train_df["query_id"] + "|" + train_df["content_id"]
    valid_df["combined_id"] = valid_df["query_id"] + "|" + valid_df["content_id"]

    # For few-shot examples
    comp_df = pd.read_csv(os.path.join(comp_dir, "train.csv")).rename(columns={"QuestionId": "query_id"})
    if cfg.full_fit:
        print_line()
        accelerator.print("using full fit for Few Shot")
        fs_df = comp_df
        print_line()
    else:
        fs_df, _ = train_valid_split(cfg, comp_df)

    # fs_df, _ = train_valid_split(cfg, comp_df)  # use few shot examples from training set only
    query_df, _, content2query = eedi_process_df(fs_df)

    label_df = pd.read_csv(os.path.join(label_dir, "train.csv"))
    _, corr_df_all, _ = eedi_process_df(label_df)

    query_df["demo"] = query_df.apply(format_example, axis=1)
    query2example = dict(zip(query_df["query_id"], query_df["demo"]))

    train_df = add_fs_examples(train_df, content2query, query2example, rng, is_train=True, k_shot=cfg.model.k_shot)
    valid_df = add_fs_examples(valid_df, content2query, query2example, rng, is_train=False, k_shot=cfg.model.k_shot)

    qa2mc = dict(zip(corr_df_all["query_id"], corr_df_all["content_id"]))  # label map (qa_id -> mc_id)
    query2candidates_train = train_df.groupby("query_id")["content_id"].agg(list).reset_index()  # candidate map
    query2candidates_train = dict(zip(query2candidates_train["query_id"], query2candidates_train["content_id"]))

    if cfg.debug:
        n = min(1024, len(train_df))
        n = min(n, len(valid_df))
        train_df = train_df.head(n).reset_index(drop=True)
        valid_df = valid_df.head(n).reset_index(drop=True)

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    print_line()

    # create dataset ---
    dataset_creator = RankerDataset(cfg)
    train_ds = dataset_creator.get_dataset(train_df, is_train=True)
    valid_ds = dataset_creator.get_dataset(valid_df, is_train=False)

    tokenizer = dataset_creator.tokenizer
    valid_ds = valid_ds.sort("length", reverse=True)

    data_collator = RankerCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    if cfg.use_distillation:
        assert "teacher_score" in train_df.columns, "teacher_score is not in the train dataframe"
        teacher_map = dict(zip(train_df["combined_id"], train_df["teacher_score"]))
    else:
        teacher_map = None

    kwargs = dict(
        cfg=cfg,
        ds=train_ds,
        query2candidates=query2candidates_train,
        label_map=qa2mc,
        teacher_map=teacher_map,
    )

    data_collator_train = RankerCollatorTrain(tokenizer=tokenizer, kwargs=kwargs)

    qids = train_df["query_id"].unique().tolist()
    qid_df = pd.DataFrame([{"query_id": qid} for qid in qids])

    train_dl = DataLoader(
        Dataset.from_pandas(qid_df),
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,  # True,
        collate_fn=data_collator_train,
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
    for b in train_dl:
        break
    show_batch(b, tokenizer, task="training", print_fn=accelerator.print)
    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task="validation", print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    base_model = get_base_model(cfg)
    model = EediRanker(cfg, base_model, tokenizer)

    if cfg.model.use_gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
            with accelerator.accumulate(model):
                batch["temperature"] = cfg.temperature
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
                    accelerator.log({"yes_logit_mean": round(outputs.yes_logit_mean.item(), 5)}, step=current_iteration)
                    accelerator.log({"no_logit_mean": round(outputs.no_logit_mean.item(), 5)}, step=current_iteration)
                    try:
                        accelerator.log({"distillation_loss": round(outputs.distillation_loss.item(), 5)}, step=current_iteration)
                        accelerator.log({"ce_loss": round(outputs.ce_loss.item(), 5)}, step=current_iteration)
                    except Exception:
                        pass

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                model.eval()
                accelerator.wait_for_everyone()

                if cfg.save_model:
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save(cfg.outputs.model_dir)
                        tokenizer.save_pretrained(cfg.outputs.model_dir)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

    # --- end training
    # save at the end of training
    if cfg.save_model:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(cfg.outputs.model_dir)
            tokenizer.save_pretrained(cfg.outputs.model_dir)

    accelerator.end_training()


if __name__ == "__main__":
    run_training()
