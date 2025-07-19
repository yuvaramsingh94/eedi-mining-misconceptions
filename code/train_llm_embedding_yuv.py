# ...existing code...
import json
import os

os.environ["KAGGLEHUB_CACHE"] = (
    "/storage/nas/da24s015/eedi-mining-misconceptions/dataset"
)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # set to your GPU ID
import random
import time
from copy import deepcopy
from dataclasses import dataclass

import hydra
import pandas as pd
import torch
from datasets import Dataset

# local imports
from llm_embedding.eedi_dataset import MathDataset
from llm_embedding.eedi_loader import (
    RetrieverDataCollator,
    TextCollator,
    show_batch,
    show_batch_fs,
)
from llm_embedding.eedi_model import BiEncoderModel, get_base_model
from llm_embedding.eedi_optimizer import get_optimizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.utils import logging as hf_logging
from utils.metric_utils import compute_retrieval_metrics, mapk
from utils.retriever_utils import semantic_search
from utils.train_utils import (
    as_minutes,
    download_kaggle_dataset,
    eedi_process_df,
    get_custom_cosine_schedule_with_warmup,
    get_lr,
    train_valid_split,
)

# options
pd.options.display.max_colwidth = 1000
os.environ["TOKENIZERS_PARALLELISM"] = "false"
hf_logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utils -------------------------------------------------------------------------------------------#


def get_temperature(cfg, progress):
    """use for temperature scheduling"""
    max_temp = cfg.model.max_temperature
    min_temp = cfg.model.min_temperature
    temp = max_temp - (max_temp - min_temp) * progress
    return temp


@dataclass
class IDTracker:
    """Track different IDs during training and evaluation"""

    query_train_ids: list
    query_valid_ids: list
    content_train_ids: list
    content_comp_ids: list


# -------- Mine Hard Negatives --------------------------------------------------------------------#
def mine_hard_negatives_online(cfg, model, query_dl, content_dl, label_df, id_tracker):
    query2content = label_df.groupby("query_id")["content_id"].apply(list).to_dict()
    model.eval()

    # query embeddings ---
    query_embeddings = []
    progress_bar = tqdm(query_dl, desc="Query Embeddings")
    for batch in progress_bar:
        with torch.no_grad():
            batch_query_embeddings = model.encode(batch)
        query_embeddings.append(batch_query_embeddings.cpu())
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_ids = id_tracker.query_train_ids
    print(f"shape of query embeddings (train): {query_embeddings.shape}")
    assert query_embeddings.shape[0] == len(query_ids)

    # content embeddings ---
    content_embeddings = []
    progress_bar = tqdm(content_dl, desc="Content Embeddings")
    for batch in progress_bar:
        with torch.no_grad():
            batch_content_embeddings = model.encode(batch)
        content_embeddings.append(batch_content_embeddings.cpu())
    content_embeddings = torch.cat(content_embeddings, dim=0)
    content_ids = id_tracker.content_train_ids
    print(f"shape of content embeddings (train): {content_embeddings.shape}")
    assert content_embeddings.shape[0] == len(content_ids)

    # search for negatives
    hard_negatives_map = dict()
    results = semantic_search(
        query_embeddings,
        content_embeddings,
        top_k=cfg.train_params.negative_depth_end + 1,
    )

    for idx, re_i in enumerate(results):
        query_id = query_ids[idx]
        hit_i = [node["corpus_id"] for node in re_i]
        top_content_ids_i = [content_ids[pos] for pos in hit_i]
        true_ids = query2content[query_id]
        negative_ids = [x for x in top_content_ids_i if x not in true_ids]
        hard_negatives_map[query_id] = negative_ids[
            cfg.train_params.negative_depth_start : cfg.train_params.negative_depth_end
        ]
    return hard_negatives_map


# -------- Evaluation -----------------------------------------------------------------------------#
def run_evaluation(cfg, model, query_dl, content_dl, label_df, id_tracker):
    cutoffs = [1, 2, 4, 8, 16, 25, 32, 64]
    label_df = deepcopy(label_df)
    query2content = label_df.groupby("query_id")["content_id"].apply(list).to_dict()
    model.eval()

    query_embeddings = []
    progress_bar = tqdm(query_dl, desc="Eval Query Embeddings")
    for batch in progress_bar:
        with torch.no_grad():
            batch_query_embeddings = model.encode(batch)
        query_embeddings.append(batch_query_embeddings.cpu())
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_ids = id_tracker.query_valid_ids
    print(f"shape of query embeddings: {query_embeddings.shape}")
    assert query_embeddings.shape[0] == len(query_ids)

    # get content embeddings ---
    content_embeddings = []
    progress_bar = tqdm(content_dl, desc="Eval Content Embeddings")
    for batch in progress_bar:
        with torch.no_grad():
            batch_content_embeddings = model.encode(batch)
        content_embeddings.append(batch_content_embeddings.cpu())
    content_embeddings = torch.cat(content_embeddings, dim=0)
    content_ids = id_tracker.content_comp_ids
    print(f"shape of content embeddings: {content_embeddings.shape}")
    assert content_embeddings.shape[0] == len(content_ids)

    # ------ evaluation ----------------------------------------------------------------#
    results = semantic_search(
        query_embeddings, content_embeddings, top_k=cfg.model.n_neighbour
    )

    true_content_ids = []
    pred_content_ids = []
    pred_scores = []

    for idx, re_i in enumerate(results):  # loop over query
        query_id = query_ids[idx]
        hit_i = [node["corpus_id"] for node in re_i]
        top_scores_i = [node["score"] for node in re_i]
        top_content_ids_i = [content_ids[pos] for pos in hit_i]
        pred_content_ids.append(top_content_ids_i)
        pred_scores.append(top_scores_i)
        true_content_ids.append(query2content[query_id])

    result_df = pd.DataFrame()
    result_df["query_id"] = query_ids
    result_df["true_ids"] = true_content_ids
    result_df["pred_ids"] = pred_content_ids
    result_df["pred_scores"] = pred_scores

    # compute metric ----
    eval_dict = dict()
    for cutoff in cutoffs:
        cdf = result_df.copy()
        cdf["pred_ids"] = cdf["pred_ids"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)
        eval_dict[f"precision@{cutoff}"] = m["precision_score"]
        eval_dict[f"recall@{cutoff}"] = m["recall_score"]

    # get mapk ---
    eval_dict["lb"] = mapk(
        result_df["true_ids"].values, result_df["pred_ids"].values, k=25
    )
    print(f">>> LB: {eval_dict['lb']}")

    # seen vs unseen
    content_train_ids = id_tracker.content_train_ids
    result_df["seen"] = result_df["true_ids"].apply(
        lambda x: True if x[0] in content_train_ids else False
    )
    seen_df = result_df[result_df["seen"]].reset_index(drop=True)
    unseen_df = result_df[~result_df["seen"]].reset_index(drop=True)
    eval_dict["seen_lb"] = mapk(
        seen_df["true_ids"].values, seen_df["pred_ids"].values, k=25
    )
    eval_dict["unseen_lb"] = mapk(
        unseen_df["true_ids"].values, unseen_df["pred_ids"].values, k=25
    )

    # get oof df
    oof_df = result_df.copy()
    oof_df = oof_df.drop(columns=["true_ids"])
    oof_df = oof_df.rename(columns={"query_id": "QuestionId_Answer"})
    oof_df = oof_df.rename(columns={"pred_ids": "MisconceptionId"})
    oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(
        lambda x: list(map(str, x))
    )
    oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(lambda x: " ".join(x))

    to_return = {
        "lb": eval_dict["lb"],
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
    }

    # logs -----
    scores_dict = eval_dict
    print("--------------------------------")
    print(f">>> LB: {scores_dict['lb']}")
    print(f">>> Seen LB: {scores_dict['seen_lb']}")
    print(f">>> Unseen LB: {scores_dict['unseen_lb']}")
    print("--------------------------------")
    for pt in cutoffs:
        print(f">>> Current Recall@{pt} = {round(scores_dict[f'recall@{pt}'], 4)}")

    return to_return


# -------- Main Function --------------------------------------------------------------------------#
@hydra.main(
    version_base=None,
    config_path="../conf/llm_embedding",
    config_name="conf_yuv.yaml",
)
def run_training(cfg):
    # ------- Runtime Configs ---------------------------------------------------------------------#
    def print_line(print_fn=print):
        prefix, unit, suffix = "#", "~~", "#"
        print_fn(prefix + unit * 50 + suffix)

    rng = random.Random(cfg.seed)
    ##! Modified for local rank instead of accelerator
    cfg.local_rank = 5  # accelerator.process_index
    # ------- load data --------------------------------------------------------------------------#
    data_dir_comp = download_kaggle_dataset(
        cfg.dataset.comp_dataset
    )  ## Competition dataset
    data_dir = download_kaggle_dataset(cfg.dataset.input_dataset)
    df = pd.read_csv(os.path.join(data_dir, "train.csv")).rename(
        columns={"QuestionId": "query_id"}
    )
    content_df = pd.read_csv(
        os.path.join(data_dir, "misconception_mapping.csv")
    ).rename(columns={"MisconceptionId": "content_id"})
    content_df_comp = pd.read_csv(
        os.path.join(data_dir_comp, "misconception_mapping.csv")
    ).rename(columns={"MisconceptionId": "content_id"})
    train_df, valid_df = train_valid_split(cfg, df)

    # process data --------------------------------------------------------------------------------#
    query_df_train, corr_df_train, content2query_train = eedi_process_df(
        train_df, debug=False
    )
    query_df_valid, corr_df_valid, _ = eedi_process_df(valid_df, debug=True)
    train_content_ids = corr_df_train["content_id"].unique().tolist()
    content_df_train = content_df[content_df["content_id"].isin(train_content_ids)]

    print_line()
    print(f"# of queries (train): {query_df_train.shape[0]}")
    print(f"# of queries (valid): {query_df_valid.shape[0]}")
    print(f"# shape of content data (train): {content_df_train.shape}")
    print(f"# shape of content data (comp): {content_df_comp.shape}")
    print(f"shape of corr_df_train: {corr_df_train.shape}")
    print_line()

    # mappings ---
    query2subject_train = dict(
        zip(query_df_train["query_id"], query_df_train["SubjectId"])
    )
    query2content_train = dict(
        zip(corr_df_train["query_id"], corr_df_train["content_id"])
    )
    subject2query_train = (
        query_df_train.groupby("SubjectId")["query_id"].apply(list).to_dict()
    )

    # ------- Datasets ----------------------------------------------------------------------------#
    ds_handle = MathDataset(cfg)
    tokenizer = ds_handle.tokenizer

    # query datasets ---
    query_train_ds = ds_handle.get_dataset(
        query_df_train, is_query=True, is_train=True, rng=rng
    ).sort("input_length")
    query_valid_ds = ds_handle.get_dataset(query_df_valid, is_query=True).sort(
        "input_length"
    )

    # content datasets ---
    content_train_ds = ds_handle.get_dataset(content_df_train, is_query=False).sort(
        "input_length"
    )
    content_comp_ds = ds_handle.get_dataset(content_df_comp, is_query=False).sort(
        "input_length"
    )

    # retrieval dataset ---
    retrieval_ds = Dataset.from_pandas(corr_df_train)

    # manage ids ---
    id_tracker = IDTracker(
        query_train_ids=query_train_ds["query_id"],
        query_valid_ids=query_valid_ds["query_id"],
        content_train_ids=content_train_ds["content_id"],
        content_comp_ids=content_comp_ds["content_id"],
    )

    # ------- data collators ----------------------------------------------------------------------#
    text_collator = TextCollator(tokenizer=tokenizer)

    # load hard negatives
    print_line()
    if cfg.train_params.load_hard_negatives:
        print(f"Loading hard negatives from {cfg.train_params.hard_negative_dataset}")
        hn_dir = download_kaggle_dataset(cfg.train_params.hard_negative_dataset)
        print(f"hn_dir: {hn_dir}")

        with open(os.path.join(hn_dir, cfg.train_params.hard_negative_file), "r") as f:
            hard_negatives = json.load(f)
        for k, v in hard_negatives.items():
            hard_negatives[k] = v[: cfg.train_params.negative_depth_end]

        # load teacher logits
        with open(os.path.join(hn_dir, cfg.train_params.teacher_logits_file), "r") as f:
            teacher_logits = json.load(f)
        online_hard_negatives = dict()
    else:
        print("Not loading hard negatives")
        hard_negatives = dict()
        teacher_logits = dict()
        online_hard_negatives = dict()
    print_line()

    kwargs = dict(
        cfg=cfg,
        query_ds=query_train_ds,
        content_ds=content_train_ds,
        content2query=content2query_train,
        negative_map=hard_negatives,
        online_negative_map=online_hard_negatives,
        teacher_logits=teacher_logits,
        subject2queries=subject2query_train,
        query2subject=query2subject_train,
        query2content=query2content_train,
    )

    retriever_collator = RetrieverDataCollator(tokenizer=tokenizer, kwargs=kwargs)

    # ------- data loaders ------------------------------------------------------------------------#
    query_train_dl = DataLoader(
        query_train_ds,
        batch_size=cfg.train_params.query_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    query_valid_dl = DataLoader(
        query_valid_ds,
        batch_size=cfg.train_params.query_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    content_train_dl = DataLoader(
        content_train_ds,
        batch_size=cfg.train_params.content_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    content_comp_dl = DataLoader(
        content_comp_ds,
        batch_size=cfg.train_params.content_bs,
        shuffle=False,
        collate_fn=text_collator,
    )

    retrieval_dl = DataLoader(
        retrieval_ds,
        batch_size=cfg.train_params.retriever_bs,
        shuffle=True,
        collate_fn=retriever_collator,
        num_workers=1,
        drop_last=True,
    )

    # --- show batch ------------------------------------------------------------------------------#
    print_line()
    print("showing a batch...")
    for b in retrieval_dl:
        break
    show_batch(b, tokenizer, print_fn=print)

    print_line()
    print("showing first valid batch...")
    for b in query_valid_dl:
        break
    show_batch_fs(b, tokenizer, print_fn=print)
    print_line()

    # ------- Config ------------------------------------------------------------------------------#
    print_line()
    print("Config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))
    print_line()

    # ------- Model -------------------------------------------------------------------------------#
    print_line()
    print("Loading model....")
    base_model = get_base_model(cfg)
    model = BiEncoderModel(cfg, base_model, device)
    model = model.to(device)

    if cfg.model.gradient_checkpointing:
        print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if cfg.model.compile:
        print("compiling model")
        model = torch.compile(model)
    print("Model loaded")
    print_line()

    # ------- Optimizer ---------------------------------------------------------------------------#
    optimizer = get_optimizer(cfg, model, print_fn=print)

    # ------- Scheduler ---------------------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(retrieval_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ------- training setup ----------------------------------------------------------------------#
    best_lb = -1.0
    patience_tracker = 0
    current_iteration = 0
    progress = 0.0

    start_time = time.time()
    progress_bar = None

    # initial evaluation --------------------------------------------------------------------------#
    if cfg.train_params.eval_at_start:
        model.eval()
        eval_response = run_evaluation(
            cfg, model, query_valid_dl, content_comp_dl, corr_df_valid, id_tracker
        )

    for epoch in range(num_epochs):
        if progress_bar:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        print_line()
        print(f"Current epoch: {epoch+1}/{num_epochs}")
        print_line()

        # Training ------
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(retrieval_dl):
            temperature = torch.tensor(get_temperature(cfg, progress)).to(device)
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            outputs = model(**batch, temperature=temperature)
            loss = outputs.loss
            loss.backward()

            if (step + 1) % grad_accumulation_steps == 0 or (
                step + 1
            ) == num_update_steps_per_epoch:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optimizer.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_description(
                f"STEP: {step+1:5}/{num_update_steps_per_epoch:5}. "
                f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                f"LR: {get_lr(optimizer):.4f}. "
                f"Loss: {loss.item():.4f}. "
            )

            progress_bar.update(1)
            current_iteration += 1
            progress = current_iteration / num_training_steps

            # Optionally log to wandb if needed
            if cfg.use_wandb:
                import wandb

                wandb.log({"train_loss": round(loss.item(), 5)}, step=current_iteration)
                wandb.log({"lr": get_lr(optimizer)}, step=current_iteration)
                wandb.log(
                    {"total_grad_l2_norm": round(grad_norm.item(), 5)},
                    step=current_iteration,
                )
                wandb.log({"temperature": temperature.item()}, step=current_iteration)
                wandb.log(
                    {"nce_loss": round(outputs.nce_loss.item(), 5)},
                    step=current_iteration,
                )
                wandb.log(
                    {"distill_loss": round(outputs.distill_loss.item(), 5)},
                    step=current_iteration,
                )

        # Evaluation -----
        print_line()
        et = as_minutes(time.time() - start_time)
        print(
            f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
        )

        model.eval()
        eval_response = run_evaluation(
            cfg, model, query_valid_dl, content_comp_dl, corr_df_valid, id_tracker
        )

        lb, scores_dict, result_df, oof_df = (
            eval_response["lb"],
            eval_response["scores"],
            eval_response["result_df"],
            eval_response["oof_df"],
        )
        print_line()

        # best scores and saving -----
        is_best = False
        if lb >= best_lb:
            best_lb = lb
            is_best = True
            patience_tracker = 0
            best_dict = dict()
            for k, v in scores_dict.items():
                best_dict[f"{k}_at_best"] = v
        else:
            patience_tracker += 1

        if is_best:
            oof_df.to_csv(
                os.path.join(cfg.outputs.model_dir, "oof_df_best.csv"), index=False
            )
            result_df.to_csv(
                os.path.join(cfg.outputs.model_dir, "result_df_best.csv"), index=False
            )
        else:
            print(
                f">>> patience reached {patience_tracker}/{cfg.train_params.patience}"
            )
            print(f">>> current best score: {round(best_lb, 4)}")

        oof_df.to_csv(
            os.path.join(cfg.outputs.model_dir, "oof_df_last.csv"), index=False
        )
        result_df.to_csv(
            os.path.join(cfg.outputs.model_dir, "result_df_last.csv"), index=False
        )

        # save checkpoint ---
        torch.save(model.state_dict(), os.path.join(cfg.outputs.model_dir, "model.pt"))
        tokenizer.save_pretrained(cfg.outputs.model_dir)

        # logging ----
        if cfg.use_wandb:
            import wandb

            wandb.log({"lb": lb}, step=current_iteration)
            wandb.log({"best_lb": best_lb}, step=current_iteration)
            wandb.log({"seen_lb": scores_dict["seen_lb"]}, step=current_iteration)
            wandb.log({"unseen_lb": scores_dict["unseen_lb"]}, step=current_iteration)
            for k, v in scores_dict.items():
                wandb.log({k: round(v, 4)}, step=current_iteration)

        # mine hard negatives & refresh dataloader ---
        if (cfg.train_params.iterative_hard_negatives) & (
            cfg.train_params.num_hard_negatives > 0
        ):
            if epoch == num_epochs - 1:
                continue  # no need to mine hard negatives at the last epoch
            if epoch >= cfg.train_params.iterative_hard_negatives_trigger:
                print(f">>> Mining hard negatives at epoch {epoch+1}")
                online_hard_negatives = mine_hard_negatives_online(
                    cfg,
                    model,
                    query_train_dl,
                    content_train_dl,
                    corr_df_train,
                    id_tracker,
                )
                kwargs = dict(
                    cfg=cfg,
                    query_ds=query_train_ds,
                    content_ds=content_train_ds,
                    content2query=content2query_train,
                    negative_map=hard_negatives,
                    online_negative_map=online_hard_negatives,
                    teacher_logits=teacher_logits,
                    subject2queries=subject2query_train,
                    query2subject=query2subject_train,
                    query2content=query2content_train,
                )
                retriever_collator = RetrieverDataCollator(
                    tokenizer=tokenizer, kwargs=kwargs
                )
                retrieval_dl = DataLoader(
                    retrieval_ds,
                    batch_size=cfg.train_params.retriever_bs,
                    shuffle=True,
                    collate_fn=retriever_collator,
                    num_workers=1,
                    drop_last=True,
                )

        # -- post eval
        model.train()
        torch.cuda.empty_cache()
        print_line()

        # early stopping ----
        if patience_tracker >= cfg.train_params.patience:
            return


if __name__ == "__main__":
    run_training()
# ...existing code...
