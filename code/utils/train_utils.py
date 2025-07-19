import logging
import math
import os

import shutil
import uuid
from collections import defaultdict

import datasets
import kagglehub
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

logger = get_logger(__name__)


def generate_random_string():
    return str(uuid.uuid4())


def print_line(logger=None):
    prefix, unit, suffix = "#", "~~", "#"
    if logger is None:
        print(prefix + unit * 50 + suffix)
    else:
        logger.print(prefix + unit * 50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm%ds" % (m, s)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"] * 1e6


class AverageMeter(object):
    """Computes and stores the average and current value using exponential smoothing.
    Maintains similar structure to the original version with additional smoothing functionality.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Smoothing factor, adjustable according to the desired responsiveness
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # Initialized as 0; assumes first update sets to first value directly if required
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == n:  # First update
            self.avg = val
        else:
            self.avg = (
                self.alpha * val + (1 - self.alpha) * self.avg
            )  # Apply exponential smoothing


def save_checkpoint(cfg, state, is_best):
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    name = "eedi"

    filename = f"{cfg.outputs.model_dir}/{name}_last.pth.tar"
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f"{cfg.outputs.model_dir}/{name}_best.pth.tar")


class EMA:
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def enable_cuda_optimizations():
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
    torch.backends.cudnn.allow_tf32 = True


def setup_training_run(cfg):
    """set up training run

    Args:
        cfg: config for the training run
    """

    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            log_with="wandb",
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.wandb.run_name}},
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
        )
    accelerator.print(f"using wandb: {cfg.use_wandb}")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    if cfg.enable_cuda_optimizations:
        enable_cuda_optimizations()
    return accelerator


def get_custom_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 10% of it,
    following a cosine curve, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of times the learning rate will decay to 10% of the maximum learning rate. Default: 0.5 (half a cycle).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale to decay to 10% of the max lr
        decay_target = 0.1  # Decay to 10% of the max lr
        decay_factor = (1 - decay_target) * cosine_decay + decay_target

        return decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def log_gradient_norms(accelerator, model, step):
    grad_l2_norm = 0.0
    param_logs = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = torch.norm(param.grad, 2).item()
            # param_logs[f"{name}_grad_l2_norm"] = norm
            grad_l2_norm += norm

    # Log aggregate norms and individual parameter norms
    accelerator.log(
        {
            "step": step,
            "total_grad_l2_norm": grad_l2_norm,
            **param_logs,  # Unpack and log individual parameter norms
        }
    )


def is_nan(x):
    return x != x


def eedi_process_df(df, debug=False):
    df = df.copy()
    df = df.rename(columns={"QuestionId": "query_id"})
    grouped = df.groupby("query_id")

    question_dict = {}
    for question_id, group in grouped:
        question_data = group.to_dict(orient="records")[0]
        del question_data["query_id"]
        question_dict[question_id] = question_data

    all_questions = list(question_dict.keys())

    queries = []
    query2content = defaultdict(list)
    content2query = defaultdict(list)

    # if debug:
    #     import pdb

    #     pdb.set_trace()

    # ---
    for qid in all_questions:
        info = question_dict[qid]

        for answer_key in ["A", "B", "C", "D"]:
            if info["CorrectAnswer"] == answer_key:
                continue
            this_example = dict()
            this_key = f"{qid}_{answer_key}"
            this_example["query_id"] = this_key

            if is_nan(info[f"Misconception{answer_key}Id"]):
                continue

            mid = int(info[f"Misconception{answer_key}Id"])
            query2content[this_key].append(mid)
            content2query[mid].append(this_key)

            # ---
            for col in ["SubjectId", "SubjectName", "ConstructName", "QuestionText"]:
                this_example[col] = info[col]

            this_example["CorrectAnswerText"] = info[
                f"Answer{info['CorrectAnswer']}Text"
            ]
            this_example["InCorrectAnswerText"] = info[f"Answer{answer_key}Text"]
            this_example["AllOptionText"] = "\n- ".join(
                [info[f"Answer{x}Text"] for x in ["A", "B", "C", "D"]]
            )
            this_example["AllOptionText"] = f"\n- {this_example['AllOptionText']}"
            queries.append(this_example)
    # --
    query_df = pd.DataFrame(queries)
    corr_df = (
        pd.Series(query2content)
        .reset_index()
        .rename(columns={"index": "query_id", 0: "content_id"})
    )
    corr_df["content_id"] = corr_df["content_id"].apply(lambda x: x[0])

    query_df = query_df.reset_index(drop=True)

    return query_df, corr_df, content2query


def download_kaggle_dataset(handle: str):
    try:
        return kagglehub.dataset_download(handle)
    except ValueError:
        try:
            return kagglehub.competition_download(handle)
        except Exception:
            # return f"/root/.cache/kagglehub/competitions/{handle}"
            return "../data"
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def train_valid_split(cfg, df):
    fold_dir = download_kaggle_dataset(cfg.dataset.fold_dataset)
    fold_df = pd.read_parquet(os.path.join(fold_dir, "folds.parquet")).rename(
        columns={"QuestionId": "query_id"}
    )
    df = pd.merge(df, fold_df, on="query_id", how="left")
    df["kfold"] = df["kfold"].fillna(99).astype(int)
    print(f"# of folds: {df['kfold'].nunique()}")
    print("Fold distribution:")
    print(df["kfold"].value_counts())

    if cfg.full_fit:
        train_df = df.copy()
    else:
        train_df = df[df["kfold"] != cfg.fold].copy()
    valid_df = df[df["kfold"] == cfg.fold].copy()

    train_df = train_df.drop(columns=["kfold"]).reset_index(drop=True)
    valid_df = valid_df.drop(columns=["kfold"]).reset_index(drop=True)

    print(f"# of train: {train_df.shape[0]}")
    print(f"# of valid: {valid_df.shape[0]}")

    return train_df, valid_df


def add_fs_examples(df, content2query, query2example, rng, is_train=False, k_shot=2):
    def _add_examples(row):
        qids = content2query[
            int(row["content_id"])
        ]  # content2query is a defaultdict(list)
        qids = [qid for qid in qids if qid != row["query_id"]]

        if is_train:  # for training use random number of few shot examples
            n = rng.randint(0, k_shot)
        else:
            n = k_shot

        if len(qids) == 0 or n == 0:
            return ""

        qids = rng.sample(qids, k=min(n, len(qids)))
        examples = [query2example[qid] for qid in qids]
        fs = "\n--\n".join(examples)
        return fs

    df["examples"] = df.apply(_add_examples, axis=1)
    return df


def add_fs_examples_listwise(df, content2query, query2example, rng):
    def _add_examples(row):
        cids = row["content_ids"]

        selected_qids = []
        for cid in cids:
            cid = int(cid)
            qids = content2query[cid]
            qids = [qid for qid in qids if qid != row["query_id"]]
            if len(qids) > 0:
                selected_qids.append(rng.choice(qids))

        if len(selected_qids) == 0:
            return ""

        n = rng.randint(0, 4)
        selected_qids = rng.sample(selected_qids, k=min(n, len(selected_qids)))
        selected_qids = sorted(selected_qids)  # for prefix cache
        examples = [query2example[qid] for qid in selected_qids]
        fs = "\n--\n".join(examples)
        return fs

    df["examples"] = df.apply(_add_examples, axis=1)
    return df
