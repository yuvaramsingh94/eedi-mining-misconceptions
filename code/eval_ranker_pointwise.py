import json
import os
import random
from itertools import chain

import hydra
import kagglehub
import numpy as np
import pandas as pd
import torch
import vllm
from omegaconf import OmegaConf

# local imports ---
from ranker_pointwise.ranker_dataset import RankerDataset
from sklearn import metrics
from utils.metric_utils import compute_retrieval_metrics, mapk
from utils.train_utils import eedi_process_df, train_valid_split

torch._dynamo.config.optimize_ddp = False


def format_example(row):
    example = f"Question: {row['QuestionText']}\nAnswer:{row['CorrectAnswerText']}\nMisconception Answer: {row['InCorrectAnswerText']}"
    return example


def add_fs_examples_for_eval(df, content2query, query2example, rng, k_shot=2):
    cache = {}

    def _add_examples(row):
        cid = row["content_id"]
        if cid in cache:
            return cache[cid]
        else:
            qids = content2query[cid]
            qids = [qid for qid in qids if qid != row["query_id"]]
            if len(qids) == 0:
                cache[cid] = ""
                return ""

            qids = rng.sample(qids, k=min(k_shot, len(qids)))
            examples = [query2example[qid] for qid in qids]
            fs = "\n--\n".join(examples)
            cache[cid] = fs
            return fs

    df["examples"] = df.apply(_add_examples, axis=1)
    return df


def sort_by_scores(pred_ids, scores):
    keep_idxs = np.argsort(-np.array(scores)).tolist()
    ret_ids = [pred_ids[idx] for idx in keep_idxs]
    ret_scores = [scores[idx] for idx in keep_idxs]
    return {"sorted_ids": ret_ids, "sorted_scores": ret_scores}


def run_evaluation(result_df, qa2mc):
    agg_df = result_df.groupby("QuestionId_Answer")["MisconceptionId"].agg(list).reset_index()
    score_agg_df = result_df.groupby("QuestionId_Answer")["score"].agg(list).reset_index()
    agg_df = pd.merge(agg_df, score_agg_df, on="QuestionId_Answer", how="left")
    agg_df["truth"] = agg_df["QuestionId_Answer"].map(qa2mc)
    agg_df["truth"] = agg_df["truth"].apply(lambda x: [str(x)])
    agg_df["y"] = agg_df.apply(lambda x: [1 if y in x["truth"] else 0 for y in x["MisconceptionId"]], axis=1)

    truths = list(chain(*agg_df["y"].values))
    preds = list(chain(*agg_df["score"].values))
    fpr, tpr, thresholds = metrics.roc_curve(truths, preds)
    ranker_auc = metrics.auc(fpr, tpr)

    # sort by score and keep top 25, remember score is a list
    agg_df["topk_info"] = agg_df.apply(lambda x: sort_by_scores(x["MisconceptionId"], x["score"]), axis=1)
    agg_df["MisconceptionId"] = agg_df["topk_info"].apply(lambda x: x["sorted_ids"])
    agg_df["score"] = agg_df["topk_info"].apply(lambda x: x["sorted_scores"])

    lb = mapk(agg_df["truth"].values, agg_df["MisconceptionId"].values, k=25)

    # get recall scores ---
    recall_dict = dict()
    cutoffs = [1, 2, 4, 8, 16, 25, 32]
    for cutoff in cutoffs:
        cdf = agg_df.copy()
        cdf["MisconceptionId"] = cdf["MisconceptionId"].apply(lambda x: x[:cutoff])
        m = compute_retrieval_metrics(cdf["truth"].values, cdf["MisconceptionId"].values)
        recall_dict[f"recall@{cutoff}"] = m["recall_score"]

    print("--------------------------------")
    for pt in cutoffs:
        print(f">>> Current Recall@{pt} = {round(recall_dict[f'recall@{pt}'], 4)}")
    print("--------------------------------")

    # compute oof dataframe ---
    oof_df = agg_df.copy()
    oof_df = oof_df[["QuestionId_Answer", "MisconceptionId", "score"]].copy()
    oof_df = oof_df.rename(columns={"score": "pred_scores"})

    to_return = {
        "lb": lb,
        "auc": ranker_auc,
        "result_df": result_df,
        "oof_df": oof_df,
        "recall_dict": recall_dict,
    }

    return to_return


@hydra.main(version_base=None, config_path="../conf/ranker_pointwise", config_name="conf_pointwise_eval")
def main(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rng = random.Random(cfg.seed)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        print(prefix + unit * 50 + suffix)

    print_line()
    print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    # download data in main process
    input_dir = kagglehub.dataset_download(cfg.dataset.input_dataset)
    try:
        comp_dir = kagglehub.competition_download(cfg.dataset.comp_dataset)
    except Exception:
        comp_dir = kagglehub.dataset_download(cfg.dataset.comp_dataset)

    if cfg.infer_on_train:
        print("Warning: using train.parquet as valid.parquet for inference")
        valid_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))

    else:
        valid_df = pd.read_parquet(os.path.join(input_dir, "valid.parquet"))

    # For few-shot examples
    comp_df = pd.read_csv(os.path.join(comp_dir, "train.csv")).rename(columns={"QuestionId": "query_id"})
    fs_df, _ = train_valid_split(cfg, comp_df)
    query_df, _, content2query = eedi_process_df(fs_df)

    ### CHANGED HERE ###
    label_dir = kagglehub.dataset_download(cfg.dataset.label_dataset)
    label_df = pd.read_csv(os.path.join(label_dir, "train.csv"))
    _, corr_df_all, _ = eedi_process_df(label_df)
    ### CHANGED HERE ###

    query_df["demo"] = query_df.apply(format_example, axis=1)
    query2example = dict(zip(query_df["query_id"], query_df["demo"]))

    # valid_df = add_fs_examples(valid_df, content2query, query2example, rng, is_train=False, k_shot=cfg.k_shot)
    valid_df = add_fs_examples_for_eval(valid_df, content2query, query2example, rng, k_shot=cfg.k_shot)
    # valid_df = valid_df.sort_values(by="content_id").reset_index(drop=True)
    valid_df = valid_df.sort_values(by="query_id").reset_index(drop=True)

    qa2mc = dict(zip(corr_df_all["query_id"], corr_df_all["content_id"]))  # label map (qa_id -> mc_id)

    print(f"shape of validation data: {valid_df.shape}")
    print_line()

    dataset_creator = RankerDataset(cfg)

    valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer
    # valid_ds = valid_ds.sort("length", reverse=True)

    yes_tok_id = tokenizer("Yes", add_special_tokens=False)["input_ids"][-1]
    no_tok_id = tokenizer("No", add_special_tokens=False)["input_ids"][-1]

    print(f">> EediRanker: Yes token id: {yes_tok_id}")
    print(f">> EediRanker: No token id: {no_tok_id}")

    valid_qa_ids = valid_ds["query_id"]
    valid_mc_ids = valid_ds["content_id"]

    valid_ds = valid_ds.map(lambda example: {"prompt": tokenizer.decode(example["input_ids"], skip_special_tokens=False)})
    prompts = valid_ds["prompt"]

    print(f"# of requests: {len(prompts)}")
    print(f"Example:\n\n{prompts[0]}")
    print("data preparation done...")
    print_line()

    llm = vllm.LLM(
        cfg.model.backbone_path,
        tensor_parallel_size=cfg.num_gpus,  # 2,  # 2
        # quantization="awq",
        gpu_memory_utilization=0.99,
        trust_remote_code=True,
        dtype="bfloat16",
        # dtype="bfloat16",
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    # ------- Prepare -------------------------------------------------------------------#
    sampling_params = vllm.SamplingParams(n=1, top_p=0.8, logprobs=20, max_tokens=1, temperature=0.0, skip_special_tokens=False)
    responses = llm.generate(prompts, sampling_params, use_tqdm=True)

    print("inference done...")
    print_line()

    QuestionId_Answer = []
    MisconceptionId = []
    scores = []

    for qid, cid, response in zip(valid_qa_ids, valid_mc_ids, responses):
        logprob_dict = response.outputs[0].logprobs[0]

        top_tok_ids = set(list(logprob_dict.keys()))
        if len(top_tok_ids.intersection(set([yes_tok_id, no_tok_id]))) == 0:
            print(f"Bad Output for {qid} - {cid}")
            continue

        yes_logit, no_logit = -10.0, -10.0

        if yes_tok_id in logprob_dict:
            yes_logit = logprob_dict[yes_tok_id].logprob

        if no_tok_id in logprob_dict:
            no_logit = logprob_dict[no_tok_id].logprob

        score = yes_logit - no_logit

        QuestionId_Answer.append(qid)
        MisconceptionId.append(cid)
        scores.append(score)

    result_df = pd.DataFrame()
    result_df["QuestionId_Answer"] = QuestionId_Answer
    result_df["MisconceptionId"] = MisconceptionId
    result_df["score"] = scores

    eval_response = run_evaluation(result_df, qa2mc)

    lb = eval_response["lb"]
    ranker_auc = eval_response["auc"]

    result_df = eval_response["result_df"]
    oof_df = eval_response["oof_df"]

    os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "ext_oof_df_best.csv"), index=False)
    result_df.to_csv(os.path.join(cfg.outputs.model_dir, "ext_result_df_best.csv"), index=False)

    print(f">>> Current LB (MAP@25) = {round(lb, 4)}")
    print(f">>> AUC = {round(ranker_auc, 4)}")


if __name__ == "__main__":
    main()
