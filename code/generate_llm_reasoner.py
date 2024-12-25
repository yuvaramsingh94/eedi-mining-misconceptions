# Credit: Mark Tenenholtz
import argparse
import gc
import os
import random

import kagglehub
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # noqa

sp = "Analyze the incorrect answer to detect flaws in the student's reasoning."


def get_tokenizer(backbone_path):
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, add_eos_token=True)

    if tokenizer.eos_token == "":
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.bos_token = "<|im_start|>"
    tokenizer.padding_side = "left"
    return tokenizer


def main(args):
    seed = random.randint(100, 10000)
    print(f"setting seed to: {seed}")
    random.seed(seed)

    input_dir = kagglehub.dataset_download(args.input_dataset)
    df = pd.read_csv(os.path.join(input_dir, "train.csv")).rename(columns={"QuestionId": "query_id"})
    df["QuestionId"] = df["query_id"].apply(lambda x: x.split("_")[0])
    df["QuestionId"] = df["QuestionId"].astype(int)

    # ---
    fold_dir = kagglehub.dataset_download("conjuring92/eedi-five-folds")
    fold_df = pd.read_parquet(os.path.join(fold_dir, "folds.parquet"))
    df = pd.merge(df, fold_df, on="QuestionId", how="left")
    df["kfold"] = df["kfold"].fillna(99).astype(int)
    print(f"# of folds: {df['kfold'].nunique()}")
    print("Fold distribution:")
    print(df["kfold"].value_counts())

    focus_fold = args.fold  # 0  # 0  # 99
    df = df[df["kfold"] == focus_fold].copy()
    df = df.reset_index(drop=True)
    print(f"# of examples for fold {focus_fold}: {len(df)}")

    # df = df.head(100)

    # ---

    ds = Dataset.from_pandas(df)
    print(f"Number of examples: {len(ds)}")

    # ds = ds.shuffle(seed=seed)
    # ds = ds.select(range(args.num_examples))
    query_ids = ds["query_id"]
    print(f"# of query_ids: {len(query_ids)}")

    print("==" * 50)
    print(f"Generating for model: {args.model}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=768,
        gpu_memory_utilization=0.999,
        enable_prefix_caching=True,
    )
    # tokenizer = get_tokenizer(args.model)

    prompts = []
    for example in ds:
        question = example["QuestionText"]
        correct_answer = example["CorrectAnswerText"]
        incorrect_answer = example["InCorrectAnswerText"]

        user_message = f"Question: {question}\nCorrect Answer: {correct_answer}\nIncorrect Answer: {incorrect_answer}"

        text = f"{sp}\n\nQuery: {user_message}\nAnswer:\n"
        prompts.append(text)

    # print a few prompts
    for p in prompts[:5]:
        print(p)
        print("-" * 100)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.0, max_tokens=384)

    chunk_size = 32
    for i in range(0, len(prompts), chunk_size):
        results = []
        generated_texts = []
        all_prompts = []

        chunk_prompts = prompts[i : i + chunk_size]
        chunk_query_ids = query_ids[i : i + chunk_size]

        print(f"Processing chunk {i//chunk_size + 1} of {(len(prompts)-1)//chunk_size + 1}")

        generations = llm.generate(chunk_prompts, sampling_params=sampling_params)

        for output in generations:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            all_prompts.append(prompt)

            full_text = f"{prompt}{generated_text}"
            results.append(full_text)
            generated_texts.append(generated_text)

        # Save intermediate results
        df = pd.DataFrame()
        df["query_id"] = chunk_query_ids
        df["prompt"] = all_prompts
        df["generated_text"] = generated_texts
        df["full_text"] = results

        try:
            intermediate_path = os.path.join(args.save_dir, f"generated_{seed}_fold{focus_fold}_chunk_{i//chunk_size}.parquet")
            df.to_parquet(intermediate_path)
            print(f"Saved intermediate results to {intermediate_path}")
        except Exception as e:
            print(f"Error saving intermediate results: {e}")

    del llm
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dataset", type=str, default="conjuring92/eedi-pair-for-cot-comp-for-infer")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--save_dir", type=str, default="../data/vllm_cot_generated")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--fold", type=int, default=99)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)

# python ./eedi/code/vllm_generate.py --model ../models/qwen25_math_7b_cot --tensor_parallel_size 2 --fold 99

# export MKL_THREADING_LAYER=GNU
# export MKL_SERVICE_FORCE_INTEL=1
