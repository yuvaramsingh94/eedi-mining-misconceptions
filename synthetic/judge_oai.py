import argparse
import glob
import json
import os
import random
import time

import kagglehub
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

client = OpenAI()


def is_nan(x):
    return x != x


def get_pair_df(df, name2id):
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
    for qid in all_questions:
        info = question_dict[qid]

        for answer_key in ["A", "B", "C", "D"]:
            if info["CorrectAnswer"] == answer_key:
                continue

            this_example = dict()
            this_key = f"{qid}_{answer_key}"
            this_example["query_id"] = this_key

            if (is_nan(info[f"Misconception{answer_key}Name"])) or (info[f"Misconception{answer_key}Name"] == ""):
                continue

            this_example["content_id"] = name2id.get(info[f"Misconception{answer_key}Name"], -1)
            this_example["MisconceptionName"] = info[f"Misconception{answer_key}Name"]

            for col in ["SubjectName", "ConstructName", "QuestionText"]:
                this_example[col] = info[col]

            this_example["CorrectAnswerText"] = info[f"Answer{info['CorrectAnswer']}Text"]
            this_example["InCorrectAnswerText"] = info[f"Answer{answer_key}Text"]

            queries.append(this_example)
    # --
    pair_df = pd.DataFrame(queries)
    pair_df = pair_df.reset_index(drop=True)
    return pair_df


def get_content_id_for_eval(df, cfg):
    all_ids = set(df["content_id"].unique().tolist())

    existing_ids = []
    for dir_name in cfg.existing_dirs:
        fns = glob.glob(os.path.join(dir_name, "*.json"))
        existing_ids.extend([int(os.path.basename(fn).split(".")[0]) for fn in fns])
    existing_ids = list(set(existing_ids))
    remaining_ids = all_ids - set(existing_ids)
    remaining_ids = list(remaining_ids)

    print(f"# of remaining ids: {len(remaining_ids)}")
    print(f"# of existing ids: {len(existing_ids)}")
    print(f"# of all ids: {len(all_ids)}")

    if len(remaining_ids) == 0:
        raise ValueError("No remaining ids to evaluate")

    return random.choice(remaining_ids)


system_prompt = """You are an expert mathematics educator specializing in diagnostic assessment.
Your task is to evaluate whether an incorrect answer stems from a specific misconception.
"""

TEMPLATE = """You will analyze how well an incorrect answer reflects a suspected misconception in a mathematics problem. Your goal is to determine whether there is a clear, logical connection between the misconception and the wrong answer.

Here is the problem with both correct and incorrect answers. The suspected misconception is also provided:
<problem>
{PROBLEM_DATA}
</problem>

First, analyze the problem in your scratchpad:
<scratchpad>
1. Solve the problem independently to verify the correct answer
2. Examine how someone holding the suspected misconception would approach the problem
3. Trace the logical path from misconception to incorrect answer
4. Identify any gaps or inconsistencies in this connection
</scratchpad>

Then provide your evaluation using this format:
<evaluation>
1. Brief explanation of how the misconception could lead to the wrong answer
2. Score from 0-10 based on these criteria:
   - 10: Perfect alignment - wrong answer is direct result of misconception
   - 8-9: Strong alignment - clear logical path from misconception to answer
   - 5-7: Moderate alignment - connection exists but has some gaps
   - 1-4: Weak alignment - connection is unclear or requires assumptions
   - 0: No alignment - misconception does not explain wrong answer
</evaluation>

Format your response as:
Explanation: [your explanation]
Score: [0-10]

Important guidelines:
- Focus solely on the logical connection between misconception and wrong answer
- Do not speculate about other possible misconceptions
- Be specific about how the misconception leads to the error
- Flag and deduct scores if any assumptions are required to connect misconception to answer
- Consider whether a student with this misconception would consistently arrive at this wrong answer"""


def format_data(row):
    d = f"""# Question: {row['QuestionText']}
# Correct Answer: {row['CorrectAnswerText']}
# Incorrect Answer: {row['InCorrectAnswerText']}
# Suspected Misconception: {row['MisconceptionName']}
"""
    return d


def get_prompt(example):
    return TEMPLATE.format(PROBLEM_DATA=format_data(example))


class MathMCQEvaluation(BaseModel):
    """Evaluation of a math MCQ"""

    Explanation: str = Field(description="Short explanation of what caused the incorrect answer (1-2 sentences)")
    Score: int = Field(description="0-10 score describing the affinity between the incorrect answer and the misconception")


def upload_dataset(handle: str, local_dir: str):
    kagglehub.dataset_upload(handle, local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = OmegaConf.load(f)
    os.makedirs(cfg.output_dir, exist_ok=True)

    collection_dir = os.path.join(cfg.output_dir, "bank")
    stats_dir = os.path.join(cfg.output_dir, "stats")

    os.makedirs(collection_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # load data ---------------------------------------------------------------------------------#
    comp_dir = kagglehub.competition_download(cfg.comp_dataset)
    misconception_df = pd.read_csv(os.path.join(comp_dir, "misconception_mapping.csv"))
    id2name = dict(zip(misconception_df["MisconceptionId"], misconception_df["MisconceptionName"]))
    name2id = dict(zip(misconception_df["MisconceptionName"], misconception_df["MisconceptionId"]))

    json_files = glob.glob(os.path.join(cfg.input_dir, "*.json"))
    mcq_df = pd.DataFrame([json.load(open(f)) for f in json_files])

    mcq_df = get_pair_df(mcq_df, name2id)
    print(f"shape of pair df: {mcq_df.shape}")

    # create pair data

    # evaluate examples -------------------------------------------------------------------------#
    pbar = tqdm(range(cfg.batch_size))
    quality_threshold = 9
    min_good_examples = 3
    max_try = 12

    for _ in range(cfg.batch_size):
        content_id = get_content_id_for_eval(mcq_df, cfg)
        focused_df = mcq_df[mcq_df["content_id"] == content_id].copy()
        print("--------------------------------------")
        mname = focused_df["MisconceptionName"].unique()[0]
        print(f"Running for -> {mname}...")
        print(f"# of examples: {len(focused_df)}")

        print("--------------------------------------")

        # iterate over all examples in the focused_df
        evals = []
        n_good = 0
        try_count = 0

        for _, example in focused_df.iterrows():
            try_count += 1
            if try_count > max_try:
                break

            prompt = get_prompt(example)
            example = example.to_dict()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            completion = client.beta.chat.completions.parse(model=cfg.model, messages=messages, response_format=MathMCQEvaluation, max_tokens=512)

            response = completion.choices[0].message
            # print(completion.usage)

            if response.parsed:
                scores = response.parsed.model_dump()
            else:
                print(response.refusal)
                continue

            val = scores["Score"]
            evals.append(val)
            if val > quality_threshold:
                n_good += 1

            for key, value in scores.items():
                example[key] = value

            # save the example
            example_id = f"{example['query_id']}_{example['content_id']}"
            with open(os.path.join(cfg.output_dir, f"{example_id}.json"), "w") as f:
                json.dump(example, f)

            print(f"# good: {n_good} | Ave: {np.mean(evals):.2f} | Evals: {evals}")
            print("~~" * 30)

            if n_good >= min_good_examples:
                # save stats
                stats = {
                    "content_id": content_id,
                    "misconception": mname,
                    "evals": evals,
                    "ave_score": np.mean(evals),
                    "std_score": np.std(evals),
                }
                with open(os.path.join(cfg.output_dir, "stats", f"{content_id}.json"), "w") as f:
                    json.dump(stats, f)
                break

        # write stats at the end of the batch
        stats = {
            "content_id": content_id,
            "misconception": mname,
            "evals": evals,
            "ave_score": np.mean(evals),
            "std_score": np.std(evals),
        }
        with open(os.path.join(cfg.output_dir, "stats", f"{content_id}.json"), "w") as f:
            json.dump(stats, f)

        print("##" * 40)
        print("\n\n\n")

        time.sleep(cfg.sleep_time)
        pbar.update(1)
    pbar.close()

    # upload to kaggle -------------------------------------------------------------------------#
    if cfg.upload_to_kaggle:
        json_files = glob.glob(os.path.join(cfg.output_dir, "*.json"))
        mcq_df = pd.DataFrame([json.load(open(f)) for f in json_files])
        mcq_df.to_csv(os.path.join(collection_dir, "synthetic_evals.csv"), index=False)
        upload_dataset(cfg.upload_dataset, collection_dir)
