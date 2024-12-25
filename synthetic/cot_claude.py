import argparse
import glob
import json
import os
import random
import time

import kagglehub
import numpy as np
import pandas as pd
from claudette import Client
from fastcore.basics import BasicRepr, store_attr
from omegaconf import OmegaConf
from tqdm.auto import tqdm


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


system_prompt = """You are an expert mathematics educator specializing in diagnostic assessment."""

TEMPLATE = """You will analyze a student's incorrect answer to identify the specific reasoning flaw that led to their error. 
Your goal is to explain precisely how their misconception caused them to arrive at the wrong answer.

Here is the problem information:
<problem_data>
{PROBLEM_DATA}
</problem_data>

Here are related misconceptions that are similar but do not explain this specific error as precisely:
<related_misconceptions>
{RELATED_MISCONCEPTIONS}
</related_misconceptions>

First, examine all components of the problem carefully:
1. The problem statement and question asked
2. The correct answer and solution method
3. The student's incorrect answer
4. The primary misconception given
5. The related misconceptions that should be distinguished from the primary one

Then, reconstruct the student's likely thought process:
- Identify the exact point where their reasoning diverged from the correct solution path
- Note which specific mathematical operations or concepts they misapplied
- Connect their error directly to the stated primary misconception
- Verify that this explanation better fits the error than the related misconceptions

Write your analysis in <evaluation> tags, following this structure:
- Show the correct calculation first
- Show the incorrect calculations that demonstrate the error
- Explain the specific flaw in the student's reasoning
- Demonstrate how the misconception led to this particular error
- Distinguish from the related misconceptions
- Keep your explanation to 5-6 clear, non-repetitive sentences
- Focus solely on the reasoning that produced this specific error

Guidelines for writing your explanation:
- Do not restate the problem or name the misconception
- Be precise about the mathematical concepts involved
- Show exactly how the misconception led to the error
- Distinguish from related misconceptions
- Avoid repetition
- Stay focused on this specific error
"""


def format_data(row):
    d = f"""# Question: {row['QuestionText']}
# Correct Answer: {row['CorrectAnswerText']}
# Incorrect Answer: {row['InCorrectAnswerText']}
# Primary Misconception: {row['MisconceptionName']}
"""
    return d


def get_prompt(example):
    return TEMPLATE.format(PROBLEM_DATA=format_data(example), RELATED_MISCONCEPTIONS=example["related_misconceptions"])


class MathMCQEvaluation(BasicRepr):
    """Evaluation of a math MCQ"""

    def __init__(
        self,
        explanation: str,  # Explanation of what mistake led to the incorrect answer
    ):
        store_attr()

    def to_dict(self):
        return {
            "Explanation": self.explanation,
        }


def validate_eval(response: dict):
    focus_columns = ["Explanation"]
    for col in focus_columns:
        if col not in response:
            return False

    # check for new columns
    for col in response.keys():
        if col not in focus_columns:
            return False

    return True


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
    data_dir = kagglehub.dataset_download(cfg.input_dataset)
    mcq_df = pd.read_csv(os.path.join(data_dir, cfg.input_file))

    # evaluate examples -------------------------------------------------------------------------#
    pbar = tqdm(range(cfg.batch_size))
    max_try = 3

    for _ in range(cfg.batch_size):
        content_id = get_content_id_for_eval(mcq_df, cfg)
        focused_df = mcq_df[mcq_df["content_id"] == content_id].copy()

        print("--------------------------------------")
        mname = focused_df["MisconceptionName"].unique()[0]
        print(f"Running for -> {mname}...")
        print(f"# of examples: {len(focused_df)}")

        print("--------------------------------------")
        # iterate over all examples in the focused_df --
        evals = []
        try_count = 0

        for _, example in focused_df.iterrows():
            try_count += 1

            prompt = get_prompt(example)
            example = example.to_dict()

            print(prompt)
            print("-" * 100)

            cli = Client(cfg.model)
            res = cli.structured(prompt, temp=0.5, tools=MathMCQEvaluation, sp=system_prompt)[0]
            res = res.to_dict()

            if not validate_eval(res):
                print(f"Invalid evaluation: {res}")
                continue

            val = 0  # res["Score"]
            evals.append(val)

            print(res["Explanation"])
            print("-" * 100)

            for key, value in res.items():
                example[key] = value
            # save the example
            example_id = f"{example['query_id']}_{example['content_id']}"
            with open(os.path.join(cfg.output_dir, f"{example_id}.json"), "w") as f:
                json.dump(example, f)

            print("~~" * 30)

            if (try_count > max_try) or (len(focused_df) == try_count):
                # save stats
                stats = {
                    "content_id": content_id,
                    "misconception": mname,
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
        # collect all json files in the output dir
        json_files = glob.glob(os.path.join(cfg.output_dir, "*.json"))
        mcq_df = pd.DataFrame([json.load(open(f)) for f in json_files])
        mcq_df.to_csv(os.path.join(collection_dir, "synthetic.csv"), index=False)
        upload_dataset(cfg.upload_dataset, collection_dir)
