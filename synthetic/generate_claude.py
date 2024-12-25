import argparse
import glob
import json
import os
import random
import time
import uuid

import kagglehub
import pandas as pd
from claudette import Client
from fastcore.basics import BasicRepr, store_attr
from omegaconf import OmegaConf
from tqdm.auto import tqdm

DEBUG = True


def upload_dataset(handle: str, local_dir: str):
    kagglehub.dataset_upload(handle, local_dir)


class MathMCQ(BasicRepr):
    """Represents a mathematical multiple-choice question for diagnosing misconceptions."""

    def __init__(
        self,
        subject_name: str,  # The subject area of the question
        construct_name: str,  # The specific mathematical construct being tested
        question_text: str,  # The text of the question
        answer_a_text: str,  # Text for answer choice A
        answer_b_text: str,  # Text for answer choice B
        answer_c_text: str,  # Text for answer choice C
        answer_d_text: str,  # Text for answer choice D
        correct_answer: str,  # The letter (A, B, C, or D) of the correct answer
        misconception_a_name: str,  # The misconception name targeted by answer A (if any)
        misconception_b_name: str,  # The misconception name targeted by answer B (if any)
        misconception_c_name: str,  # The misconception name targeted by answer C (if any)
        misconception_d_name: str,  # The misconception name targeted by answer D (if any)
    ):
        store_attr()

    def to_dict(self):
        """Convert the MathMCQ object to a dictionary format."""
        return {
            "SubjectName": self.subject_name,
            "ConstructName": self.construct_name,
            "QuestionText": self.question_text,
            "AnswerAText": self.answer_a_text,
            "AnswerBText": self.answer_b_text,
            "AnswerCText": self.answer_c_text,
            "AnswerDText": self.answer_d_text,
            "CorrectAnswer": self.correct_answer,
            "MisconceptionAName": self.misconception_a_name,
            "MisconceptionBName": self.misconception_b_name,
            "MisconceptionCName": self.misconception_c_name,
            "MisconceptionDName": self.misconception_d_name,
        }


class MathMCQList(BasicRepr):
    "list of MathMCQs"

    def __init__(self, examples: list[MathMCQ]):
        store_attr()


focus_columns = [
    "QuestionId",
    "SubjectName",
    "ConstructName",
    "QuestionText",
    "AnswerAText",
    "AnswerBText",
    "AnswerCText",
    "AnswerDText",
    "CorrectAnswer",
    "MisconceptionAName",
    "MisconceptionBName",
    "MisconceptionCName",
    "MisconceptionDName",
]


def validate_mcq(mcq: dict):
    for col in focus_columns:
        if col == "QuestionId":
            continue

        if col not in mcq:
            return False
    for col in mcq.keys():
        if col not in focus_columns:
            return False
    return True


def get_qids_by_misconception(df, misconception_id):
    qids = []
    misconception_id = str(misconception_id)
    for _, row in df.iterrows():
        for opt in "ABCD":
            if row[f"Misconception{opt}Id"] == misconception_id:
                qids.append(row["QuestionId"])
                break
    return qids


def get_cluster(cfg, clusters):
    completed_idxs = []
    for dir_name in cfg.existing_dirs:
        fns = glob.glob(os.path.join(dir_name, "*.json"))
        completed_idxs.extend([int(os.path.basename(fn).split(".")[0]) for fn in fns])
    remaining_idxs = [i for i in range(len(clusters)) if i not in completed_idxs]
    this_cluster_idx = random.choice(remaining_idxs)
    this_cluster = clusters[this_cluster_idx]

    print(f"# of remaining clusters: {len(remaining_idxs)}")
    print(f"# of existing clusters: {len(completed_idxs)}")
    print(f"# of all clusters: {len(clusters)}")

    if len(remaining_idxs) == 0:
        raise ValueError("No remaining clusters to evaluate")

    return this_cluster_idx, this_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = OmegaConf.load(f)
    os.makedirs(cfg.output_dir, exist_ok=True)

    stats_dir = os.path.join(cfg.output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    seed = random.randint(100, 100000)
    print(f"Using seed: {seed}")
    cfg.seed = seed

    # Load data -----------------------------------------------------------------------------------#
    data_dir = kagglehub.competition_download(cfg.dataset.comp_dataset)

    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    content_df = pd.read_csv(os.path.join(data_dir, "misconception_mapping.csv"))
    content_df["MisconceptionId"] = content_df["MisconceptionId"].astype(str)
    id2name = dict(zip(content_df["MisconceptionId"], content_df["MisconceptionName"]))
    name2id = dict(zip(content_df["MisconceptionName"], content_df["MisconceptionId"]))

    # misconception stats -------------------------------------------------------------------------#
    misconception_ids = set()
    for col in ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]:
        misconception_ids.update(df[col].dropna().astype(int).unique())
    misconception_ids = list(map(str, misconception_ids))
    print(f"Number of unique MisconceptionIds: {len(misconception_ids)}")

    # misconception stats -------------------------------------------------------------------------#
    misconception_ids = set()
    for col in ["MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId"]:
        misconception_ids.update(df[col].dropna().astype(int).unique())
    misconception_ids = list(map(str, misconception_ids))
    print(f"Number of unique MisconceptionIds: {len(misconception_ids)}")
    missing_ids = content_df[~content_df["MisconceptionId"].isin(misconception_ids)]["MisconceptionId"].unique().tolist()

    # processing ---------------------------------------------------------------------------------#
    for option in ["A", "B", "C", "D"]:
        col = f"Misconception{option}Id"
        new_col = f"Misconception{option}Name"
        df[col] = df[col].apply(lambda x: str(int(x)) if x == x else "")
        df = df.merge(content_df[["MisconceptionId", "MisconceptionName"]], left_on=col, right_on="MisconceptionId", how="left")
        df[new_col] = df["MisconceptionName"]
        df = df.drop(["MisconceptionId", "MisconceptionName"], axis=1)
    df = df.drop(columns=["ConstructId", "SubjectId"])

    # context ------------------------------------------------------------------------------------#
    example_df = df[focus_columns].copy()
    for col in ["MisconceptionAName", "MisconceptionBName", "MisconceptionCName", "MisconceptionDName"]:
        example_df[col] = example_df[col].fillna("")

    # load clusters -----------------------------------------------------------------------------#
    cluster_dir = kagglehub.dataset_download(cfg.dataset.cluster_dataset)
    with open(os.path.join(cluster_dir, "diverse_clusters.json"), "r") as f:
        clusters = json.load(f)

    # chat ---------------------------------------------------------------------------------------#
    messages = []

    prompt_template = """You will be generating Multiple Choice Questions (MCQs) that diagnose specific mathematical misconceptions. Here are the misconceptions you should focus on:

<misconceptions>
{misconceptions}
</misconceptions>

Here are reference MCQs that demonstrate how to effectively diagnose these misconceptions:

<reference_mcqs>
{fs_examples}
</reference_mcqs>

Your task is to generate {n} new MCQs that diagnose misconceptions not already covered by the reference MCQs.

First, analyze the reference MCQs carefully:
1. For each reference MCQ, identify in your <analysis> tags:
   - Which misconception it targets
   - How the incorrect answers map to specific misconceptions
   - What makes the question effective at diagnosing the misconception
2. Note the style, difficulty level, and precision of language used

Then, in your <planning> tags:
- List which misconceptions still need coverage
- For each needed misconception, brainstorm mathematical contexts where it commonly appears
- Design questions where the misconception leads naturally to specific wrong answers
- Take notes on how you can craft new MCQs that adheres to the reference MCQs' style, difficulty level, and precision of language

Finally, generate new MCQs following these important guidelines:
- Make sure each incorrect answer maps clearly to exactly one misconception
- Use precise mathematical language matching the style of reference MCQs
- Make questions challenging enough that students must demonstrate real understanding
- Ensure wrong answers are plausible and stem from genuine misconceptions, not careless errors
- Use the exact wording of misconceptions as given in the misconceptions list
- Pay careful attention to subtle differences between the misconceptions and observe which one is the most appropriate for a given incorrect answer
- Keep the construct name and subject name as short as possible hiding the details of the misconception
- Questions should be of higher difficulty level than reference MCQs
"""

    def generate_examples(cluster):
        fs_qids = []
        for misconception_id in cluster:
            qids = get_qids_by_misconception(df, misconception_id)
            fs_qids.extend(qids)
        print(f"{cluster} has {len(fs_qids)} examples")

        # get example pool
        example_pool = example_df[example_df["QuestionId"].isin(fs_qids)]
        n_ex = min(cfg.n_ex_in_context, len(example_pool))
        example_pool = example_pool.sample(n_ex, random_state=cfg.seed).to_dict(orient="records")

        # reset misconceptions to nan for the ones outside the cluster ---
        misconception_names = [id2name[c] for c in cluster]
        for mcq in example_pool:
            for opt in "ABCD":
                if mcq[f"Misconception{opt}Name"] not in misconception_names:
                    mcq[f"Misconception{opt}Name"] = ""

        # remove question ids
        for mcq in example_pool:
            mcq.pop("QuestionId")

        fs_examples = f"""{example_pool}"""
        target_misconceptions = "\n- ".join(misconception_names)
        target_misconceptions = f"- {target_misconceptions}"
        prompt = prompt_template.format(n=cfg.num_gen_per_group, misconceptions=target_misconceptions, fs_examples=fs_examples)

        if DEBUG:
            print("--" * 40)
            print(prompt)
            print("--" * 40)

        try:
            cli = Client(cfg.model)
            res = cli.structured(prompt, tools=MathMCQList)[0]
            print(f"Usage: {cli.use}")
            print(res)
        except Exception as e:
            print(f"Error generating: {e}")
            return

        to_return = []
        for mcq in res.examples:
            if DEBUG:
                print(mcq)

            if not validate_mcq(mcq):
                print(f"Invalid MCQ: {mcq}")
                continue  #

            # add metadata
            qid = str(uuid.uuid4())
            mcq["QuestionId"] = qid

            # save this mcq
            with open(os.path.join(cfg.output_dir, f"{qid}.json"), "w") as f:
                json.dump(mcq, f)
            to_return.append(mcq)
        return to_return

    # save & upload -----------------------------------------------------------------------------#

    pbar = tqdm(range(cfg.batch_size))

    for _ in range(cfg.batch_size):
        # get remaining misconceptions
        idx, cluster = get_cluster(cfg, clusters)

        print("-" * 100)
        print("Running for cluster with misconceptions:")
        for c in cluster:
            print(f"  - {id2name[c]}")
        print("-" * 100)

        group_examples = generate_examples(cluster)

        # save group --
        with open(os.path.join(cfg.output_dir, "stats", f"{idx}.json"), "w") as f:
            json.dump(group_examples, f)
        time.sleep(cfg.sleep_time)
        pbar.update(1)

    pbar.close()

    # handle upload -----------------------------------------------------------------------------#
    if cfg.upload_to_kaggle:
        collection_dir = os.path.join(cfg.output_dir, "synthetic")
        os.makedirs(collection_dir, exist_ok=True)

        # collect all json files in the output dir
        json_files = glob.glob(os.path.join(cfg.output_dir, "*.json"))
        mcq_df = pd.DataFrame([json.load(open(f)) for f in json_files])
        mcq_df.to_csv(os.path.join(collection_dir, "synthetic.csv"), index=False)
        upload_dataset(cfg.upload_dataset, collection_dir)

    # Usage:
    # python synthetic/generate_claude.py --config-path ./conf/synthetic/conf_gen_claude.yaml
