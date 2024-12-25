import random
import time
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import DataCollatorWithPadding


@dataclass
class RetrieverDataCollator(DataCollatorWithPadding):
    """
    data collector for retriever training, it does the job of sampler and collator
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        [setattr(self, k, v) for k, v in self.kwargs.items()]

        # mappings
        query_ids = self.query_ds["query_id"]
        self.all_query_ids = deepcopy(query_ids)
        self.query2idx = {qid: idx for idx, qid in enumerate(query_ids)}

        content_ids = self.content_ds["content_id"]
        self.content_ids = deepcopy(content_ids)
        self.all_content_ids = content_ids
        self.content2idx = {cid: idx for idx, cid in enumerate(content_ids)}

        # Add local rank to seed to ensure different seeds across processes
        local_seed = self.cfg.seed + self.cfg.local_rank
        self.rng = random.Random(local_seed)

        self.sampling_methods = deepcopy(self.cfg.train_params.batch_sampling)
        self.sampling_weights = deepcopy(self.cfg.train_params.batch_sampling_weights)

        if len(self.negative_map) == 0:
            print("Warning! No negative map provided")
        else:
            self.sampling_methods.append("hard_negative")
            hn_weight = 1.0  # sum(self.sampling_weights)  # TODO: put in config
            self.sampling_weights.append(hn_weight)

        for method, weight in zip(self.sampling_methods, self.sampling_weights):
            print(f"Batching: {method}: {weight}")

    # sampling functions --------------------------------------------------------------------------#
    def hard_negative_sampler(self, query_ids, content_ids):
        timestamp_seed = int(time.time() * 1000) + self.cfg.local_rank
        temp_rng = random.Random(timestamp_seed)

        bs = len(query_ids)
        ratio = self.cfg.train_params.num_hard_negatives

        selected_query_ids = []
        selected_content_ids = []

        visited = set()
        for qid, cid in zip(query_ids, content_ids):
            if cid not in visited:
                selected_query_ids.append(qid)
                selected_content_ids.append(cid)
                visited.add(cid)

                # hard negative sampling
                xx = self.negative_map[qid]
                neg_cids = temp_rng.sample(xx, k=ratio)

                for neg_cid in neg_cids:
                    if neg_cid not in visited:
                        q_candidates = self.content2query[neg_cid]
                        if len(q_candidates) > 0:
                            neg_qid = temp_rng.sample(q_candidates, k=1)[0]
                            selected_query_ids.append(neg_qid)
                            selected_content_ids.append(neg_cid)
                            visited.add(neg_cid)
        # --
        if len(selected_query_ids) >= bs:
            selected_query_ids = selected_query_ids[:bs]
            selected_content_ids = selected_content_ids[:bs]
            return selected_query_ids, selected_content_ids

        # fill up remaining ---
        n_rem = bs - len(selected_query_ids)
        random_pool = list(set(self.all_content_ids).difference(set(selected_content_ids)))
        random_pool = temp_rng.sample(random_pool, k=n_rem)

        for cid in random_pool:
            qid = temp_rng.sample(self.content2query[cid], k=1)[0]
            selected_query_ids.append(qid)
            selected_content_ids.append(cid)

        return selected_query_ids, selected_content_ids

    def random_sampler(self, query_ids, content_ids):
        bs = len(query_ids)

        selected_content_ids = list(set(content_ids))
        rem = bs - len(selected_content_ids)

        if rem > 0:
            random_pool = list(set(self.all_content_ids).difference(set(selected_content_ids)))
            random_pool = self.rng.sample(random_pool, k=rem)
            selected_content_ids.extend(random_pool)

        selected_query_ids = []
        for cid in selected_content_ids:
            qid = self.rng.sample(self.content2query[cid], k=1)[0]
            selected_query_ids.append(qid)

        return selected_query_ids, selected_content_ids

    def sampler(self, query_ids, content_ids):
        methods = self.sampling_methods
        weights = self.sampling_weights

        sampler_fn = self.rng.choices(methods, weights=weights, k=1)[0]

        if sampler_fn == "hard_negative":
            return self.hard_negative_sampler(query_ids, content_ids)
        elif sampler_fn == "random":
            return self.random_sampler(query_ids, content_ids)
        else:
            raise ValueError(f"Unknown sampler function: {sampler_fn}")

    def process_features(self, query_ids, content_ids):
        updated_features = []

        for query_id, content_id in zip(query_ids, content_ids):
            example = dict()

            example["query_id"] = query_id
            example["content_id"] = content_id

            # get fields
            ex_query_info = self.query_ds[self.query2idx[example["query_id"]]]
            ex_content_info = self.content_ds[self.content2idx[example["content_id"]]]

            # use fields
            example["q_input_ids"] = ex_query_info["input_ids"]
            example["q_attention_mask"] = ex_query_info["attention_mask"]

            example["c_input_ids"] = ex_content_info["input_ids"]
            example["c_attention_mask"] = ex_content_info["attention_mask"]

            updated_features.append(example)
        return updated_features

    def __call__(self, features):
        query_ids = [feature["query_id"] for feature in features]
        content_ids = [feature["content_id"] for feature in features]

        query_ids, content_ids = self.sampler(query_ids, content_ids)

        features = self.process_features(query_ids, content_ids)

        # ----
        q_features = [
            {
                "input_ids": feature["q_input_ids"],
                "attention_mask": feature["q_attention_mask"],
            }
            for feature in features
        ]
        c_features = [
            {
                "input_ids": feature["c_input_ids"],
                "attention_mask": feature["c_attention_mask"],
            }
            for feature in features
        ]

        # --------
        q_batch = self.tokenizer.pad(
            q_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        c_batch = self.tokenizer.pad(
            c_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # create teacher logits matrix if available
        teacher_scores = np.full((len(query_ids), len(content_ids)), -10.0)
        for i, qid in enumerate(query_ids):
            for j, cid in enumerate(content_ids):
                if f"{qid}|{cid}" in self.teacher_logits:
                    teacher_scores[i, j] = self.teacher_logits[f"{qid}|{cid}"]
                elif cid in self.online_negative_map.get(qid, []):
                    teacher_scores[i, j] = -2.0
                else:
                    teacher_scores[i, j] = -10.0

                teacher_scores[i, j] = self.teacher_logits.get(f"{qid}|{cid}", -10.0)
        teacher_scores = torch.tensor(teacher_scores, dtype=torch.float32)

        return {"queries": q_batch, "contents": c_batch, "teacher_scores": teacher_scores}


@dataclass
class TextCollator(DataCollatorWithPadding):
    """
    data collector for query/content inputs
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        # Extract just the needed fields from each feature
        features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]

        # padding
        padded_inputs = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = dict()
        batch["input_ids"] = padded_inputs["input_ids"]
        batch["attention_mask"] = padded_inputs["attention_mask"]

        tensor_keys = ["input_ids", "attention_mask"]
        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)
        return batch


# ---
def show_batch(batch, tokenizer, n_examples=4, print_fn=print):
    bs = batch["queries"]["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids (query): {batch['queries']['input_ids'].shape}")
    print_fn(f"shape of input_ids (content): {batch['contents']['input_ids'].shape}")

    print("--" * 80)
    for idx in range(n_examples):
        print_fn(f"[Query]:\n{tokenizer.decode(batch['queries']['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 20)
        print_fn(f"[Content]:\n{tokenizer.decode(batch['contents']['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("--" * 80)


def show_batch_fs(batch, tokenizer, n_examples=4, print_fn=print):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")

    print("--" * 80)
    for idx in range(n_examples):
        print_fn(f"[Query]:\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 20)
