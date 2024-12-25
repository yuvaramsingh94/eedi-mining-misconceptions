import random
import time
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from transformers import DataCollatorWithPadding


@dataclass
class RankerCollatorTrain(DataCollatorWithPadding):
    """
    data collector for ranker training, it does the job of sampler and collator!
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
        combined_ids = self.ds["combined_id"]
        self.all_combined_ids = deepcopy(combined_ids)
        self.id2idx = {cid: idx for idx, cid in enumerate(combined_ids)}

        query_ids = [x.split("|")[0] for x in combined_ids]
        self.all_query_ids = deepcopy(query_ids)

        # Add local rank to seed to ensure different seeds across processes
        local_seed = self.cfg.seed + self.cfg.local_rank
        self.rng = random.Random(local_seed)

        self.unlabelled_qid_th = 2_000

    def process_features(self, query_ids, content_groups):
        input_ids = []
        attention_masks = []
        labels = []
        teacher_logits = []
        ce_mask = []
        teacher_mask = []

        for qid, cids in zip(query_ids, content_groups):
            pos_id = cids[0]
            self.rng.shuffle(cids)  # shuffle
            label_idx = cids.index(pos_id)  # find in cids

            for cid in cids:
                combined_id = f"{qid}|{cid}"
                info = self.ds[self.id2idx[combined_id]]

                input_ids.append(info["input_ids"])
                attention_masks.append(info["attention_mask"])

                if self.teacher_map is not None:
                    teacher_logits.append(self.teacher_map[combined_id])

            labels.append(label_idx)
            qid_base = int(qid.split("_")[0])

            if qid_base >= self.unlabelled_qid_th:  # do not train on external data labels
                ce_mask.append(0.0)
                teacher_mask.append(1.0)
            else:
                ce_mask.append(1.0)
                teacher_mask.append(1.0)

        return input_ids, attention_masks, labels, teacher_logits, ce_mask, teacher_mask

    def __call__(self, features):
        query_ids = [feature["query_id"] for feature in features]
        content_groups = []

        for qid in query_ids:
            pos_id = self.label_map[qid]
            candidates = self.query2candidates[qid]
            neg_ids = [x for x in candidates if x != pos_id]

            timestamp_seed = int(time.time() * 1000) + self.cfg.local_rank
            temp_rng = random.Random(timestamp_seed)
            neg_ids = temp_rng.sample(neg_ids, k=self.cfg.train_params.per_device_train_group_size - 1)

            content_ids = [pos_id] + neg_ids
            content_groups.append(content_ids)

        input_ids, attention_masks, labels, teacher_logits, ce_mask, teacher_mask = self.process_features(query_ids, content_groups)

        features = [{"input_ids": input_ids[i], "attention_mask": attention_masks[i]} for i in range(len(input_ids))]

        # --------
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["ce_mask"] = torch.tensor(ce_mask, dtype=torch.float32)
        batch["teacher_mask"] = torch.tensor(teacher_mask, dtype=torch.float32)

        if self.teacher_map is not None:
            batch["teacher_logits"] = torch.tensor(teacher_logits, dtype=torch.float32)
        return batch


@dataclass
class RankerCollator(DataCollatorWithPadding):
    """
    Data collector
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return batch


def show_batch(batch, tokenizer, task="training", print_fn=print):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")
    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")
    print_fn(f"shape of attention_mask: {batch['attention_mask'].shape}")

    if "labels" in batch.keys():
        print_fn(f"shape of labels: {batch['labels'].shape}")
        print_fn(f"labels: {batch['labels']}")

    print_fn("\n\n")
    for idx in range(bs):
        print_fn(f"=== Example {idx} ===")
        print_fn(f"Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}")
        print_fn("~~" * 40)
