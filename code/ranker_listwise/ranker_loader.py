from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


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
        unlabelled_qid_th = 2000
        query_ids = [feature["query_id"] for feature in features]
        query_ids = [int(q.split("_")[0]) for q in query_ids]
        ce_mask = [0.0 if qid >= unlabelled_qid_th else 1.0 for qid in query_ids]

        labels = None
        if "label" in features[0].keys():
            labels = [feature["label"] for feature in features]

        teacher_scores = None
        if "teacher_logits" in features[0].keys():
            teacher_scores = [feature["teacher_logits"] for feature in features]

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

        batch["ce_mask"] = torch.tensor(ce_mask, dtype=torch.float32)

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if teacher_scores is not None:
            batch["teacher_logits"] = torch.tensor(teacher_scores, dtype=torch.float32)
        return batch


def show_batch(batch, tokenizer, print_fn=print, **kwargs):
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
