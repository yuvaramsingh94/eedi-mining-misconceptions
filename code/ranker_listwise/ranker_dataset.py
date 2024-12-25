from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )

    tokenizer.padding_side = "left"  # use left padding

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class RankerDataset:
    """
    Dataset class for EEDI - Misconception Detection
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)

    def tokenize_function(self, examples):
        tx = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def preprocess_function(self, df, is_train=False, rng=None):
        formatted_texts = []
        system = "Pick the misconception that explains the incorrect answer most specifically."

        for _, row in df.iterrows():
            few_shot_examples = row["examples"]

            user_message = ""
            if len(few_shot_examples.strip()) > 0:
                user_message += "The following reference examples are provided to help you understand relevant mistakes and the underlying misconceptions.\n"
                user_message += f"Reference examples:\n{few_shot_examples}\n\n"

            user_message += "A math problem is provided below together with the correct answer and an incorrect answer.\n\n"

            # optionally add topic
            if is_train:
                if rng.random() < 0.5:
                    user_message += f"Topic: {row['SubjectName']} - {row['ConstructName']}\n"
            else:
                user_message += f"Topic: {row['SubjectName']} - {row['ConstructName']}\n"

            user_message += f"Question: {row['QuestionText']}\n\nCorrect Answer: {row['CorrectAnswerText']}\nIncorrect Answer: {row['InCorrectAnswerText']}\n\n"

            if is_train:
                num_thoughts = rng.choice([0, 1, 2, 3])
                if num_thoughts > 0:
                    cot_list = [row["cot_7b"], row["cot_14b"], row["cot_32b"]]
                    selected_cots = rng.sample(cot_list, k=num_thoughts)
                    for cot in selected_cots:
                        user_message += f"Thought: {cot}\n\n"

            else:
                user_message += f"Thought: {row['cot_7b']}\n\n"
                user_message += f"Thought: {row['cot_14b']}\n\n"
                user_message += f"Thought: {row['cot_32b']}\n\n"
            user_message += "--\n"

            user_message += f"A list of {len(row['MisconceptionNameList'])} candidate misconceptions that may explain the incorrect answer:\n"
            for letter, misconception in zip("ABCDE", row["MisconceptionNameList"]):
                user_message += f"{letter}. {misconception}\n"

            user_message += "Which misconception explains the incorrect answer most specifically? (A, B, C, D, or E)"

            conversation = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]

            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)

        df["text"] = formatted_texts
        return df

    def get_dataset(self, df, is_train=False, rng=None):
        """use this function to get the dataset

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Dataset: HF Dataset object with tokenized inputs and labels
        """

        df = deepcopy(df)
        df = self.preprocess_function(df, is_train, rng)
        task_dataset = Dataset.from_pandas(df)
        remove_columns = [col for col in df.columns if col not in ["query_id", "content_ids", "combined_id", "label", "teacher_logits"]]

        task_dataset = task_dataset.map(self.tokenize_function, batched=True, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)

        return task_dataset
