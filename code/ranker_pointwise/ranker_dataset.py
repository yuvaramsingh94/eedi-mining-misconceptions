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

    def preprocess_function(self, df, is_train=False):
        formatted_texts = []
        system = "You are an expert in detecting grade-school level math misconceptions. Verify if the incorrect answer stems from the provided misconception."

        for row_idx, row in df.iterrows():
            target_misconception = row["MisconceptionName"]
            few_shot_examples = row["examples"]
            user_message = f"Misconception: {target_misconception}\n\n"

            if len(few_shot_examples.strip()) > 0:
                user_message += f"Demos for the misconception:\n{few_shot_examples}\n\n"

            user_message += f"Subject: {row['SubjectName']}\nTopic: {row['ConstructName']}\nQuestion: {row['QuestionText']}\nCorrect Answer: {row['CorrectAnswerText']}\n"
            user_message += f"Incorrect Answer: {row['InCorrectAnswerText']}\n"

            if self.cfg.use_cot:
                if is_train:
                    if row_idx % 2 == 0:
                        user_message += f"Thought: {row['cot']}\n\n"
                else:
                    user_message += f"Thought: {row['cot']}\n\n"

            user_message += f"Does the misconception ({target_misconception}) lead to the incorrect answer? (Yes/No)"

            conversation = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]

            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)

        df["text"] = formatted_texts
        return df

    def get_dataset(self, df, is_train=False):
        """use this function to get the dataset

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            Dataset: HF Dataset object with tokenized inputs and labels
        """

        df = deepcopy(df)
        df["content_id"] = df["content_id"].astype(str)
        df = self.preprocess_function(df)
        task_dataset = Dataset.from_pandas(df)
        remove_columns = [col for col in df.columns if col not in ["query_id", "content_id", "combined_id", "label"]]

        task_dataset = task_dataset.map(self.tokenize_function, batched=True, num_proc=self.cfg.model.num_proc, remove_columns=remove_columns)

        return task_dataset
