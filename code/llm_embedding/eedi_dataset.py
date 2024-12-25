from datasets import Dataset
from transformers import AutoTokenizer


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path, add_eos_token=cfg.model.add_eos_token)
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = cfg.model.padding_side  # "left"
    return tokenizer


def _formatting_func(query):
    task_description = """Retrieve the key misconception behind the wrong answer when given a math problem and its incorrect and correct solutions."""

    return f"Instruct: {task_description}\nQuery: {query}"


class MathDataset:
    """
    Dataset class for processing EEDI math MCQs into query/content inputs for retrieval
    """

    def __init__(self, cfg, query_formatting_func=None):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.query_formatting_func = query_formatting_func if query_formatting_func is not None else _formatting_func

    def pre_process(self, df, is_query, is_train, rng):
        def _get_query(row):
            query = ""
            if is_train:
                if rng.random() < 0.5:
                    query = f"{row['SubjectName']} - {row['ConstructName']}\n"
            else:
                query = f"{row['SubjectName']} - {row['ConstructName']}\n"
            query += f"# Question: {row['QuestionText']}\n"
            query += f"# Correct Answer: {row['CorrectAnswerText']}\n"
            query += f"# Wrong Answer: {row['InCorrectAnswerText']}"
            query = self.query_formatting_func(query)
            return query

        def _get_content(row):
            return row["MisconceptionName"]

        if is_query:
            df["text"] = df.apply(lambda x: _get_query(x), axis=1)
            df = df.rename(columns={"QuestionId_Answer": "query_id"})
            df = df[["query_id", "text"]]
        else:
            df["text"] = df.apply(lambda x: _get_content(x), axis=1)
            df = df.rename(columns={"MisconceptionId": "content_id"})
            df = df[["content_id", "text"]]

        return df

    def tokenize(self, examples):
        tokenized = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_length=True,
        )
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "input_length": tokenized["length"]}

    def get_dataset(self, df, is_query=True, is_train=False, rng=None):
        df = df.copy()
        df = self.pre_process(df, is_query, is_train, rng).reset_index(drop=True)

        ds = Dataset.from_pandas(df)
        ds = ds.map(self.tokenize, batched=True)

        return ds
