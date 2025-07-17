from __future__ import annotations

import torch
import json
from typing import TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    import pathlib


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_file: pathlib.Path, task: str) -> None:
        load = partial(self.load_file, input_file)

        match task:
            case "boolq":
                load("question", "passage")
            case "cola":
                load("sentence")
            case "mnli":
                load("premise", "hypothesis")
            case "mrpc":
                load("sentence1", "sentence2")
            case "multirc":
                load("question", "answer", "paragraph", "Question: {} Answer: {}")
            case "qnli":
                load("question", "sentence")
            case "qqp":
                load("question1", "question2")
            case "rte":
                load("sentence1", "sentence2")
            case "sst2":
                load("sentence")
            case "wsc":
                load("span2_text", "span1_text", "text", "Does \"{}\" refer to \"{}\" in this passage?")
            case _:
                raise ValueError("This is not an implemented task! Please implement it!")

    def load_file(self, input_file: pathlib.Path, key1: str, key2: str | None = None, key3: str | None = None, template: str | None = None) -> None:
        self.texts = []
        self.labels = []

        with input_file.open("r") as file:
            for line in file:
                data = json.loads(line)

                if key2 is not None:
                    if template is not None:
                        assert key3 is not None
                        self.texts.append((template.format(data[key1], data[key2]), data[key3]))
                    else:
                        self.texts.append((data[key1], data[key2]))
                else:
                    self.texts.append(data[key1])

                self.labels.append(data["label"])

    def __len__(self) -> None:
        return len(self.texts)

    def __getitem__(self, index: int) -> None:
        text = self.texts[index]
        label = self.labels[index]

        return text, label


def collate_function(tokenizer: Tokenizer, data: list[tuple[str, str] | int]) -> tuple[torch.LongTensor, torch.BoolTensor, torch.LongTensor]:

    texts = []
    labels = []

    for text, label in data:
        texts.append(text)
        labels.append(label)

    labels = torch.LongTensor(labels)
    encodings = tokenizer.encode_batch(texts)

    input_ids = []
    attention_mask = []

    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)

    input_ids = torch.LongTensor(input_ids)
    attention_mask = ~torch.BoolTensor(attention_mask)

    return input_ids, attention_mask, labels
