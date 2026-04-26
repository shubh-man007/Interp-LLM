# the dataset is taken from https://github.com/hannamw/gpt2-greater-than/blob/main/dataset.py
import numpy as np
import json
import os
import random
from typing import List, Union
from pathlib import Path
import torch
from transformers import PreTrainedTokenizer
from .base import BaseDataset, BaseAccuracyTracker


POTENTIAL_NOUNS = [
    "war",
    # "abduction",
    # "accord",
    # "affair",
    # "agreement",
    # "appraisal",
    # "assaults",
    # "assessment",
    # "attack",
    # "attempts",
    "campaign",
    # "captivity",
    # "case",
    # "challenge",
    # "chaos",
    # "clash",
    # "collaboration",
    # "coma",
    "competition",
    "confrontation",
    # "consequence",
    "conspiracy",
    # "construction",
    # "consultation",
    "contact",
    "contract",
    # "convention",
    # "cooperation",
    # "custody",
    # "deal",
    # "decline",
    # "decrease",
    # "demonstrations",
    # "development",
    # "disagreement",
    # "disorder",
    "dispute",
    # "domination",
    # "dynasty",
    # "effect",
    # "effort",
    # "employment",
    # "endeavor",
    # "engagement",
    "epidemic",
    # "evaluation",
    "exchange",
    "existence",
    # "expansion",
    # "expedition",
    # "experiments",
    # "fall",
    "fame",
    "flights",
    # "friendship",
    # "growth",
    # "hardship",
    # "hostility",
    # "illness",
    "impact",
    "imprisonment",
    # "improvement",
    # "incarceration",
    # "increase",
    # "insurgency",
    "invasion",
    "investigation",
    # "journey",
    # "kingdom",
    "marriage",
    # "modernization",
    # "negotiation",
    # "notoriety",
    # "obstruction",
    "operation",
    # "order",
    # "outbreak",
    # "outcome",
    # "overhaul",
    # "patrols",
    # "pilgrimage",
    # "plague",
    "plan",
    # "practice",
    # "process",
    # "program",
    # "progress",
    # "project",
    # "pursuit",
    # "quest",
    # "raids",
    # "reforms",
    # "reign",
    "relationship",
    # "retaliation",
    # "riot",
    # "rise",
    # "rivalry",
    # "romance",
    # "rule",
    # "sanctions",
    # "shift",
    # "siege",
    # "slump",
    # "stature",
    # "stint",
    "strikes",
    "study",
    # "test",
    # "testing",
    # "tests",
    "therapy",
    # "tour",
    # "tradition",
    "treaty",
    "trial",
    # "trip",
    # "unemployment",
    # "voyage",
    "warfare",
    "work",
]


def generate_real_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {year} to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def generate_bad_sentence(noun: str, year: int, eos: bool = False) -> str:
    century = year // 100
    sentence = f"The {noun} lasted from the year {century}01 to the year {century}"
    if eos:
        sentence = "<|endoftext|> " + sentence
    return sentence


def get_valid_years(
    tokenizer,
    detokenizer,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years, prepend_bos=False)

    detokenized = [
        [detokenizer(year_tok) for year_tok in year_toks] for year_toks in tokens
    ]
    valid = torch.tensor(
        [(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized]
    )

    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]


class YearAccuracyTracker(BaseAccuracyTracker):
    def __init__(self, tokenizer, detokenizer):
        # super().__init__(None, None)
        self.correct = 0
        self.total = 0
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

    def update(self, outputs, targets):
        # Assuming outputs is a list of strings (model's answers)
        predictions = torch.argmax(outputs, dim=1)
        detokenized_predictions = [self.detokenizer(output) for output in predictions]
        # and targets is a list of integers (correct answers)
        for prediction, target in zip(detokenized_predictions, targets):
            try:
                # Convert the model's output to an integer
                if int(prediction) > target.squeeze():
                    self.correct += 1
            except ValueError:
                # If the model's output can't be converted to an integer, count it as incorrect
                pass
            self.total += 1
        return detokenized_predictions

    def get_accuracy(self):
        return self.correct / self.total if self.total > 0 else 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def save(self, file_path):
        """
        Save the accuracies
        """

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if self.total == 0:
            net_accuracy = 0
        else:
            net_accuracy = self.get_accuracy()

        with open(file_path, "w") as f:
            json.dump(
                {
                    "correct": self.correct,
                    "total": self.total,
                    "accuracy": net_accuracy,
                },
                f,
            )

    def print_accuracies(self):
        print(f"Accuracy: {self.get_accuracy():.2f}")


class YearDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        detokenizer,
        N: int,
        nouns: Union[str, List[str], Path],
        max_token_length=512,
        padding_side="right",
        split="test",
        last_token=False,
        add_question_structure=False,
        icl_few_shot=0,
        balanced: bool = True,
        eos: bool = False,
        device: str = "cpu",
        seed: int = 42,
    ):
        super().__init__(
            tokenizer,
            detokenizer,
            max_token_length,
            padding_side,
            last_token,
            add_question_structure,
            icl_few_shot,
        )

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.years_to_sample_from = get_valid_years(tokenizer, detokenizer)
        self.N = N
        self.eos = eos
        self.device = device

        self.nouns = self._load_nouns(nouns)
        self.years = self._generate_years(balanced)
        self.years_XX = self.years // 100
        self.years_YY = self.years % 100

        self.good_sentences = [
            generate_real_sentence(noun, int(year.item()), eos=eos)
            for noun, year in zip(self.nouns, self.years)
        ]
        self.bad_sentences = [
            generate_bad_sentence(noun, int(year.item()), eos=eos)
            for noun, year in zip(self.nouns, self.years)
        ]

        self._tokenize_sentences()
        self._create_logits_mask()

        pre_text = "Answer the following questions about historical events. For each question, determine if the end year is greater than the start year.\n"
        self.pre_questionnaire_tokens, self.post_questionnaire_tokens = (
            self._get_question_structure_tokens(pre_text=pre_text)
        )

    def _load_nouns(self, nouns):
        if nouns is None:
            return random.choices(POTENTIAL_NOUNS, k=self.N)
        elif isinstance(nouns, str):
            return [nouns] * self.N
        elif isinstance(nouns, list):
            return random.choices(nouns, k=self.N)
        elif isinstance(nouns, Path):
            with open(nouns, "r") as f:
                noun_list = [line.strip() for line in f]
            return random.choices(noun_list, k=self.N)
        else:
            raise ValueError(
                f"Got bad type of nouns: {type(nouns)}; for nouns: {nouns}"
            )

    def _generate_years(self, balanced):
        if balanced:
            years = []
            current_year = 2
            years_to_sample_from_YY = self.years_to_sample_from % 100
            for i in range(self.N):
                sample_pool = self.years_to_sample_from[
                    years_to_sample_from_YY == current_year
                ]
                years.append(sample_pool[random.randrange(len(sample_pool))])
                current_year += 1
                if current_year >= 99:
                    current_year -= 97
            return torch.tensor(years)
        else:
            return torch.tensor(
                self.years_to_sample_from[
                    torch.randint(0, len(self.years_to_sample_from), (self.N,))
                ]
            )

    def _tokenize_sentences(self):
        good_tokenized = self.tokenizer(
            self.good_sentences,
            prepend_bos=True,
            # return_tensors="pt",
            # padding=True,
            # truncation=True
        )
        self.good_toks = good_tokenized.to(self.device)

        bad_tokenized = self.tokenizer(
            self.bad_sentences,
            prepend_bos=False,
            # return_tensors="pt",
            # padding=True,
            # truncation=True
        )
        self.bad_toks = bad_tokenized.to(self.device)

    def _create_logits_mask(self):
        _good_logits_masks = []
        for year in self.years_YY:
            logits_mask = torch.arange(100)
            _good_logits_masks.append(logits_mask > year)
        self.good_mask = torch.stack(_good_logits_masks).to(self.device)

    def get_accuracy_tracker(self):
        return YearAccuracyTracker(
            tokenizer=self.tokenizer, detokenizer=self.detokenizer
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        tokenized_text, label = self.good_toks[idx], self.years_YY[idx]

        if self.add_question_structure:
            tokenized_text = tokenized_text[1:]

        tokenized_text, last_token_index = self._pad_tokens(tokenized_text.unsqueeze(0))
        tokenized_text = tokenized_text.squeeze()

        if self.last_token:
            return (tokenized_text, label), last_token_index
        else:
            return tokenized_text, label

    def _get_question_structure_tokens(self, pre_text):
        pre_questionnaire_text = pre_text + "Question: "
        post_questionnaire_text = "\nAnswer: "

        pre_questionnaire_tokens = self.tokenizer(pre_questionnaire_text)
        post_questionnaire_tokens = self.tokenizer(post_questionnaire_text)

        return torch.tensor(pre_questionnaire_tokens), torch.tensor(
            post_questionnaire_tokens
        )

    def _few_shot_pre_text(self, pre_text, num_examples):
        if num_examples == 0:
            return pre_text

        for _ in range(num_examples):
            if random.random() < 0.5:
                example = random.choice(self.good_sentences)
                answer = "True"
            else:
                example = random.choice(self.bad_sentences)
                answer = "False"
            pre_text += f"{example}\nAnswer: {answer}\n\n"

        return pre_text
