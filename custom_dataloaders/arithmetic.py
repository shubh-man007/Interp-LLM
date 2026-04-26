import json
import os
import torch
import numpy as np
from datasets import load_dataset

from .base import BaseDataset, BaseAccuracyTracker
import random


class ArithmeticAccuracyTracker(BaseAccuracyTracker):
    def __init__(self):
        # super().__init__(None, None)
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=1)
        for prediction, target in zip(predictions, targets):
            try:
                # Convert the model's output to an integer
                if prediction == target.squeeze():
                    self.correct += 1
            except ValueError:
                # If the model's output can't be converted to an integer, count it as incorrect
                pass
            self.total += 1
        return predictions

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


class ArithmeticDataset(BaseDataset):
    def __init__(
        self,
        tokenizer,
        detokenizer,
        max_token_length=512,
        padding_side="right",
        split="test",
        last_token=False,
        add_question_structure=False,
        icl_few_shot=0,
        operation="addition",  # Can be "addition" or "multiplication"
        seed=42,
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
        self.operation = operation
        self.generate_dataset(split)
        self.few_shot = icl_few_shot

        self.pre_text = self._few_shot_pre_text("", self.few_shot)

    def generate_dataset(self, split):
        self.ds = []
        num_samples = 1000 if split == "test" else 10000  # Adjust as needed

        for _ in range(num_samples):
            x1 = random.randint(0, 99)
            x2 = random.randint(1, 50)

            if self.operation == "addition":
                y = x1 + x2
                op_text = "plus"
            elif self.operation == "multiplication":
                y = x1 * x2
                op_text = "times"
            else:
                raise ValueError("Invalid operation")

            prompt = f"Q:What is {x1} {op_text} {x2}? A:"
            if self.tokenizer(str(y), prepend_bos=False).shape[-1] > 1:
                continue
            self.ds.append({"prompt": prompt, "answer": y})

    def get_accuracy_tracker(self):
        return ArithmeticAccuracyTracker()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        tokenized_text = self.tokenizer(
            self.pre_text + item["prompt"], prepend_bos=False
        )

        if self.add_question_structure:
            tokenized_text = tokenized_text[:, 1:]

        tokenized_text, last_token_index = self._pad_tokens(tokenized_text)
        tokenized_text = tokenized_text.squeeze()
        if self.last_token:
            return (
                tokenized_text,
                self.tokenizer(str(item["answer"]), prepend_bos=False).squeeze(),
            ), last_token_index
        else:
            return tokenized_text, item["answer"]

    def _few_shot_pre_text(self, pre_text, num_examples):
        if num_examples == 0:
            return pre_text

        for _ in range(num_examples):
            example = random.choice(self.ds)
            pre_text += f"{example['prompt']}{example['answer']}\n"

        return pre_text

    def _get_question_structure_tokens(self, pre_text):
        pre_questionnaire_text = pre_text + "Question: "
        post_questionnaire_text = "\nAnswer: "

        pre_questionnaire_tokens = self.tokenizer(pre_questionnaire_text)["input_ids"]
        post_questionnaire_tokens = self.tokenizer(post_questionnaire_text)["input_ids"]

        return torch.tensor(pre_questionnaire_tokens), torch.tensor(
            post_questionnaire_tokens
        )
