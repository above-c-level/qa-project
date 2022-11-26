#! /usr/bin/env python3
import argparse
import os
import sys
import re
import time
from typing import Dict, List, Set, Tuple, Protocol

from helpers import (read_questions, read_story, Story,
                     get_story_question_answers)
from terminalhelper import NEWLINE, VERBATIM, stringformat
import numpy as np
from pprint import pprint
from ml_approach import ml_friendly_sentences, ml_friendly_words
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle

TIMING = False


class SKLearnModel(Protocol):

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...


class QA:
    """
    A question answerer class. Acts as a controller to bind together the machine
    learning models found in ml_approach.py and ml_models.py with the driver in
    qa.py. Requires that the models be trained and saved in the models folder.
    """

    def __init__(
        self,
        sentence_model_path="models/sentence_model.pkl",
        word_start_model_path="models/word_start_model.pkl",
        word_end_model_path="models/word_end_model.pkl",
    ):
        if not os.path.exists("models"):
            raise FileNotFoundError(
                "Models folder not found. Please train the models first using "
                "`python3 train_models.py train`")
        if not os.path.exists(sentence_model_path):
            raise FileNotFoundError(
                "Sentence model not found. Please train the models first "
                "using `python3 train_models.py train`")
        if not os.path.exists(word_start_model_path):
            raise FileNotFoundError(
                "Word start model not found. Please train the models first "
                "using `python3 train_models.py train`")
        if not os.path.exists(word_end_model_path):
            raise FileNotFoundError(
                "Word end model not found. Please train the models first "
                "using `python3 train_models.py train`")
        with open(sentence_model_path, "rb") as f:
            self.sentence_model = pickle.load(f)
        with open(word_start_model_path, "rb") as f:
            self.word_start_model = pickle.load(f)
        with open(word_end_model_path, "rb") as f:
            self.word_end_model = pickle.load(f)

    def find_answer(self, question: str, story: Story) -> str:
        """
        Compare the question with the story, and return the best answer.

        Parameters
        ----------
        question : str
            The current question being asked.
        story : Story
            The saved story.

        Returns
        -------
        str
            The best response to the given question.
        """
        sent_preds_in = ml_friendly_sentences(story, question)
        best_answer = ""
        best_score = 0
        for sent_pred_in, sentence in zip(sent_preds_in, story.sentences):
            sent_pred = self.sentence_model.predict_proba(sent_pred_in)
            word_preds_in = ml_friendly_words(sentence, question)
            words = sentence.split()
            for word_start_pred, word_a in zip(word_preds_in, words):
                start_pred = self.word_start_model.predict_proba(
                    word_start_pred)
                start_index = sentence.index(word_a)
                for word_end_pred, word_b in zip(word_preds_in, words):
                    end_pred = self.word_end_model.predict_proba(word_end_pred)
                    end_index = sentence.index(word_b)
                    score = sent_pred * start_pred * end_pred**1 / 3
                    if score > best_score:
                        best_answer = " ".join(words[start_index:end_index +
                                                     1])
                        best_score = score
        return best_answer

    def answer_questions(self, story: Story,
                         questions: List[Dict[str, str]]) -> None:
        """
        Answers the questions receieved from the questions list with the
        information saved in the story.

        Parameters
        ----------
        story : Story
            The story object.
        questions : List[Dict[str,str]]
            The list of question dictionaries.
        """
        for question_dict in questions:

            # Get and print question ID
            question_id = question_dict["QuestionID"]
            print(f"QuestionID: {question_id}")
            # Print the question itself
            question_text = question_dict["Question"]

            #print(f"Question: {question_text}")
            # Get question and run it through our answer function with the story

            answer = self.find_answer(question_text, story)
            # Print the answer
            print(f"Answer: {answer}")
            difficulty = question_dict["Difficulty"]
            # print(f"Difficulty: {difficulty}")
            print()