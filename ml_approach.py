import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     cross_validate, StratifiedKFold)
from sklearn.ensemble import ExtraTreesClassifier

from helpers import Bert, Story, get_story_question_answers, text_f_score
from qa import find_answer
from terminalhelper import NEWLINE, VERBATIM, stringformat
from tqdm import tqdm
from tpot import TPOTClassifier


def answer_in_sentence(answer: str, sentence: str) -> bool:
    """Returns True if the answer is in the sentence, False otherwise."""
    for accepted_answer in answer.split("|"):
        accepted_answer = accepted_answer.strip()
        if accepted_answer in sentence:
            return True
    return False


story_qas = get_story_question_answers('devset-official')
bert = Bert()
X = []
y = []
scores = []
for story_dict, question_answer_pairs in tqdm(story_qas):
    already_seen = {}
    story_object = Story(story_dict)
    for question, answer in question_answer_pairs:
        if question not in already_seen:
            already_seen[question] = bert.get_embeddings(question)
        question_embedding = already_seen[question]
        for sentence in story_object.sentences:
            if sentence not in already_seen:
                already_seen[sentence] = bert.get_embeddings(sentence)
            sentence_embedding = already_seen[sentence]
            # Is answer in sentence?
            target = 1 if answer_in_sentence(answer, sentence) else 0
            X.append(np.concatenate((question_embedding, sentence_embedding)))
            y.append(target)

tpot = TPOTClassifier(
    generations=100,
    population_size=100,
    verbosity=3,
    n_jobs=-1,
    scoring='f1',
    max_time_mins=60 * 8,
    early_stop=10,
    checkpoint_folder='ml_checkpoints',
)
X = np.array(X)
y = np.array(y)
tpot.fit(X, y)
print(tpot.score(X, y))
tpot.export('tpot_pipeline.py')

scores = []
index = 0
for story_dict, question_answer_pairs in tqdm(story_qas):
    story_object = Story(story_dict)
    for question, answer in question_answer_pairs:
        highest_probability = 0
        best_sentence = ""
        for sentence in story_object.sentences:
            probability = cross_val[index][1]
            if probability > highest_probability:
                highest_probability = probability
                best_sentence = sentence
            index += 1
        prediction = best_sentence
        f_score = text_f_score(answer, prediction)
        scores.append(f_score)
print(f"\rAverage f score: {np.mean(scores)}")
print(f"Median f score: {np.median(scores)}")
print(f"Standard deviation of f score: {np.std(scores)}")
