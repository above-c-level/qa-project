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
from ml_model import model
import pickle


def answer_in_sentence(answer: str, sentence: str) -> bool:
    """Returns True if the answer is in the sentence, False otherwise."""
    for accepted_answer in answer.split("|"):
        accepted_answer = accepted_answer.strip()
        if accepted_answer in sentence:
            return True
    return False


if os.path.exists("data/stories.pkl"):
    print("Loading stories from pickle file")
    with open("data/stories.pkl", "rb") as f:
        stories = pickle.load(f)
        story_qas, X, y = stories
else:
    print("No stories.pkl found, generating")
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
                X.append(
                    np.concatenate((question_embedding, sentence_embedding)))
                y.append(target)
    X = np.array(X)
    y = np.array(y)
    stories = (story_qas, X, y)
    with open("data/stories.pkl", "wb") as f:
        pickle.dump(stories, f)

# train_x, test_x, train_y, test_y = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     shuffle=False)
# print(f"Training set size: {len(train_x)}")
# print(f"Test set size: {len(test_x)}")
cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val = cross_val_predict(model, X, y, cv=cv_model, method="predict_proba")

# tpot = TPOTClassifier(
#     generations=10,
#     population_size=100,
#     verbosity=3,
#     scoring='f1',
#     max_eval_time_mins=2,
#     early_stop=10,
#     periodic_checkpoint_folder='ml_checkpoints',
#     memory='cache',
#     warm_start=True,
# )
# X = np.array(train_x)
# y = np.array(train_y)
# tpot.fit(X, y)
# print(tpot.score(X, y))
# tpot.export('tpot_pipeline.py')

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
