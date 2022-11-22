import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from helpers import (Bert, Story, get_story_question_answers, text_f_score)
from qa import find_answer
from terminalhelper import NEWLINE, VERBATIM, stringformat
from tqdm import tqdm

story_qas = get_story_question_answers('devset-official')

scores = []


def get_best_sentence(answer_str: str, story_object: Story) -> str:
    """
    Returns the best sentence from the story that answers the question.
    """
    best_sentence = ''
    best_score = 0
    answers = answer_str.split('|')
    answer_to_use = answers[0]
    answer_len = len(answer_to_use)
    for answer in answers:
        if len(answer) > answer_len:
            answer_to_use = answer
            answer_len = len(answer)
    answer_set = set(answer_to_use.split())
    for sentence in story_object.sentences:
        sentence_set = set(sentence.split())
        score = len(answer_set.intersection(sentence_set))
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence


for story_dict, question_answer_pairs in tqdm(story_qas):
    story_object = Story(story_dict)
    for question, answer in question_answer_pairs:
        # prediction = get_best_sentence(answer, story_object)
        prediction = find_answer(question, story_object)

        # Most similar BERT embedding, average f score of 0.0621117
        # Also took 17 minutes to run, so presumably running on CPU
        # prediction = story_help.most_similar_embedding(bert, question)

        # Most similar signature vector, average f score of 0.1869697
        # Took 3 minutes 17 seconds
        f_score = text_f_score(answer, prediction)
        scores.append(f_score)
print(f"\rAverage f score: {np.mean(scores)}")
print(f"Median f score: {np.median(scores)}")
print(f"Standard deviation of f score: {np.std(scores)}")
