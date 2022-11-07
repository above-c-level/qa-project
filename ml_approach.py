import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from helpers import Bert, Story, get_story_question_answers, text_f_score
from qa import find_answer
from terminalhelper import NEWLINE, VERBATIM, stringformat
from tqdm import tqdm

story_qas = get_story_question_answers('devset-official')

scores = []
for story_dict, question_answer_pairs in tqdm(story_qas):
    story_object = Story(story_dict)

    for question, answer in question_answer_pairs:
        # The naive approach, average f score of 0.0034256
        # Took 42 seconds to run
        prediction = find_answer(question, story_object)

        # Most similar BERT embedding, average f score of 0.0621117
        # Also took 17 minutes to run, so presumably running on CPU
        # prediction = story_help.most_similar_embedding(bert, question)

        # Most similar signature vector, average f score of 0.1869697
        # Took 3 minutes 17 seconds
        f_score = text_f_score(answer, prediction)
        scores.append(f_score)
print(f"Average f score: {np.mean(scores)}")
print(f"Median f score: {np.median(scores)}")
print(f"Standard deviation of f score: {np.std(scores)}")