import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from helpers import Bert, StoryHelper, read_answers, read_questions, read_story
from qa import find_answer
from terminalhelper import NEWLINE, VERBATIM, stringformat




# scores = []
# for story, question_answer_pairs in story_qas:
#     story_text = story["TEXT"]
#     story_help = StoryHelper(story_text)
#     for question, answer in question_answer_pairs:
#         # The naive approach, average f score of 0.0034256
#         # Took 42 seconds to run
#         # prediction = find_answer(question, story)

#         # Most similar BERT embedding, average f score of 0.0621117
#         # Also took 17 minutes to run, so presumably running on CPU
#         # prediction = story_help.most_similar_embedding(bert, question)

#         # Most similar signature vector, average f score of 0.1869697
#         # Took 3 minutes 17 seconds
#         prediction = story_help.most_similar_signature(question)
#         f_score = text_f_score(answer, prediction)
#         scores.append(f_score)
# print(f"Average f score: {np.mean(scores)}")
# print(f"Median f score: {np.median(scores)}")
# print(f"Standard deviation of f score: {np.std(scores)}")