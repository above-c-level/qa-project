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
from sklearn.base import BaseEstimator, ClassifierMixin
from qa_controller import QA
import pickle
from ml_approach import collect_data, create_word_training_data
from ml_models import sentence_model, start_word_model, end_word_model
TIMING = False


def parse_args():
    parser = argparse.ArgumentParser(
        "python3 train_models.py",
        description="Trains models for the QA system",
    )

    parser.add_argument(
        "train_dir",
        type=str,
        help='The path to the training directory. The training directory ' 
        'should contain .answers, .story, and .questions files.',
        metavar="<train_dir>",
        nargs=1,
    )
    args = parser.parse_args()
    train_dir = args.train_dir[0]
    return train_dir

if __name__ == "__main__":
    start = time.time()
    train_dir = parse_args()
    if os.path.exists("data/stories.pkl"):
        print("Loading stories from pickle file")
        with open("data/stories.pkl", "rb") as f:
            stories = pickle.load(f)
            story_qas, sentence_X, sentence_y = stories
    else:
        print("No stories.pkl found, generating")
        stories = collect_data()
        story_qas, sentence_X, sentence_y = stories
        if not os.path.exists("data"):
            os.mkdir("data")
        with open("data/stories.pkl", "wb") as f:
            pickle.dump(stories, f)

    if os.path.exists("data/word_train.pkl"):
        print("Loading word training from pickle file")
        with open("data/word_train.pkl", "rb") as f:
            results = pickle.load(f)
            word_X, word_start_y, word_end_y = results
    else:
        print("No word_train.pkl found, generating")
        results = create_word_training_data(story_qas)
        word_X, word_start_y, word_end_y = results
        with open("data/word_train.pkl", "wb") as f:
            pickle.dump(results, f)
    
    print("Training sentence model")
    sentence_model.fit(sentence_X, sentence_y)
    with open("models/sentence_model.pkl", "wb") as f:
        pickle.dump(sentence_model, f)

    print("Training start word model")
    start_word_model.fit(word_X, word_start_y)
    with open("models/start_word_model.pkl", "wb") as f:
        pickle.dump(start_word_model, f)
    
    print("Training end word model")
    end_word_model.fit(word_X, word_end_y)
    with open("models/end_word_model.pkl", "wb") as f:
        pickle.dump(end_word_model, f)
    end = time.time()
    print(f"Total time: {(end - start)/60} minutes")