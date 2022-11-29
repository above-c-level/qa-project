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
from qa_controller import QA

TIMING = False


def parse_args():
    parser = argparse.ArgumentParser(
        "python3 qa.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="A question and answer program. Takes in a single file "
        "which contains the location of stories",
    )

    parser.add_argument(
        "inputfile",
        type=str,
        help=stringformat(f'''
The path to the input file. The first line of the input file is a directory
path. Each following line is a story ID. The program will then look for stories
matching the story ID in the directory path.
{NEWLINE}
For example, an input file might look like:
{NEWLINE}{NEWLINE}{VERBATIM}/home/cs5340/project/developset
{NEWLINE}{VERBATIM}1999-W02-5
{NEWLINE}{VERBATIM}1999-W03-5
{NEWLINE}{VERBATIM}1999-W04-5
{NEWLINE}{VERBATIM}1999-W05-4
{NEWLINE}{VERBATIM}1999-W05-5
{NEWLINE}{VERBATIM}1999-W06-5
{NEWLINE}{VERBATIM}1999-W07-5
{NEWLINE}{VERBATIM}1999-W08-1
{NEWLINE}Each story file must be formatted with key value pairs in the form of
capitalized keys followed by a colon, then followed by any value. The are, in
order, HEADLINE, DATE, STORYID, and TEXT. The TEXT key can be followed by any
number of lines of text. For example:
{NEWLINE}{VERBATIM}HEADLINE: An Arctic Struggle
{NEWLINE}{VERBATIM}DATE: June 29, 1999
{NEWLINE}{VERBATIM}STORYID: 1999-W27-2
{NEWLINE}
{NEWLINE}{VERBATIM}TEXT:
A group of 50 beluga whales is fighting to stay alive in an icy trap in the
Canadian Arctic near Ellesmere Island.
{NEWLINE}
{NEWLINE}{VERBATIM}An unexpected freeze has left dozens of the whales trapped
in a sea of ice, with one small hole as their only window for air. The open sea
is 20 kilometres away.
{NEWLINE}etc.
'''),
        metavar="<inputfile>",
        nargs=1,
    )
    args = parser.parse_args()
    inputfile = args.inputfile[0]
    return inputfile


def n_gram(questions: List[Dict[str, str]], n: int = 2) -> Dict[str, int]:
    """
    Creates a dictionary of n-grams from the beginnings of the questions,
    and returns a dictionary mapping each starting n-gram to the number of
    times it appears in the questions.

    Parameters
    ----------
    questions : List[Dict[str, str]]
        The list of question dictionaries.
    n : int, optional
        The length of the n-grams, by default 2

    Returns
    -------
    Dict[str, int]
        A dictionary mapping each starting n-gram to the number of times it
        appears in the questions.
    """
    n_gram_dict = {}
    for question_dict in questions:
        question_text = question_dict["Question"].lower()
        words = question_text.split()
        sliced = words[:0 + n]
        # if sliced[0] not in {"what"}:
        #     continue
        n_gram = " ".join(sliced)
        if n_gram in n_gram_dict:
            n_gram_dict[n_gram] += 1
        else:
            n_gram_dict[n_gram] = 1
    return n_gram_dict

def join_n_gram_dicts(
    dicts: List[Dict[str, int]]
    ) -> Dict[str, Tuple[int, Dict[str, Tuple[int, ...]]]]:
    """
    Takes a list of dictionaries mapping n-grams to counts, and returns a
    nested dictionary mapping each n-gram to a tuple containing the total
    number of times that n-gram appears in the dictionaries, and a
    dictionary mapping each (n+1)-gram to the number of times that it
    appears in the dictionaries.

    Parameters
    ----------
    dicts : List[Dict[str, int]]
        A list of dictionaries mapping n-grams to counts.

    Returns
    -------
    Dict[str, Tuple[int, Dict[str, Tuple[int, ...]]]]
        The nested dictionary mapping
    """
    filtered_dicts = []
    for d in dicts:
        current_dict = {k: v for k, v in d.items() if v > 1}
        filtered_dicts.append(current_dict)

    n_gram_dict = {}
    # Construct a tree of n-grams, so for example if we have the 1-gram "what"
    # showing up 10 times, the 2-gram "what is" showing up 7 times, the
    # 2-gram "what are" showing up 3 times, and the 3-gram "what is the"
    # showing up 5 times, the resulting dictionary would be:
    # {
    #     "what": (10, {
    #         "what is": (7, {
    #             "what is the": (5, {})
    #         }),
    #         "what are": (3, {})
    #     })
    # }
    for d in filtered_dicts:
        for key, value in d.items():
            words = key.split()
            if len(words) == 1:
                n_gram_dict[key] = (value, {})
            elif len(words) == 2:
                word_one = words[0]
                n_gram_dict[word_one][1][key] = (value, {})
            elif len(words) == 3:
                word_one = words[0]
                word_two = f"{words[0]} {words[1]}"
                n_gram_dict[word_one][1][word_two][1][key] = (value, {})
            elif len(words) == 4:
                word_one = words[0]
                word_two = f"{words[0]} {words[1]}"
                word_three = f"{words[0]} {words[1]} {words[2]}"
                n_gram_dict[word_one][1][word_two][1][word_three][1][key] = (value, {})
            elif len(words) == 5:
                word_one = words[0]
                word_two = f"{words[0]} {words[1]}"
                word_three = f"{words[0]} {words[1]} {words[2]}"
                word_four = f"{words[0]} {words[1]} {words[2]} {words[3]}"
                n_gram_dict[word_one][1][word_two][1][word_three][1][word_four][1][key] = (value, {})
            elif len(words) == 6:
                word_one = words[0]
                word_two = f"{words[0]} {words[1]}"
                word_three = f"{words[0]} {words[1]} {words[2]}"
                word_four = f"{words[0]} {words[1]} {words[2]} {words[3]}"
                word_five = f"{words[0]} {words[1]} {words[2]} {words[3]} {words[4]}"
                n_gram_dict[word_one][1][word_two][1][word_three][1][word_four][1][word_five][1][key] = (value, {})
    return n_gram_dict

def add_n_grams(questions, current_dict, n):
    n_gram_dict = n_gram(questions, n=n)
    for key in n_gram_dict:
        if key in current_dict:
            current_dict[key] += n_gram_dict[key]
        else:
            current_dict[key] = n_gram_dict[key]


if __name__ == "__main__":
    start = time.time()
    stories = 0
    inputfile = parse_args()
    if TIMING:
        print("Parsed args")
    with open(inputfile, "r") as f:
        lines = f.readlines()
        directory = lines[0].strip()
        lines = lines[1:]
    n_gram_total = {}
    if TIMING:
        print("Read input file")
    # Add all words from stories to an input set so we can have a
    # constant-length signature vector
    if TIMING:
        print("Collected story question answers")
    qa = QA()
    if TIMING:
        print("Created QA")
    # one_grams = {}
    # two_grams = {}
    # three_grams = {}
    # four_grams = {}
    # five_grams = {}
    # six_grams = {}
    for line in lines:
        story_id = line.strip()
        try:
            story_dict = read_story(directory, story_id)
            questions = read_questions(directory, story_id)
            story_object = Story(story_dict)

            # add_n_grams(questions, one_grams, 1)
            # add_n_grams(questions, two_grams, 2)
            # add_n_grams(questions, three_grams, 3)
            # add_n_grams(questions, four_grams, 4)
            # add_n_grams(questions, five_grams, 5)
            # add_n_grams(questions, six_grams, 6)

            qa.answer_questions(story_object, questions)
            stories += 1
        except FileNotFoundError:
            sys.stderr.write(f"Could not find story {story_id}")
            continue
        # print(story_object.story_id)
    end = time.time()
    if TIMING:
        total_time = end - start
        time_per_story = total_time / stories
        print(f"Took {time_per_story} seconds on average to answer a story")
        print(f"Took {total_time} seconds to answer {stories} stories")
    # Sort the n-grams by frequency
    # n_gram_total = join_n_gram_dicts([one_grams, two_grams, three_grams, four_grams])
    # pprint(n_gram_total)
