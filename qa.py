#! /usr/bin/env python3
import argparse
import os
import sys
import re
import time
from typing import Dict, List, Set, Tuple

from helpers import Bert, read_questions, read_story, Story
from terminalhelper import NEWLINE, VERBATIM, stringformat
import numpy as np
from pprint import pprint

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


def find_answer(question: str, story: Story) -> str:
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
    return story.most_similar_signature(question)


def answer_questions(story: Story, questions: List[Dict[str, str]]) -> None:
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

        answer = find_answer(question_text, story)
        # Print the answer
        print(f"Answer: {answer}")
        difficulty = question_dict["Difficulty"]
        # print(f"Difficulty: {difficulty}")
        print()

def n_gram(questions: List[Dict[str, str]], n: int = 2) -> Dict[str, int]:
    """
    Creates a dictionary of n-grams from the questions, and returns a dictionary
    mapping each n-gram to the number of times it appears in the questions.

    Parameters
    ----------
    questions : List[Dict[str, str]]
        The list of question dictionaries.
    n : int, optional
        The length of the n-grams, by default 2

    Returns
    -------
    Dict[str, int]
        The dictionary mapping each n-gram to the number of times it appears in
        the questions.
    """
    n_gram_dict = {}
    for question_dict in questions:
        question_text = question_dict["Question"].lower()
        words = question_text.split()
        for i in range(len(words) - n + 1):
            sliced = words[i : i + n]
            if sliced[0] not in {"how"}:
                continue
            n_gram = " ".join(sliced)
            if n_gram in n_gram_dict:
                n_gram_dict[n_gram] += 1
            else:
                n_gram_dict[n_gram] = 1
    return n_gram_dict

# def n_gram(questions: List[Dict[str, str]], n: int = 2) -> Dict[str, int]:
#     """
#     Creates a dictionary of n-grams from the questions, and returns a dictionary
#     mapping each n-gram to the number of times it appears in the questions.

#     Parameters
#     ----------
#     questions : List[Dict[str, str]]
#         The list of question dictionaries.
#     n : int, optional
#         The length of the n-grams, by default 2

#     Returns
#     -------
#     Dict[str, int]
#         The dictionary mapping each n-gram to the number of times it appears in
#         the questions.
#     """
#     n_gram_dict = {}
#     for question_dict in questions:
#         question_text = question_dict["Question"]
#         words = question_text.split()
#         for i in range(len(words) - n + 1):
#             n_gram = " ".join(words[i:i + n])
#             if "how" not in n_gram:
#                 continue
#             if n_gram in n_gram_dict:
#                 n_gram_dict[n_gram] += 1
#             else:
#                 n_gram_dict[n_gram] = 1
#     return n_gram_dict


if __name__ == "__main__":
    start = time.time()
    stories = 0
    inputfile = parse_args()
    with open(inputfile, "r") as f:
        lines = f.readlines()
        directory = lines[0].strip()
        lines = lines[1:]
    # n_gram_total = {}
    for line in lines:
        story_id = line .strip()
        try:
            story_dict = read_story(directory, story_id)
            questions = read_questions(directory, story_id)
            story_object = Story(story_dict)

            # n_gram_dict = n_gram(questions)
            # for key in n_gram_dict:
            #     if key in n_gram_total:
            #         n_gram_total[key] += n_gram_dict[key]
            #     else:
            #         n_gram_total[key] = n_gram_dict[key]
            answer_questions(story_object, questions)
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
    n_gram_total = sorted(n_gram_total.items(),
                          key=lambda x: x[1],
                          reverse=True)
    pprint(n_gram_total)
