#! /usr/bin/env python3
import argparse
from typing import Dict, List, Set, Tuple
from terminalhelper import stringformat, VERBATIM, NEWLINE
import os
import re


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


def read_story(directory: str, story_id: str) -> Dict[str, str]:
    """
    Read a story file and return a dictionary of key value pairs.

    Parameters
    ----------
    directory : str
        The directory path to the story file.
    story_id : str
        The story ID.

    Returns
    -------
    Dict[str, str]
        A dictionary of key value pairs.
    """
    file_path = os.path.join(directory, story_id)
    file_path += ".story"
    story_re = re.compile(r"HEADLINE\:(?:\s+)?(?P<HEADLINE>(?:.|\n)*)"
                          r"DATE\:(?:\s+)?(?P<DATE>(?:.|\n)*)"
                          r"STORYID\:(?:\s+)?(?P<STORYID>(?:.|\n)*)"
                          r"TEXT\:(?:\s+)?(?P<TEXT>(?:.|\n)*)")
    read_data = ""
    with open(file_path, "r") as f:
        read_data = f.read()

    match = story_re.match(read_data)
    if not match:
        raise ValueError("Invalid story file format.")
    groupdict = match.groupdict()
    space_re = re.compile(r"(\n|\s)+")
    for key, value in match.groupdict().items():
        groupdict[key] = value.strip()
        # Remove extra spaces
        groupdict[key] = space_re.sub(" ", groupdict[key])
    return groupdict


def read_questions(directory: str, story_id: str) -> List[Dict[str, str]]:
    """
    Read a question file and return a list of dictionaries of key value pairs.

    Parameters
    ----------
    directory : str
        The directory path to the story file.
    story_id : str
        The story ID.

    Returns
    -------
    List[Dict[str, str]]
        A list of question saved in a dictionary of key value pairs.
    """
    # Construct the path to the question file
    file_path = os.path.join(directory, story_id)
    file_path += ".questions"
    # Read the file
    read_data = ""
    with open(file_path, "r") as f:
        read_data = f.read()

    questions_re = re.compile(r"QuestionID\:(?:\s+)?(?P<QuestionID>(?:.|\n)*)"
                              r"Question\:(?:\s+)?(?P<Question>(?:.|\n)*)"
                              r"Difficulty\:(?:\s+)?(?P<Difficulty>(?:.|\n)*)")
    # Questions are separated by double newline
    question_groups = read_data.split("\n\n")
    question_dicts = []
    for group in question_groups:
        # Match questions as well as the question ID and difficulty
        group = group.strip()
        match = questions_re.match(group)
        if not match:
            continue
        groupdict = match.groupdict()
        for key, value in match.groupdict().items():
            # Clean up leading and trailing whitespace
            groupdict[key] = value.strip()
        # Add question info to the list
        question_dicts.append(groupdict)
    return question_dicts


def find_answer(question: str, story: Dict[str, str]) -> str:
    """
    Compare the question with the story, and return the best answer.

    Parameters
    ----------
    question : str
        The current question being asked.
    story : Dict[str, str]
        The saved story.

    Returns
    -------
    str
        The best response to the given question.
    """
    answer = "every answer is correct?"
    return answer


def answer_questions(story: Dict[str, str],
                     questions: List[Dict[str, str]]) -> None:
    """
    Answers the questions receieved from the questions list with the
    information saved in the story.

    Parameters
    ----------
    story : Dict[str,str]
        The story dictionary.
    questions : List[Dict[str,str]]
        The list of question dictionaries.
    """
    for question_dict in questions:
        # Get and print question ID
        question_id = question_dict["QuestionID"]
        print(f"QuestionID: {question_id}")
        # Print the question itself
        question_text = question_dict["Question"]
        print(f"Question: {question_text}")
        # Get question and run it through our answer function with the story
        answer = find_answer(question_text, story)
        # Print the answer
        print(f"Answer: {answer}")
        difficulty = question_dict["Difficulty"]
        print(f"Difficulty: {difficulty}")


if __name__ == "__main__":
    inputfile = parse_args()
    with open(inputfile, "r") as f:
        lines = f.readlines()
        directory = lines[0].strip()
        lines = lines[1:]
    for line in lines:
        story_id = line.strip()
        try:
            story = read_story(directory, story_id)
            questions = read_questions(directory, story_id)
            answer_questions(story, questions)
        except FileNotFoundError:
            print(f"Could not find story {story_id}")
            continue
        print(story["STORYID"])
