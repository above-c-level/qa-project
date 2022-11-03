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
    with open(file_path, "r") as f:
        lines = f.readlines()

    story = {"HEADLINE": lines[0].split(":")[1].strip()}
    story["DATE"] = lines[1].split(":")[1].strip()
    story["STORYID"] = lines[2].split(":")[1].strip()
    # Every single line after TEXT is part of the story
    for line in lines[5:]:
        story["TEXT"] = story.get("TEXT", "") + line
    story["TEXT"] = story["TEXT"].strip()
    # Remove multiple whitespace and replace with single space
    story["TEXT"] = re.sub(r"\s+", " ", story["TEXT"])
    return story


if __name__ == "__main__":
    inputfile = parse_args()
    with open(inputfile, "r") as f:
        lines = f.readlines()
        directory = lines[0].strip()
        lines = lines[1:]
    for line in lines:
        story_id = line.strip()
        story_id += ".story"
        try:
            story = read_story(directory, story_id)
        except FileNotFoundError:
            print(f"Could not find story {story_id}")
            continue
        print(story["STORYID"])


# foreach question in questions:
#    Find the answer()
#    Print QuestionID to console
#    Print Question to console
#    Print Answer to console