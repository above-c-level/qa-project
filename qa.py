#! /usr/bin/env python3
import argparse
from typing import List, Set, Tuple
from terminalhelper import stringformat, VERBATIM, NEWLINE


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
{NEWLINE}{VERBATIM}An unexpected freeze has left dozens of the whales trapped in a sea of ice,
with one small hole as their only window for air. The open sea is 20 kilometres
away.
{NEWLINE}etc.
'''),
        metavar="<inputfile>",
        nargs=1,
    )
    args = parser.parse_args()
    inputfile = args.inputfile[0]
    return inputfile


if __name__ == "__main__":
    inputfile = parse_args()
    print(inputfile)
