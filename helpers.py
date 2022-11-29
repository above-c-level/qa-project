import os
import re
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import numpy.linalg as la
import spacy
from numpy.typing import ArrayLike


def cosine_similarity(v1: ArrayLike, v2: ArrayLike) -> float:
    """
    Computes the cosine similarity between two vectors.

    Parameters
    ----------
    v1 : ArrayLike
        The first vector.
    v2 : ArrayLike
        The second vector.

    Returns
    -------
    float
        The cosine similarity between the two vectors.
    """
    denominator = (la.norm(v1) * la.norm(v2))
    return np.dot(v1, v2) / denominator if denominator != 0 else 0


class NLP:
    """
    A helper class to load the NLP model from spacy
    """
    nlp = spacy.load("en_core_web_md")


class Story:
    """
    A class to help out with some important functions in answering questions
    about a story, such as separating the story into sentences and vectorizing
    individual sentences.
    """

    def __init__(self, story: Union[str, Dict[str, str]]):
        # sourcery skip: set-comprehension
        """
        Parameters
        ----------
        story_text : Union[str, Dict[str, str]]
            The information about the story.
        """
        # If we were given a dictionary, get the text from it
        if isinstance(story, dict):
            assert "TEXT" in story
            assert "STORYID" in story
            self.story_text = story["TEXT"]
            self.story_id = story["STORYID"]
        # Otherwise, assume we were given the text
        else:
            self.story_text = story
            self.story_id = None
        # Count each word in the story
        self.nlp = NLP.nlp
        self.doc = self.nlp(self.story_text)
        words = set()
        for token in self.doc:
            if token.is_stop or not token.is_alpha:
                continue
            words.add(token.lemma_)
        self.word_list = list(words)
        self.word_to_index = defaultdict(lambda: -1)
        for i, word in enumerate(self.word_list):
            self.word_to_index[word] = i
        self.sentences: List[str] = [sent.text for sent in self.doc.sents]
        """
        A list of sentences in the story. Each sentence is a string.
        """

    def __eq__(self, __other: object) -> bool:
        if isinstance(__other, Story):
            return self.story_text == __other.story_text
        return False

    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        """
        Get the vector representation of a sentence.

        Parameters
        ----------
        sentence : str
            The sentence to vectorize.

        Returns
        -------
        ArrayLike
            The vector representation of the sentence.
        """
        # Initialize the vector to zeros
        word_vector = np.zeros(len(self.word_list))
        # Parse the sentence with spacy
        doc = self.nlp(sentence)

        for token in doc:
            # Skip stop words and punctuation
            if token.is_stop or not token.is_alpha:
                continue
            # Lemmatize the word so we have a consistent representation
            lemma = token.lemma_
            # If the word is in our word list, add it to the vector
            if lemma in self.word_to_index:
                index = self.word_to_index[lemma]
                if index == -1:
                    continue
                word_vector[index] += 1
        # Return the vector
        return word_vector


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
