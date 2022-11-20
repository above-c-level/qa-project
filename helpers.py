import os
import random
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.linalg as la
import spacy
import torch
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer


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


class ModelLoadError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Bert:
    """
    A helper class to load the BERT model and perform some operations
    """

    def __init__(self):
        """
        Load the BERT model and tokenizer from the HuggingFace library.

        Raises
        ------
        ModelLoadError
            If the model or tokenizer cannot be loaded.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained("bert-base-uncased",
                                          max_position_embeddings=512)
        if model is not None:
            self.model = model.to(self.device)
        else:
            raise ModelLoadError("Unable to load pretrained BERT model")

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get the embeddings for the given text.

        Parameters
        ----------
        text : str
            The text to get the embeddings for.

        Returns
        -------
        np.ndarray
            The embeddings for the given text as a numpy array.
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
        embedding = self.model(**encoded_input.to(self.device)).pooler_output
        return torch.ravel(embedding).cpu().detach().numpy()
        # embedding = self.model(**encoded_input).last_hidden_state
        # return torch.ravel(torch.mean(embedding, dim=1)).detach().numpy()


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

    def most_similar_signature(
        self,
        sentence: str,
        metric: Optional[Callable[[ArrayLike, ArrayLike],
                                  float]] = None) -> str:
        """
        Finds the sentence in the story most similar to `sentence` in terms of
        its signature vector.

        Parameters
        ----------
        sentence : str
            The sentence to compare to the story.
        metric : Optional[Callable[[ArrayLike, ArrayLike], float]], optional
            The metric to use to compare the signatures, by default None. If
            none is provided, the cosine similarity is used.

        Returns
        -------
        str
            The sentence in the story most similar to `sentence`.
        """
        if metric is None:
            metric = cosine_similarity
        # Get the vector representation of the sentence
        sentence_vector = self.get_sentence_vector(sentence)
        # Find the sentence with the highest cosine similarity
        best_sentence = ""
        best_similarity = -float("inf")
        # Look through each sentence in the story
        for story_sentence in self.sentences:
            # Vectorize the other sentence
            story_vector = self.get_sentence_vector(story_sentence)
            # See how similar it is to the sentence passed in
            similarity = metric(sentence_vector, story_vector)
            # If it's the most similar so far, save it
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = story_sentence
        # Return the most similar sentence
        return best_sentence.strip()

    def most_similar_embedding(
        self,
        bert: Bert,
        sentence: str,
        metric: Optional[Callable[[ArrayLike, ArrayLike],
                                  float]] = None) -> str:
        """
        Finds the sentence in the story most similar to `sentence` in terms of
        its embedding vector.

        Parameters
        ----------
        sentence : str
            The sentence to compare to the story.
        metric : Optional[Callable[[ArrayLike, ArrayLike], float]], optional
            The metric to use to compare the embeddings, by default None. If
            none is provided, the cosine similarity is used.

        Returns
        -------
        str
            The sentence in the story most similar to `sentence`.
        """
        if metric is None:
            metric = cosine_similarity
        # Get the vector representation of the sentence
        sentence_vector = bert.get_embeddings(sentence)
        # Find the sentence with the highest cosine similarity
        best_sentence = ""
        best_similarity = -float("inf")
        # Look through each sentence in the story
        for story_sentence in self.sentences:
            # Vectorize the other sentence
            story_vector = bert.get_embeddings(story_sentence)
            # See how similar it is to the sentence passed in
            similarity = metric(sentence_vector, story_vector)
            # If it's the most similar so far, save it
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = story_sentence
        # Return the most similar sentence
        return best_sentence.strip()


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


def read_answers(directory: str, story_id: str) -> List[Dict[str, str]]:
    """
    Read an answer file and return a list of dictionaries of key value pairs.

    Parameters
    ----------
    directory : str
        The directory path to the story file.
    story_id : str
        The story ID.

    Returns
    -------
    List[Dict[str, str]]
        A list of answers saved in a dictionary of key value pairs.
    """
    # Construct the path to the question file
    file_path = os.path.join(directory, story_id)
    file_path += ".answers"
    # Read the file
    read_data = ""
    with open(file_path, "r") as f:
        read_data = f.read()

    questions_re = re.compile(r"QuestionID\:(?:\s+)?(?P<QuestionID>(?:.|\n)*)"
                              r"Question\:(?:\s+)?(?P<Question>(?:.|\n)*)"
                              r"Answer\:(?:\s+)?(?P<Answer>(?:.|\n)*)"
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


def text_f_score(answer: str, prediction: str) -> Tuple[float, float, float]:
    """
    Returns the recall, precision, and f measure of a prediction compared
    with the answer

    Parameters
    ----------
    answer : str
        The ground truth answer(s)
    prediction : str
        The predicted answer

    Returns
    -------
    Tuple[float, float, float]
        The recall, precision, and f measure of the prediction, respectively
    """
    prediction = prediction.strip()
    # Unlike predictions, answers can have multiple valid answers
    # Split the answer into a list of answers about the pipe character (|)
    answers = [valid.strip() for valid in answer.split("|")]
    best_recall = -float('inf')
    best_precision = -float('inf')
    best_f_score = -float('inf')
    prediction_words = prediction.split()
    for valid in answers:
        # Recall is the number of correct words divided by the number of words
        # in the answer
        valid_words = valid.split()
        # Number of correct words in prediction
        correct_words = sum(word in valid_words for word in prediction_words)
        recall = correct_words / len(valid_words) if valid_words else 0
        # Precision is the number of correct words divided by the number of
        # words in the prediction
        precision = correct_words / len(
            prediction_words) if prediction_words else 0
        # f measure is the harmonic mean of precision and recall
        if recall == 0 and precision == 0:
            f_measure = 0
        else:
            f_measure = 2 * (precision * recall) / (precision + recall)
        if f_measure > best_f_score:
            best_f_score = f_measure
        if recall > best_recall:
            best_recall = recall
        if precision > best_precision:
            best_precision = precision
    # Return the best f measure given the prediction since there can be multiple
    # valid answers
    return best_recall, best_precision, best_f_score


def get_story_question_answers(
    directory: str,
    limit: Optional[int] = None
) -> List[Tuple[Dict[str, str], List[Tuple[str, str]]]]:
    """
    Get the story, question, and answer for each story in a directory.

    Parameters
    ----------
    directory : str
        The directory path to the story files.
    limit : Optional[int], optional
        The number of stories to read, by default None. If limit is None, all
        stories are read.

    Returns
    -------
    List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        A list of tuples of story, question, and answer.
    """
    # First, we need to find all the files in the directory that end with .story
    files = []
    for file in os.listdir(directory):
        if not file.endswith(".story"):
            continue
        if limit is not None and len(files) >= limit:
            break
        # Strip the file extension
        file = os.path.splitext(file)[0]
        files.append(file)
    story_qas = []
    for file in files:
        if limit is not None and len(story_qas) >= limit:
            break
        story = read_story(directory, file)
        answers = read_answers(directory, file)
        question_answer_pairs = [(group["Question"], group["Answer"])
                                 for group in answers]

        story_qas.append((story, question_answer_pairs))
    return story_qas
