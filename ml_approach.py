import os
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     cross_validate, StratifiedKFold)

from sklearn.feature_selection import (SelectKBest, VarianceThreshold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   RobustScaler, StandardScaler,
                                   PowerTransformer)
from helpers import (NLP, Story, get_story_question_answers, text_f_score)
from sklearn.metrics import accuracy_score
from sentence_scorer import SentenceScorer
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import time
import warnings
from numpy.typing import ArrayLike
from tpot import TPOTClassifier
from pprint import pprint
from pathlib import Path
from config_dicts import classifier_config_dict_extra_fast as config_to_use
from functools import lru_cache
import json

warnings.filterwarnings("ignore")


def answer_in_sentence(answer: str, sentence: str) -> bool:
    """Returns True if the answer is in the sentence, False otherwise."""
    for accepted_answer in answer.split("|"):
        accepted_answer = accepted_answer.strip()
        if accepted_answer in sentence:
            return True
    return False


def get_representative_vectors(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
    full_signature_text: Story,
) -> Dict[str, np.ndarray]:
    """
    Get the representative vectors for each story and question-answer pair.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story and question-answer pairs to get the representative vectors
        for.
    full_signature_text : Story
        The full signature text to use to get the representative vectors.

    Returns
    -------
    Dict[str, np.ndarray]
        The representative vectors for each story and question-answer pair.
        Contains the full signature text embeddings calculated by word counts.
    """
    seen_signatures: Dict[str, np.ndarray] = {}
    for story_dict, question_answer_pairs in tqdm(story_qas):
        story_object = Story(story_dict)
        for question, _ in question_answer_pairs:
            # Get the embedding and signature vectors, and store them
            if question not in seen_signatures:
                vector = full_signature_text.get_sentence_vector(question)
                seen_signatures[question] = vector
            # Look at each sentence in the story
            for sentence in story_object.sentences:
                # And get the embedding and signature vectors for those too
                if sentence not in seen_signatures:
                    vector = full_signature_text.get_sentence_vector(sentence)
                    seen_signatures[sentence] = vector
    return seen_signatures


def get_or_create_all_words() -> List[str]:
    """
    Get the list of all words in the dataset. Does not contain duplicates of
    words.

    Returns
    -------
    List[str]
        The list of all words in the dataset.
    """
    try:
        with open("all_words.json", "r") as f:
            all_words = json.load(f)
    except FileNotFoundError:
        story_qas = get_story_question_answers('devset-official')

        # Add all words from stories to an input set so we can have a
        # constant-length signature vector
        story_texts = [story_dict['TEXT'] for story_dict, _ in story_qas]

        full_text = " ".join(story_texts).lower().split()
        all_words = list(set(full_text))
        with open("all_words.json", "w") as f:
            json.dump(all_words, f)
    return all_words


def ml_friendly_sentences(new_story: Story, question: str) -> np.ndarray:
    """
    Get the sentences in a story that are friendly for our machine learning
    model, so it's ready to go.

    Parameters
    ----------
    new_story : Story
        The story to get the sentences from.
    question : str
        The question to get the sentences from.
    full_story : Optional[Story], optional
        The full story to get the sentences from, by default None

    Returns
    -------
    np.ndarray
        The sentences in a story that are friendly for our model.
    """

    full_signature_text = Story(" ".join(get_or_create_all_words()))

    # Get representative vectors for question and each sentence in given story
    sent_sigs = []
    question_signature = full_signature_text.get_sentence_vector(question)
    q_sigs = []
    for sentence in new_story.sentences:
        q_sigs.append(question_signature)
        sent_sigs.append(full_signature_text.get_sentence_vector(sentence))

    signature_distances = get_distances(q_sigs, sent_sigs)

    story_type_values = []
    question_types = []

    for sentence in new_story.sentences:
        scores = SentenceScorer.get_sentence_scores(new_story, question,
                                                    sentence)
        story_type_values.append(scores)
        question_type = get_question_vector(question)
        question_types.append(question_type)

    story_type_values = np.array(story_type_values)
    question_types = np.array(question_types)
    return np.concatenate((
        signature_distances,
        story_type_values,
        question_types,
    ),
                          axis=1)


def ml_friendly_words(sentence: str, question: str):
    X = []
    start_y = []
    end_y = []
    words = sentence.split()
    # Get the question vector
    question_vector = get_question_vector(question)
    for index, word in enumerate(words):
        word_vec = NLP.word_vector(word)
        # Get the input vector
        input_vec = np.concatenate((question_vector, word_vec))
        X.append(input_vec)
        # Get the start and end labels
        start_label = 1 if index == start_index else 0
        end_label = 1 if index == end_index else 0
        start_y.append(start_label)
        end_y.append(end_label)
    return np.array(X), np.array(start_y), np.array(end_y)


@lru_cache(maxsize=None)
def get_question_vector(question: str) -> np.ndarray:
    """
    Get the question vector for the given question.

    Parameters
    ----------
    question : str
        The question to get the question vector for.

    Returns
    -------
    np.ndarray
        The question vector for the given question.
    """
    split_words = question.lower().split()
    first_word = split_words[0]
    second_word = split_words[1] if len(split_words) > 1 else ""
    question_type = [
        0,
        0,
        0,
        0,
        0,
        0,  # Who, What, When, Where, Why, How
        0,
        0,
        0,  # much/many/long/old/far, did/does/do, modals
    ]
    if first_word == "who":
        question_type[0] = 1
    elif first_word == "what":
        question_type[1] = 1
    elif first_word == "when":
        question_type[2] = 1
    elif first_word == "where":
        question_type[3] = 1
    elif first_word == "why":
        question_type[4] = 1
    elif first_word == "how":
        question_type[5] = 1
    if second_word in {
            "much", "many", "long", "old", "far", "large", "deep", "big",
            "high", "wide", "young", "short", "tall", "heavy", "light", "small"
    }:
        question_type[6] = 1
    elif second_word in {"did", "does", "do"}:
        question_type[7] = 1
    elif second_word in {
            "can", "could", "may", "might", "must", "shall", "should", "will",
            "would"
    }:
        question_type[8] = 1
    return np.array(question_type)


def unpack_stories(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
) -> Generator[Tuple[Story, str, str, str], None, None]:
    """
    A generator which yields story objects, questions, answers, and sentences.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story and question-answer pairs to unpack.

    Yields
    ------
    Generator[Tuple[Story, str, str, str], None, None]
        A tuple containing a story object, a question, an answer, and a
        sentence.
    """
    for story_dict, question_answer_pairs in story_qas:
        story_object = Story(story_dict)
        for question, answer in question_answer_pairs:
            for sentence in story_object.sentences:
                yield story_object, question, answer, sentence


def create_word_training_data(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the training data for the word prediction model.

    Returns
    -------
    Tuple[List[np.ndarray], List[int], List[int]]
        The training data for the word prediction model. Contains input data as
        a list of numpy arrays, followed by a list of class labels indicating
        whether the input data is a start word (or not), followed by a list of
        class labels indicating whether the input data is an end word (or not).
    """
    X = []
    start_y = []
    end_y = []
    for _, question, answers, sentence in unpack_stories(story_qas):
        # Bar indicates multiple answers
        split_answers = answers.split("|")
        found_answer = False
        answer_choice = ""
        for answer in split_answers:
            if not answer_in_sentence(answer, sentence):
                continue
            found_answer = True
            answer_choice = answer.strip()
            break
        if not found_answer:
            continue

        # Get the words in the sentence
        words = [token.text for token in NLP.nlp(sentence)]
        # Get the start and end indices of the answer in the sentence
        answer_words = [token.text for token in NLP.nlp(answer_choice)]
        if any(word not in words for word in answer_words):
            continue
        start_index = words.index(answer_words[0])
        end_index = words.index(answer_words[-1])

        # Get the question vector
        question_vector = get_question_vector(question)
        for index, word in enumerate(words):
            word_vec = NLP.word_vector(word)
            # Get the input vector
            input_vec = np.concatenate((question_vector, word_vec))
            X.append(input_vec)
            # Get the start and end labels
            start_label = 1 if index == start_index else 0
            end_label = 1 if index == end_index else 0
            start_y.append(start_label)
            end_y.append(end_label)
    return np.array(X), np.array(start_y), np.array(end_y)


def process_data(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
    seen_signatures: Dict[str, np.ndarray],
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Process the data for all of the stories and question-answer pairs using
    all signature vectors. Returns the target, question signatures, and sentence
    signatures.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story and question-answer pairs to process.
    seen_signatures : Dict[str, np.ndarray]
        The signature vectors for each sentence and question.

    Returns
    -------
    Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray],
    List[np.ndarray]]
        The target, question embeddings, sentence embeddings, question
        signatures, and sentence signatures.
    """
    y: List[int] = []
    question_signatures: List[np.ndarray] = []
    sentence_signatures: List[np.ndarray] = []
    # Now we have all the embeddings, all the signatures, and knowledge of
    # what the largest signature vector is. We can now create our X and y
    # (input and target output) to train models on
    for _, question, answer, sentence in tqdm(unpack_stories(story_qas),
                                              total=18152):
        # Grab the saved vectors
        question_signature = seen_signatures[question]

        sentence_signature = seen_signatures[sentence]
        # Is answer in sentence?
        target = 1 if answer_in_sentence(answer, sentence) else 0
        question_signatures.append(question_signature)
        sentence_signatures.append(sentence_signature)
        y.append(target)
    return (y, question_signatures, sentence_signatures)


def get_distances(questions: ArrayLike, sentences: ArrayLike) -> np.ndarray:
    """
    Get the distances between each question and sentence.

    Parameters
    ----------
    questions : ArrayLike
        The question vectors
    sentences : ArrayLike
        The sentence vectors

    Returns
    -------
    np.ndarray
        Various distances between each question and sentence.
    """
    # Questions and sentences are basically 2D arrays, where each row is a
    # vector. We want to get the distance between each question and each
    # sentence (they are already paired up)
    distances = []
    for question, sentence in zip(questions, sentences):  # type: ignore
        row_distance = []
        # Get the distance between the two vectors
        row_distance.append(distance.cosine(question, sentence))
        row_distance.append(distance.euclidean(question, sentence))
        row_distance.append(distance.braycurtis(question, sentence))
        row_distance.append(distance.canberra(question, sentence))
        row_distance.append(distance.chebyshev(question, sentence))
        row_distance.append(distance.cityblock(question, sentence))
        # row_distance.append(distance.correlation(question, sentence))
        row_distance.append(distance.minkowski(question, sentence, p=3))
        row_distance.append(distance.minkowski(question, sentence, p=4))
        row_distance.append(distance.sqeuclidean(question, sentence))
        distances.append(row_distance)
    return np.array(distances)


def collect_data() -> Tuple[List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
                            np.ndarray, np.ndarray]:
    """
    Collects the data from the stories and question-answer pairs, and returns
    them processed into X and y as the input and target output.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The input and target output for the model.
    """
    story_qas = get_story_question_answers('devset-official')

    # Add all words from stories to an input set so we can have a
    # constant-length signature vector
    story_texts = [story_dict['TEXT'] for story_dict, _ in story_qas]

    full_signature_text = Story(" ".join(story_texts))
    seen_signatures = get_representative_vectors(story_qas,
                                                 full_signature_text)
    print("Processing collected data")

    (y, q_sigs, sent_sigs) = process_data(story_qas, seen_signatures)
    signature_distances = get_distances(q_sigs, sent_sigs)

    story_type_values = []
    question_types = []

    for story, question, _, sentence in tqdm(unpack_stories(story_qas),
                                             total=18152):
        scores = SentenceScorer.get_sentence_scores(story, question, sentence)
        story_type_values.append(scores)
        question_type = get_question_vector(question)
        question_types.append(question_type)

    story_type_values = np.array(story_type_values)
    question_types = np.array(question_types)
    X = np.concatenate((
        signature_distances,
        story_type_values,
        question_types,
    ),
                       axis=1)

    X = np.array(X)
    y = np.array(y)
    stories = (story_qas, X, y)
    if not os.path.exists("data"):
        os.mkdir("data")
    with open("data/stories.pkl", "wb") as f:
        pickle.dump(stories, f)
    return stories


if __name__ == "__main__":
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

    (train_sentence_X, test_sentence_X, train_sentence_y,
     test_sentence_y) = train_test_split(sentence_X, sentence_y, test_size=0.2)

    (train_word_start_X, test_word_start_X, train_word_start_y,
     test_word_start_y) = train_test_split(word_X, word_start_y, test_size=0.2)
    (train_word_end_X, test_word_end_X, train_word_end_y,
     test_word_end_y) = train_test_split(word_X, word_end_y, test_size=0.2)
    # n_fits = fit_n_models(100, X, y, 60 * 5, checkpoint_folder="random_ml")
    # Sort n_fits by test score
    # pprint(n_fits.sort(key=lambda x: x[0], reverse=True))

    # print(f"Training set size: {len(train_x)}")
    # print(f"Test set size: {len(test_x)}")
    # all_results = {}
    # results = []
    start = time.time()

    print("Fitting pipeline")

    # train sentence model on sentences
    # get probability predictions for each sentence using trained model
    # train word models (start and end) on answers
    # for each sentence:
        #  get probability predictions for each start/end pair
        # start_pred, end_pred = sentence.prob(start_index),sentence.prob(end_index)
        #  start_index, end_index = sentence.index(start_pred), sentence.index(end_pred)
        # start_index, end_index = sentence.index(start_pred), sentence.index(
        #     end_pred)
    #  if geomean(start_pred, end_pred, sentence_pred) > best_score:
    #    best_score = geomean(start_pred, end_pred, sentence_pred)
    #    best_sentence = sentence
    # return best_sentence

    end = time.time()

    # classifier_config_dict['lightgbm.LGBMClassifier'] = {
    #     'boosting_type': ['gbdt', 'dart', 'rf'],
    #     'num_leaves': [
    #         2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150,
    #         200, 250, 500
    #     ],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators':
    #     [5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000],
    #     'min_child_samples':
    #     [1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000],
    #     'subsample': [0.66, 0.7, 0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.7, 0.9, 1.0],
    # }

    tpot = TPOTClassifier(
        generations=20,
        population_size=50,
        cv=5,
        verbosity=3,
        scoring="balanced_accuracy",
        periodic_checkpoint_folder='ml_checkpoints',
        config_dict=config_to_use,
        max_eval_time_mins=5,
    )
    tpot.fit(train_word_start_X, train_word_start_y)
    print(tpot.score(test_word_start_X, test_word_start_y))
    tpot.export('tpot_pipeline.py')