from collections import Counter
import json
import os
import random
import shutil
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import numpy.linalg as la
import torch

from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     cross_validate, StratifiedKFold)

from sklearn.feature_selection import (SelectKBest, VarianceThreshold)
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer,
                                   RobustScaler, StandardScaler,
                                   PowerTransformer)
from helpers import (Bert, Story, get_story_question_answers, text_f_score)
from sklearn.metrics import accuracy_score
from sentence_scorer import SentenceScorer
from scipy.spatial import distance
from tqdm import tqdm
from ml_model import best_model
import pickle
import time
import warnings
from numpy.typing import ArrayLike
from tpot import TPOTClassifier
from pprint import pprint
from pathlib import Path
from config_dicts import classifier_config_dict_extra_fast as config_to_use

warnings.filterwarnings("ignore")


def answer_in_sentence(answer: str, sentence: str) -> bool:
    """Returns True if the answer is in the sentence, False otherwise."""
    for accepted_answer in answer.split("|"):
        accepted_answer = accepted_answer.strip()
        if accepted_answer in sentence:
            return True
    return False


def get_scores_from_prob(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
    prob: np.ndarray,
    is_pred: bool = False,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns the recall, precision, and f1 scores for the given story_qas and
    predicted probabilities by class. For example, the results from
    sklearn.model_selection.cross_val_predict with method="predict_proba" can be
    passed in as prob.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story_qas to get the scores for.
    prob : np.ndarray
        The predicted probabilities by class.
    is_pred : bool, optional
        Whether or not the prob is the result of a call to
        sklearn.model_selection.cross_val_predict with method="predict". If
        True, then the probabilities are nonexistent, so we just grab the first
        sentence with value 1.

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        The recall, precision, and f1 scores for the given story_qas and
        predicted probabilities by class.
    """
    recalls = []
    precisions = []
    f_scores = []
    index = 0
    for story_dict, question_answer_pairs in tqdm(story_qas):
        story_object = Story(story_dict)
        for _, answer in question_answer_pairs:
            highest_probability = 0
            best_sentence = ""
            for sentence in story_object.sentences:
                probability = prob[index] if is_pred else prob[index][1]
                if probability > highest_probability:
                    highest_probability = probability
                    best_sentence = sentence
                index += 1
            prediction = best_sentence
            recall, precision, f_score = text_f_score(answer, prediction)
            recalls.append(recall)
            precisions.append(precision)
            f_scores.append(f_score)
    return recalls, precisions, f_scores


def get_balanced_data(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Narrows down data to be more balanced by predicted class. Since our data has
    far fewer 0s than 1s, we want to try to balance the data so that we can
    avoid overfitting to the 0s. This is done by randomly selecting a subset of
    the 0s to equal the number of 1s.

    Parameters
    ----------
    X : np.ndarray
        The data to narrow down.
    y : np.ndarray
        The targets of the data, where each x_i in X corresponds to y_i in y.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The balanced data and targets.
    """
    true_X = []
    true_y = []
    false_X = []
    false_y = []
    for input_data, target in zip(X, y):
        if target == 1:
            true_X.append(input_data)
            true_y.append(target)
        else:
            false_X.append(input_data)
            false_y.append(target)
    print(len(true_X))
    print(len(false_X))
    new_X = []
    new_y = []
    # Since we have so many more false examples than true examples, we'll
    # randomly sample from the false examples to get a more balanced dataset.
    # Because we know what true_X and false_X map to, we can shuffle both
    # of them
    random.shuffle(false_X)
    random.shuffle(true_X)

    for i in range(len(true_X)):
        new_X.append(true_X[i])
        new_y.append(1)
        new_X.append(false_X[i])
        new_y.append(0)

    new_X = np.array(new_X)
    new_y = np.array(new_y)
    return new_X, new_y


def get_representative_vectors(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
    bert: Bert,
    full_signature_text: Story,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Get the representative vectors for each story and question-answer pair.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story and question-answer pairs to get the representative vectors
        for.
    bert : Bert
        An instance of the Bert class to use to get the representative vectors.
    full_signature_text : Story
        The full signature text to use to get the representative vectors.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        The representative vectors for each story and question-answer pair.
        Contains both bert embeddings and the full signature text embeddings
        calculated by word counts.
    """
    seen_embeddings: Dict[str, np.ndarray] = {}
    seen_signatures: Dict[str, np.ndarray] = {}
    for story_dict, question_answer_pairs in tqdm(story_qas):
        story_object = Story(story_dict)
        for question, _ in question_answer_pairs:
            # Get the embedding and signature vectors, and store them
            if question not in seen_embeddings:
                seen_embeddings[question] = bert.get_embeddings(question)
            if question not in seen_signatures:
                vector = full_signature_text.get_sentence_vector(question)
                seen_signatures[question] = vector
            # Look at each sentence in the story
            for sentence in story_object.sentences:
                # And get the embedding and signature vectors for those too
                if sentence not in seen_embeddings:
                    seen_embeddings[sentence] = bert.get_embeddings(sentence)
                if sentence not in seen_signatures:
                    vector = full_signature_text.get_sentence_vector(sentence)
                    seen_signatures[sentence] = vector
    return seen_embeddings, seen_signatures


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
        A tuple containing a story object, a question, an answer, and a sentence.
    """
    for story_dict, question_answer_pairs in story_qas:
        story_object = Story(story_dict)
        for question, answer in question_answer_pairs:
            for sentence in story_object.sentences:
                yield story_object, question, answer, sentence


def process_data(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]],
    seen_embeddings: Dict[str, np.ndarray],
    seen_signatures: Dict[str, np.ndarray],
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray],
           List[np.ndarray]]:
    """
    Process the data for all of the stories and question-answer pairs using
    all embeddings and signature vectors. Returns the target, question
    embeddings, sentence embeddings, question signatures, and sentence
    signatures.

    Parameters
    ----------
    story_qas : List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
        The story and question-answer pairs to process.
    seen_embeddings : Dict[str, np.ndarray]
        The embeddings for each sentence and question.
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
    question_embeddings: List[np.ndarray] = []
    sentence_embeddings: List[np.ndarray] = []
    question_signatures: List[np.ndarray] = []
    sentence_signatures: List[np.ndarray] = []
    # Now we have all the embeddings, all the signatures, and knowledge of
    # what the largest signature vector is. We can now create our X and y
    # (input and target output) to train models on
    for _, question, answer, sentence in tqdm(unpack_stories(story_qas), total=18152):
        # Grab the saved vectors
        question_embedding = seen_embeddings[question]
        question_signature = seen_signatures[question]

        sentence_embedding = seen_embeddings[sentence]
        sentence_signature = seen_signatures[sentence]
        # Is answer in sentence?
        target = 1 if answer_in_sentence(answer, sentence) else 0
        question_embeddings.append(question_embedding)
        sentence_embeddings.append(sentence_embedding)
        question_signatures.append(question_signature)
        sentence_signatures.append(sentence_signature)
        y.append(target)
    return (y, question_embeddings, sentence_embeddings, question_signatures,
            sentence_signatures)


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
    bert = Bert()

    # Add all words from stories to an input set so we can have a
    # constant-length signature vector
    story_texts = [story_dict['TEXT'] for story_dict, _ in story_qas]

    full_signature_text = Story(" ".join(story_texts))
    seen_embeddings, seen_signatures = get_representative_vectors(
        story_qas, bert, full_signature_text)
    print("Processing collected data")

    (y, q_embeds, sent_embeds, q_sigs,
     sent_sigs) = process_data(story_qas, seen_embeddings, seen_signatures)
    embedding_distances = get_distances(q_embeds, sent_embeds)
    signature_distances = get_distances(q_sigs, sent_sigs)

    story_type_values = []
    for story, question, _, sentence in tqdm(unpack_stories(story_qas), total=18152):
        scores = SentenceScorer.get_sentence_scores(story, question, sentence)
        story_type_values.append(scores)

    story_type_values = np.array(story_type_values)
    question_types = np.array(question_types)
    X = np.concatenate((
        embedding_distances,
        signature_distances,
        story_type_values,
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
            story_qas, X, y = stories
    else:
        print("No stories.pkl found, generating")
        story_qas, X, y = collect_data()

    cv_model = StratifiedKFold(n_splits=50, shuffle=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    # n_fits = fit_n_models(100, X, y, 60 * 5, checkpoint_folder="random_ml")
    # Sort n_fits by test score
    # pprint(n_fits.sort(key=lambda x: x[0], reverse=True))

    # print(f"Training set size: {len(train_x)}")
    # print(f"Test set size: {len(test_x)}")
    # all_results = {}
    # results = []
    start = time.time()
    pipeline = best_model

    print("Fitting pipeline")
    try:
        cross_val = cross_val_predict(pipeline,
                                      X,
                                      y,
                                      cv=cv_model,
                                      method="predict_proba")
        cross_val = np.array(cross_val)
        recalls, precisions, f_scores = get_scores_from_prob(
            story_qas, cross_val)
    except AttributeError:
        cross_val = cross_val_predict(pipeline,
                                      X,
                                      y,
                                      cv=cv_model,
                                      method="predict")
        cross_val = np.array(cross_val)
        recalls, precisions, f_scores = get_scores_from_prob(story_qas,
                                                             cross_val,
                                                             is_pred=True)

    end = time.time()

    print(f"Time taken: {end - start}")
    print(f"Recall: {np.mean(recalls)}")
    print(f"Precision: {np.mean(precisions)}")
    print(f"Accuracy: {accuracy_score(y, np.argmax(cross_val, axis=1))}")
    f1 = 2 * (np.mean(precisions) * np.mean(recalls)) / (np.mean(precisions) +
                                                         np.mean(recalls))
    print(f"F1: {f1}")

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

    # tpot = TPOTClassifier(
    #     generations=20,
    #     population_size=100,
    #     cv=5,
    #     verbosity=3,
    #     scoring="f1",
    #     periodic_checkpoint_folder='ml_checkpoints',
    #     config_dict=config_to_use,
    #     max_eval_time_mins=5,
    # )
    # tpot.fit(train_X, train_y)
    # print(tpot.score(test_X, test_y))
    # tpot.export('tpot_pipeline.py')
    # times_taken = {}
