import os
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     cross_validate, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import (SelectKBest, VarianceThreshold)
from sklearn.preprocessing import StandardScaler

from helpers import Bert, Story, get_story_question_answers, text_f_score
from tqdm import tqdm
from ml_model import model, all_models
import pickle
import time


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


if os.path.exists("data/stories.pkl"):
    print("Loading stories from pickle file")
    with open("data/stories.pkl", "rb") as f:
        stories = pickle.load(f)
        story_qas, X, y = stories
else:
    print("No stories.pkl found, generating")
    story_qas = get_story_question_answers('devset-official')
    bert = Bert()
    X = []
    y = []
    scores = []
    largest_signature = 0
    seen_embeddings = {}
    seen_signatures = {}
    story_texts = []
    question_embeddings = []
    sentence_embeddings = []
    question_signatures = []
    sentence_signatures = []

    # Add all words from stories to an input set so we can have a
    # constant-length signature vector
    for story_dict, question_answer_pairs in story_qas:
        story_texts.append(story_dict['TEXT'])
    full_signature_text = Story(" ".join(story_texts))
    for story_dict, question_answer_pairs in tqdm(story_qas):
        story_object = Story(story_dict)
        for question, answer in question_answer_pairs:
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
    print("Processing collected data")

    # Now we have all the embeddings, all the signatures, and knowledge of
    # what the largest signature vector is. We can now create our X and y
    # (input and target output) to train models on
    for story_dict, question_answer_pairs in tqdm(story_qas):
        story_object = Story(story_dict)
        for question, answer in question_answer_pairs:
            # Grab the saved vectors
            question_embedding = seen_embeddings[question]
            question_signature = seen_signatures[question]
            for sentence in story_object.sentences:
                sentence_embedding = seen_embeddings[sentence]

                sentence_signature = seen_signatures[sentence]
                # Is answer in sentence?
                target = 1 if answer_in_sentence(answer, sentence) else 0
                question_embeddings.append(question_embedding)
                sentence_embeddings.append(sentence_embedding)
                question_signatures.append(question_signature)
                sentence_signatures.append(sentence_signature)
                y.append(target)

    question_embeddings = StandardScaler().fit_transform(question_embeddings)
    sentence_embeddings = StandardScaler().fit_transform(sentence_embeddings)
    question_signatures = StandardScaler().fit_transform(question_signatures)
    sentence_signatures = StandardScaler().fit_transform(sentence_signatures)
    embeddings_diff = question_embeddings - sentence_embeddings
    signatures_diff = question_signatures - sentence_signatures
    # Convert X to a numpy array
    X = np.concatenate((
        question_embeddings,
        sentence_embeddings,
        question_signatures,
        sentence_signatures,
        embeddings_diff,
        signatures_diff,
    ),
                       axis=1)
    X = np.array(X)
    y = np.array(y)
    stories = (story_qas, X, y)
    with open("data/stories.pkl", "wb") as f:
        pickle.dump(stories, f)

train_x, test_x, train_y, test_y = train_test_split(X,
                                                    y,
                                                    test_size=0.5,
                                                    shuffle=True)
# print(f"Training set size: {len(train_x)}")
# print(f"Test set size: {len(test_x)}")
results = []
for name, model_choice in all_models.items():
    print(f"Training {name}")
    start = time.time()
    cv_model = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipeline = make_pipeline(
        VarianceThreshold(),
        SelectKBest(k=10),
        model_choice,
    )
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
    print(f"Time: {end - start}")
    # tpot = TPOTClassifier(
    #     generations=10,
    #     population_size=50,
    #     verbosity=3,
    #     max_eval_time_mins=2,
    #     early_stop=5,
    #     periodic_checkpoint_folder='ml_checkpoints',
    #     memory='D:/tpot_cache/',
    #     cv=3,
    #     warm_start=True,
    # )
    # X = np.array(train_x)
    # y = np.array(train_y)
    # tpot.fit(X, y)
    # print(tpot.score(X, y))
    # tpot.export('tpot_pipeline.py')

    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    mean_f_score = np.mean(f_scores)
    recalc_f_score = (2 * mean_precision * mean_recall /
                      (mean_precision + mean_recall))
    print(f"Recall: {mean_recall}")
    print(f"Precision: {mean_precision}")
    # print(f"F-score: {mean_f_score}")
    print(f"Recalculated F score: {recalc_f_score}")
    print()
    results.append((
        name,
        end - start,
        mean_recall,
        mean_precision,
        recalc_f_score,
    ))

print(results)
