from typing import Dict, List, Protocol

from helpers import (Story, cosine_similarity)
import numpy as np
from sentence_scorer import SentenceScorer, Best

TIMING = False


class QA:
    """
    A question answerer class. Acts as a controller to bind together the machine
    learning models found in ml_approach.py and ml_models.py with the driver in
    qa.py. Requires that the models be trained and saved in the models folder.
    """

    def __init__(self):
        pass

    def find_answer(self, question: str, story: Story) -> str:
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
        p = 0.94
        split = question.lower().split()

        if split[0] == "who":
            fun = SentenceScorer.get_who_score
        elif split[0] == "what":
            fun = SentenceScorer.get_what_score
        elif split[0] == "when":
            fun = SentenceScorer.get_when_score
        elif split[0] == "where":
            fun = SentenceScorer.get_where_score
        elif split[0] == "why":
            Best.update_story(story, question)
            fun = SentenceScorer.get_why_score
        elif split[0] == "how":
            fun = SentenceScorer.get_how_score
        else:
            fun = SentenceScorer.get_what_score

        best_sentence = ""
        best_score = 0
        q_vec = story.get_sentence_vector(question)
        for sentence in story.sentences:
            fun_score = fun(question, sentence)
            s_vec = story.get_sentence_vector(sentence)
            cos_sim = cosine_similarity(q_vec, s_vec)
            score = p * cos_sim + (1 - p) * fun_score
            if score > best_score:
                best_score = score
                best_sentence = sentence
        sentence = best_prediction = best_sentence
        if len(best_prediction) == 0:
            best_prediction = best_sentence

        best_prediction = SentenceScorer.filter_answer(question,
                                                       best_prediction)
        return best_prediction

    def answer_questions(self, story: Story,
                         questions: List[Dict[str, str]]) -> None:
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

            answer = self.find_answer(question_text, story)
            # Print the answer
            # answer = filter_answer(answer)
            print(f"Answer: {answer}")
            difficulty = question_dict["Difficulty"]
            # print(f"Difficulty: {difficulty}")
            print()
