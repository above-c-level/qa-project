from collections import Counter, defaultdict
from functools import lru_cache
from typing import Tuple, Dict
from helpers import NLP, Story


class Best:
    sentences: Dict[str, int] = defaultdict(lambda: 0)
    """
    Holds the best sentences in a story. The key is the sentence, the value is
    a tuple of booleans indicating whether it is one of the best sentences,
    whether it precedes one of the best sentences, and whether it follows one
    of the best sentences.
    """
    current_story = Story("hello there")
    question = "general kenobi"

    @staticmethod
    def update_story(story: Story,
                     question: str,
                     percentage_to_keep: float = 0.25):
        if story == Best.current_story and question == Best.question:
            return
        if question != Best.question:
            Best.question = question
        if story != Best.current_story:
            Best.sentences = defaultdict(lambda: 0)
            scores = {
                sentence: SentenceScorer.word_match(sentence, question)
                for sentence in story.sentences
            }

            scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # We want to keep the top 10% of sentences
            to_keep = {
                sentence
                for sentence, _ in
                scores[:int(len(scores) * percentage_to_keep)]
            }

            for index, item in enumerate(story.sentences):
                if item in to_keep:
                    Best.sentences[item] |= 1
                if index > 0 and story.sentences[index - 1] in to_keep:
                    Best.sentences[item] |= 2
                if (index < len(story.sentences) - 1
                        and story.sentences[index + 1] in to_keep):
                    Best.sentences[item] |= 4

    @staticmethod
    def get(item: str) -> Tuple[bool, bool, bool]:
        return (Best.sentences[item] & 1 == 1, Best.sentences[item] & 2 == 2,
                Best.sentences[item] & 4 == 4)


class SentenceScorer:
    clue = 3
    good_clue = 4
    confident = 6
    slam_dunk = 20

    @staticmethod
    @lru_cache(maxsize=None)
    def word_match(question: str, sentence: str) -> int:
        """
        Gets the number of words which are contained in both the question and the
        sentence.

        Parameters
        ----------
        question : str
            The question.
        sentence : str
            The sentence.

        Returns
        -------
        int
            The number of words which are in both the question and sentence.
        """
        question_words = Counter(question.lower().split())
        sentence_words = Counter(sentence.lower().split())
        score = 0
        for word in question_words:
            if word in sentence_words:
                score += 1
                # score += min(question_words[word], sentence_words[word])
        return score

    @staticmethod
    def get_who_score(question: str, sentence: str) -> int:
        """
        Collects the who score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        score = 0
        score += SentenceScorer.word_match(question, sentence)
        q_matched = NLP.nlp(question).ents
        s_matched = NLP.nlp(sentence).ents
        # print(f"{q_matched = }")
        # print(f"{s_matched = }")
        match_types = [token.label_ for token in s_matched]
        # print(f"{match_types = }")
        name_in_q = any(token.label_ == "PERSON" for token in q_matched)
        name_in_s = any(token.label_ == "PERSON" for token in s_matched)
        org_in_s = any(token.label_ == "ORG" for token in s_matched)
        # print(f"{name_in_q = }")
        # print(f"{name_in_s = }")
        if not name_in_q and name_in_s:
            score += SentenceScorer.confident
        if not name_in_q and "name" in question.lower():
            score += SentenceScorer.good_clue
        if org_in_s or name_in_s:
            score += SentenceScorer.good_clue
        return score

    @staticmethod
    def get_what_score(question: str, sentence: str) -> int:
        """
        Calculates the what score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        score = 0
        score += SentenceScorer.word_match(question, sentence)
        MONTH = {
            "january", "febuary", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december"
        }
        RECENCY = {"today", "yesterday", "tomorrow", "last night"}
        q_matched = NLP.nlp(question).ents
        s_matched = NLP.nlp(sentence).ents
        month_in_q = any(word in MONTH for word in sentence.lower().split())
        recency_in_s = any(word in RECENCY
                           for word in sentence.lower().split())
        if month_in_q and recency_in_s:
            score += SentenceScorer.clue
        lower_q_split = question.lower().split()
        lower_s_split = sentence.lower().split()
        if ("kind" in lower_q_split
                and any(word in {"call", "from"} for word in lower_s_split)):
            score += SentenceScorer.good_clue
        if ("name" in lower_q_split and any(word in {"name", "call", "known"}
                                            for word in lower_s_split)):
            score += SentenceScorer.slam_dunk
        # Find prep phrases starting with "name"/"named", then check whether
        # sentence contains a proper noun. If it does and the proper noun
        # contains the *head* of the prepositional phrase, then it's a slam dunk.
        doc = NLP.nlp(sentence)
        for chunk in doc.noun_chunks:
            if (chunk.root.dep_ == "pobj"
                    and chunk.root.head.text.lower() in {"name", "named"}
                    and any(token.pos_ == "PROPN" for token in chunk)):
                score += SentenceScorer.slam_dunk
                break

        if any(token.label_ == "EVENT" for token in s_matched):
            score += SentenceScorer.good_clue

        return score

    @staticmethod
    def get_when_score(question: str, sentence: str) -> int:
        """
        Calculates the when score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        score = 0
        s_matched = NLP.nlp(sentence).ents
        if any(token.label_ == "TIME" for token in s_matched):
            score += SentenceScorer.good_clue
            score += SentenceScorer.word_match(question, sentence)
        lower_q = question.lower()
        lower_s_split = sentence.lower().split()
        if ("the last" in lower_q
                and any(word in {"first", "last", "since", "ago"}
                        for word in lower_s_split)):
            score += SentenceScorer.slam_dunk
        if (any(word in {"start", "begin"} for word in lower_q.split())
                and any(word in {"start", "begin", "since", "year"}
                        for word in lower_s_split)):
            score += SentenceScorer.slam_dunk
        # if any(token.label_ == "DATE" for token in s_matched):
        #     score += SentenceScorer.confident

        return score

    @staticmethod
    def get_where_score(question: str, sentence: str) -> int:
        """
        Calculates the where score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        score = 0
        score += SentenceScorer.word_match(question, sentence)
        location_prep = {
            "in", "at", "on", "by", "above", "under", "beside", "between",
            "among", "amongst", "near", "inside", "outside", "around",
            "against", "behind", "across", "into"
        }
        s_matched = NLP.nlp(sentence).ents
        if any(word in location_prep for word in sentence.lower().split()):
            score += SentenceScorer.good_clue
        if any(token.label_ in {"GPE", "LOC"} for token in s_matched):
            score += SentenceScorer.good_clue
        return score

    @staticmethod
    def get_why_score(question: str, sentence: str) -> int:
        """
        Calculates the why score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        score = 0
        lower_s = sentence.lower().split()
        in_best, pre_best, post_best = Best.get(sentence)
        if in_best:
            score += SentenceScorer.clue
        if pre_best:
            score += SentenceScorer.clue
        if post_best:
            score += SentenceScorer.good_clue
        if "want" in lower_s:
            score += SentenceScorer.good_clue
        if any(word in {"so", "because"} for word in lower_s):
            score += SentenceScorer.good_clue

        return score

    @staticmethod
    def get_how_score(question: str, sentence: str) -> int:
        """
        Calculates the who score from a sentence based on a question.

        Parameters
        ---------
        question: str
            The question
        sentence: str
            The sentence

        Returns
        -------
        int
            The resulting score
        """
        q_matched = NLP.nlp(question).ents
        s_matched = NLP.nlp(sentence).ents
        score = 0
        score += SentenceScorer.word_match(question, sentence)
        lower_q = question.lower().split()
        lower_s = sentence.lower().split()
        second_word = lower_q[1]
        if second_word in {"does", "do"} and "by" in lower_s:
            score += SentenceScorer.clue
        if second_word in {
                "much", "many", "old", "far", "large", "deep", "big", "high",
                "wide", "young", "short", "tall", "heavy", "light", "small"
        } and any(token.label_ in
                  {"PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
                  for token in s_matched):
            score += SentenceScorer.slam_dunk
        if second_word == "long" and any(
                token.label_ in {"DATE", "TIME", "QUANTITY"} for token in s_matched):
            score += SentenceScorer.confident
        if second_word == "long" and any(
                token.label_ in {"CARDINAL"} for token in q_matched):
            score += SentenceScorer.clue
        return score

    @staticmethod
    def filter_answer(question: str, sentence: str) -> str:
        """
        Filter out named entities for the answer given a guess for the sentence
        containing the answer.

        Parameters
        ----------
        question : str
            The best answer question being asked.
        sentence : str
            The best answer to the question.

        Returns
        -------
        str
            The best response to the given question.
        """
        if not sentence:
            return sentence

        filtered_answer = ""

        sentence_matched = NLP.nlp(sentence).ents
        lower_q = question.lower()

        question_split = lower_q.split()
        first_word = question_split[0]
        second_word = question_split[1]
        third_word = question_split[2]
        lower_s = sentence.lower()
        sentence_split = lower_s.split()

        if first_word == "who":
            for token in sentence_matched:
                if token.label_ == "PERSON":
                    filtered_answer += token.text + " "
                elif token.label_ == "ORG":
                    filtered_answer += token.text + " "
                elif token.label_ == "GPE":
                    filtered_answer += token.text + " "

        elif first_word == "what":
            if "name" in lower_q:
                for token in sentence_matched:
                    if token.label_ == "PERSON":
                        filtered_answer += token.text + " "
            if "did" == second_word and "the" == third_word:
                for token in sentence_matched:
                    if token.text.split()[0] == question_split[3]:
                        token_index = sentence_split.index(token.text.split()[0])
                        filtered_answer += (" ".join(sentence_split[token_index:]) + " ")

            elif "is" in lower_q:
                for token in sentence_matched:
                    if token.label_ == "ORG":
                        filtered_answer += token.text + " "
                    if token.label_ == "DATE":
                        filtered_answer += token.text + " "
                    if token.label_ == "LOC":
                        filtered_answer += token.text + " "
                    if token.label_ == "TIME":
                        filtered_answer += token.text + " "

        elif first_word == "when":
            for token in sentence_matched:
                if token.label_ == "DATE":
                    filtered_answer += token.text + " "
                elif token.label_ == "TIME":
                    filtered_answer += token.text + " "

        elif first_word == "where":
            for token in sentence_matched:
                if token.label_ == "LOC":
                    filtered_answer += token.text + " "
                elif token.label_ == "GPE":
                    filtered_answer += token.text + " "
                elif token.label_ == "NORP":
                    filtered_answer += token.text + " "
                elif token.label_ == "ORG":
                    filtered_answer += token.text + " "

        elif first_word == "why":
            if "did" == second_word and "the" == third_word:
                for token in sentence_matched:
                    if token.text.split()[0] == question_split[3]:
                        token_index = sentence_split.index(token.text.split()[0])
                        filtered_answer += (" ".join(sentence_split[token_index:]) + " ")

        elif first_word == "how":
            if second_word in {"does", "do"}:
                for token in sentence_matched:
                    if token.label_ == "LOC":
                        filtered_answer += token.text + " "

            if second_word in {
                    "much", "many", "old", "far", "large", "deep", "big", "high",
                    "wide", "young", "short", "tall", "heavy", "light", "small"
            }:
                for token in sentence_matched:
                    if token.label_ == "PERCENT":
                        filtered_answer += token.text + " "
                    if token.label_ == "MONEY":
                        filtered_answer += token.text + " "
                    if token.label_ == "QUANTITY":
                        filtered_answer += token.text + " "
                    if token.label_ == "ORDINAL":
                        filtered_answer += token.text + " "
                    if token.label_ == "CARDINAL":
                        filtered_answer += token.text + " "

            elif second_word == "long":
                for token in sentence_matched:
                    if token.label_ == "DATE":
                        filtered_answer += token.text + " "
                    if token.label_ == "TIME":
                        filtered_answer += token.text + " "
                    if token.label_ == "QUANTITY":
                        filtered_answer += token.text + " "

            if second_word == "did" and third_word == "the":
                for token in sentence_matched:
                    if token.text.split()[0] == question_split[3]:
                        token_index = sentence_split.index(token.text.split()[0])
                        filtered_answer += (" ".join(sentence_split[token_index:]) + " ")

        filtered_answer = filtered_answer.strip()
        if len(filtered_answer) == 0:
            filtered_answer = sentence
        return filtered_answer


    @staticmethod
    def get_sentence_scores(story: Story, question: str,
                            sentence: str) -> Tuple[int, int, int, int, int]:
        """
        Collects the data from the stories and question-answer pairs, and
        returns them processed into X and y as the input and target output.

        Returns
        -------
        Tuple[int, int, int, int, int]
            The scores for each sentence from
        """
        Best.update_story(story, question)
        who_score = SentenceScorer.get_who_score(question, sentence)
        what_score = SentenceScorer.get_what_score(question, sentence)
        when_score = SentenceScorer.get_when_score(question, sentence)
        where_score = SentenceScorer.get_where_score(question, sentence)
        why_score = SentenceScorer.get_why_score(question, sentence)
        #how_score = SentenceScorer.get_how_score(question, sentence)
        return who_score, what_score, when_score, where_score, why_score
