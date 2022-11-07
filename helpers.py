from collections import defaultdict
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import numpy.linalg as la
from numpy.typing import ArrayLike
import spacy


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
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained("bert-base-uncased",
                                          max_position_embeddings=512)
        if model is not None:
            self.model = model
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
        embedding = self.model(**encoded_input).pooler_output
        return torch.ravel(embedding).detach().numpy()
        # embedding = self.model(**encoded_input).last_hidden_state
        # return torch.ravel(torch.mean(embedding, dim=1)).detach().numpy()


class StoryHelper:
    """
    A class to help out with some important functions in answering questions
    about a story, such as separating the story into sentences and vectorizing
    individual sentences.
    """

    def __init__(self, story_text: str):
        # sourcery skip: set-comprehension
        """
        Parameters
        ----------
        story_text : str
            The text of the story.
        """
        self._story_text = story_text
        # Count each word in the story
        self._nlp = spacy.load("en_core_web_sm")
        self._doc = self._nlp(story_text)
        words = set()
        for token in self._doc:
            if token.is_stop or not token.is_alpha:
                continue
            words.add(token.lemma_)
        self._word_list = list(words)
        self._word_to_index = defaultdict(lambda: -1)
        for i, word in enumerate(self._word_list):
            self._word_to_index[word] = i
        self.sentences: List[str] = [sent.text for sent in self._doc.sents]
        """
        A list of sentences in the story. Each sentence is a string.
        """

    def get_sentence_vector(self, sentence: str) -> ArrayLike:
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
        word_vector = np.zeros(len(self._word_list))
        # Parse the sentence with spacy
        doc = self._nlp(sentence)

        for token in doc:
            # Skip stop words and punctuation
            if token.is_stop or not token.is_alpha:
                continue
            # Lemmatize the word so we have a consistent representation
            lemma = token.lemma_
            # If the word is in our word list, add it to the vector
            if lemma in self._word_to_index:
                index = self._word_to_index[lemma]
                if index == -1:
                    continue
                word_vector[index] += 1
        # Return the vector
        return word_vector

    def most_similar_signature(self, sentence: str) -> str:
        """
        Finds the sentence in the story most similar to `sentence` in terms of
        its signature vector.

        Parameters
        ----------
        sentence : str
            The sentence to compare to the story.

        Returns
        -------
        str
            The sentence in the story most similar to `sentence`.
        """
        # Get the vector representation of the sentence
        sentence_vector = self.get_sentence_vector(sentence)
        # Find the sentence with the highest cosine similarity
        best_sentence = ""
        best_similarity = -1
        # Look through each sentence in the story
        for story_sentence in self.sentences:
            # Vectorize the other sentence
            story_vector = self.get_sentence_vector(story_sentence)
            # See how similar it is to the sentence passed in
            similarity = cosine_similarity(sentence_vector, story_vector)
            # If it's the most similar so far, save it
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = story_sentence
        # Return the most similar sentence
        return best_sentence.strip()

    def most_similar_embedding(self, bert: Bert, sentence: str) -> str:
        """
        Finds the sentence in the story most similar to `sentence` in terms of
        its embedding vector.

        Parameters
        ----------
        sentence : str
            The sentence to compare to the story.

        Returns
        -------
        str
            The sentence in the story most similar to `sentence`.
        """
        # Get the vector representation of the sentence
        sentence_vector = bert.get_embeddings(sentence)
        # Find the sentence with the highest cosine similarity
        best_sentence = ""
        best_similarity = -1
        # Look through each sentence in the story
        for story_sentence in self.sentences:
            # Vectorize the other sentence
            story_vector = bert.get_embeddings(story_sentence)
            # See how similar it is to the sentence passed in
            similarity = cosine_similarity(sentence_vector, story_vector)
            # If it's the most similar so far, save it
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = story_sentence
        # Return the most similar sentence
        return best_sentence.strip()
