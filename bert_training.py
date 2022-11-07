import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from helpers import Bert, StoryHelper, get_story_question_answers

bert = Bert()
tokenizer = bert.tokenizer
# If we ever need to see the tokens from ids, we can use:
# tokenizer.convert_ids_to_tokens(input_ids)
sep_token_id = tokenizer.sep_token_id
assert sep_token_id is not None
story_qas = get_story_question_answers('devset-official')


def get_training_data(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
) -> List[Tuple[List[int], List[int], List[int]]]:
    training_info = []
    # Bert doesn't have built-in question answering with the model we're allowed
    # to use, so we need to train a head on top of the model to predict the span
    # of tokens that answer the question
    for story, question_answer_pairs in story_qas:
        # Get the story text
        story_text = story["TEXT"]
        story_help = StoryHelper(story_text)

        for question, answer in question_answer_pairs:
            # If the length of the story text is greater than 512-len(question),
            # we'll split the story up into three parts (first half, last half,
            # middle 50%). The answer is the same for all three parts
            parts = []
            if len(story_text) > 512 - len(question):
                # Split the story into three parts by sentence boundaries
                sentences = story_help.sentences
                first_half = " ".join(sentences[:len(sentences) // 2])
                last_half = " ".join(sentences[len(sentences) // 2:])
                middle = " ".join(sentences[len(sentences) // 4:3 *
                                            len(sentences) // 4])
                parts.extend((tokenizer.encode(question, first_half),
                              tokenizer.encode(question, middle),
                              tokenizer.encode(question, last_half)))
            else:
                encoded = tokenizer.encode(question, story_text)
                parts.append(encoded)
            for input_ids in parts:
                sep_index = input_ids.index(sep_token_id)
                # The length of the question and story
                question_token_count = sep_index + 1
                story_token_count = len(input_ids) - question_token_count
                # Token type ids are 0 for question and 1 for story so that we can
                # tell the model where the question and story are
                token_type_ids = [0] * question_token_count + [
                    1
                ] * story_token_count
                # When there are multiple answers, we need to train the model on all
                # of them
                answer_tokens = answer.split("|")
                answer_token_ids = [
                    tokenizer.encode(answer_token)
                    for answer_token in answer_tokens
                ]
                training_info.extend((input_ids, token_type_ids, answer_ids)
                                     for answer_ids in answer_token_ids)
    return training_info


training_info = get_training_data(story_qas)

model = bert.model
# Add another couple layers on top of the model so we can fine-tune it
model.add_module("qa_head", torch.nn.Linear(768, 2))
# Only train the qa_head layer
model.qa_head.train()

# Adam tends to be a pretty solid optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use device
model = model.to(device)

# Let's split the data into training and validation sets
training_info, validation_info = train_test_split(training_info,
                                                  test_size=0.2,
                                                  shuffle=True)


# Mostly just for debugging purposes
def token(ids_to_tokenize):
    return ' '.join(tokenizer.convert_ids_to_tokens(ids_to_tokenize))


def loss(pred, true):
    return torch.nn.functional.binary_cross_entropy_with_logits(pred, true)


epochs = 10
# minimum_val_loss = 0.22842
minimum_val_loss = 0.9
minimum_val_loss_epoch = -1
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    # Shuffle the training data
    random.shuffle(training_info)
    # Keep track of the loss
    epoch_loss = 0
    items = 1
    for input_ids, token_type_ids, answer_ids in training_info:
        print(f"Loss: {epoch_loss / items}", end="\r")
        # We can't accept anything longer than 512 tokens, so we need to
        # truncate the story (for now, later we can iterate over the story
        # if it's too long)
        if len(input_ids) > 512:
            print("Skipping")
            continue
        items += 1
        # Strip out the [CLS] and [SEP] tokens from the answer
        answer_ids = answer_ids[1:-1]
        # For debugging purposes
        input_tokens = token(input_ids)
        answer_tokens = token(answer_ids)
        # Get the start and end indices of the answer
        # Since the first word could hypothetically show up multiple times in
        # the story, we need to find all instances of it and then find the
        # span that matches exactly with the answer
        start_index = None
        end_index = None
        for i in range(len(input_ids)):
            if input_ids[i:i + len(answer_ids)] == answer_ids:
                start_index = i
                end_index = i + len(answer_ids) - 1
                break
        # If we didn't find the answer, skip this example
        if start_index is None or end_index is None:
            continue
        # Convert the input to tensors
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        token_type_ids_tensor = torch.tensor(token_type_ids).unsqueeze(0)
        # Get the output from the model
        pred = model(input_ids_tensor.to(device),
                     token_type_ids=token_type_ids_tensor.to(device))
        # The logits of the predictions for the start and end tokens
        start_logits_pred = pred[0].squeeze(0)
        end_logits_pred = pred[1].squeeze(0)
        # Probabilities of the predictions for the start and end tokens
        start_probs_pred = torch.nn.functional.softmax(start_logits_pred,
                                                       dim=0)
        end_probs_pred = torch.nn.functional.softmax(end_logits_pred, dim=0)
        # The shape of the start and end logits is [x, 768], where x is the
        # number of tokens in the input. But the start and end indices are
        # just scalars, so we need to expand them to [x, 768] so we can
        # calculate the loss
        start_index_tensor = torch.tensor(start_index).unsqueeze(0).to(device)
        end_index_tensor = torch.tensor(end_index).unsqueeze(0).to(device)
        start_logits_true = torch.zeros_like(start_logits_pred).to(device)
        end_logits_true = torch.zeros_like(end_logits_pred).to(device)
        # If we use .scatter_(0, start_index_tensor, 1), we get an error about
        # how the index tensor must have the same number of dimensions as the
        # self tensor. So we need to unsqueeze the start_index_tensor and
        # end_index_tensor
        start_logits_true.scatter_(0, start_index_tensor.unsqueeze(0), 1)
        end_logits_true.scatter_(0, end_index_tensor, 1)
        # Calculate the loss
        start_loss = loss(start_logits_pred, start_logits_true)
        end_loss = loss(end_logits_pred, end_logits_true)
        total_loss = start_loss + end_loss

        # Backpropagate
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of the loss
        epoch_loss += total_loss.item()
    print(f"\rLoss: {epoch_loss / items}")
    # Also print validation loss
    validation_loss = 0
    items = 1
    for input_ids, token_type_ids, answer_ids in training_info:
        print(f"Loss: {validation_loss / items}", end="\r")
        # We can't accept anything longer than 512 tokens, so we need to
        # truncate the story (for now, later we can iterate over the story
        # if it's too long)
        if len(input_ids) > 512:
            print("Skipping")
            continue
        items += 1
        # Strip out the [CLS] and [SEP] tokens from the answer
        answer_ids = answer_ids[1:-1]
        # For debugging purposes
        input_tokens = token(input_ids)
        answer_tokens = token(answer_ids)
        # Get the start and end indices of the answer
        # Since the first word could hypothetically show up multiple times in
        # the story, we need to find all instances of it and then find the
        # span that matches exactly with the answer
        start_index = None
        end_index = None
        for i in range(len(input_ids)):
            if input_ids[i:i + len(answer_ids)] == answer_ids:
                start_index = i
                end_index = i + len(answer_ids) - 1
                break
        # If we didn't find the answer, skip this example
        if start_index is None or end_index is None:
            continue
        # Convert the input to tensors
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        token_type_ids_tensor = torch.tensor(token_type_ids).unsqueeze(0)
        # Get the output from the model
        if torch.cuda.is_available():
            pred = model(input_ids_tensor.cuda(),
                         token_type_ids=token_type_ids_tensor.cuda())
        else:
            pred = model(input_ids_tensor,
                         token_type_ids=token_type_ids_tensor)
        # The logits of the predictions for the start and end tokens
        start_logits_pred = pred[0].squeeze(0)
        end_logits_pred = pred[1].squeeze(0)
        # Probabilities of the predictions for the start and end tokens
        start_probs_pred = torch.nn.functional.softmax(start_logits_pred,
                                                       dim=0)
        end_probs_pred = torch.nn.functional.softmax(end_logits_pred, dim=0)
        # The shape of the start and end logits is [x, 768], where x is the
        # number of tokens in the input. But the start and end indices are
        # just scalars, so we need to expand them to [x, 768] so we can
        # calculate the loss
        start_index_tensor = torch.tensor(start_index).unsqueeze(0).to(device)
        end_index_tensor = torch.tensor(end_index).unsqueeze(0).to(device)
        start_logits_true = torch.zeros_like(start_logits_pred).to(device)
        end_logits_true = torch.zeros_like(end_logits_pred).to(device)
        # If we use .scatter_(0, start_index_tensor, 1), we get an error about
        # how the index tensor must have the same number of dimensions as the
        # self tensor. So we need to unsqueeze the start_index_tensor and
        # end_index_tensor
        start_logits_true.scatter_(0, start_index_tensor.unsqueeze(0), 1)
        end_logits_true.scatter_(0, end_index_tensor, 1)
        # Calculate the loss
        start_loss = loss(start_logits_pred, start_logits_true)
        end_loss = loss(end_logits_pred, end_logits_true)
        total_loss = start_loss + end_loss
        # Keep track of the loss
        validation_loss += total_loss.item()
    average_val_loss = validation_loss / items
    print(f"\rValidation loss: {average_val_loss}")
    if average_val_loss < minimum_val_loss:
        minimum_val_loss = average_val_loss
        minimum_val_loss_epoch = epoch
        # Save the model
        torch.save(model.state_dict(), "model.pt")
