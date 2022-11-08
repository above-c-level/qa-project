import os
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from helpers import (Bert, Story, cosine_similarity,
                     get_story_question_answers)

bert = Bert()
tokenizer = bert.tokenizer
# If we ever need to see the tokens from ids, we can use:
# tokenizer.convert_ids_to_tokens(input_ids)
sep_token_id = tokenizer.sep_token_id
assert sep_token_id is not None
story_qas = get_story_question_answers('devset-official')


# Mostly just for debugging purposes
def token(ids_to_tokenize):
    return ' '.join(tokenizer.convert_ids_to_tokens(ids_to_tokenize))


def nprep(tensor):
    return tensor.cpu().detach().numpy()


def get_training_data(
    story_qas: List[Tuple[Dict[str, str], List[Tuple[str, str]]]]
) -> List[Tuple[List[int], List[int], List[int]]]:
    training_info = []
    # Bert doesn't have built-in question answering with the model we're allowed
    # to use, so we need to train a head on top of the model to predict the span
    # of tokens that answer the question
    for story, question_answer_pairs in story_qas:
        # Get the story text
        story_help = Story(story)
        story_text = story_help.story_text
        for question, answer in question_answer_pairs:
            parts = []
            encoded = tokenizer.encode(question, story_text)
            if len(encoded) <= 512:
                parts.append(encoded)
            else:
                question_tokens = tokenizer.encode(question)
                # Split the story into two or more parts, trying to keep as
                # much of the story included as possible
                sentences = story_help.sentences
                first_half = ""
                for sentence in sentences:
                    first_half_tokens = tokenizer.encode(first_half)
                    sentence_tokens = tokenizer.encode(sentence)
                    if len(first_half_tokens) + len(question_tokens) + len(
                            sentence_tokens) < 512:
                        # If the sentence still fits, we'll add it to the first
                        # half
                        first_half += sentence
                    else:
                        break
                # Also do the last half of the story
                last_half = ""
                for sentence in reversed(sentences):
                    last_half_tokens = tokenizer.encode(last_half)
                    sentence_tokens = tokenizer.encode(sentence)
                    if len(last_half_tokens) + len(question_tokens) + len(
                            sentence_tokens) < 512:
                        # If the sentence still fits, we'll add it to the last
                        # half
                        last_half = sentence + last_half
                    else:
                        break
                first_half_tokens = tokenizer.encode(question, first_half)
                last_half_tokens = tokenizer.encode(question, last_half)

                # if len(first_half_tokens) > 512:
                #     print(f"Length tokens: {len(first_half_tokens)}")
                #     print(f"Question: {question}")
                #     print(f"Question length: {len(question_tokens)}")
                #     print(f"Tokens: {token(first_half_tokens)}")
                # if len(last_half_tokens) > 512:
                #     print(f"Length tokens: {len(last_half_tokens)}")
                #     print(f"Question: {question}")
                #     print(f"Question length: {len(question_tokens)}")
                #     print(f"Tokens: {token(last_half_tokens)}")
                parts = [first_half_tokens, last_half_tokens]
                # TODO: Possibly add middle half later

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
model.add_module("qa_start", torch.nn.Linear(768, 768))
model.add_module("qa_end", torch.nn.Linear(768, 768))
model.add_module(
    "qa_transformer_start",
    torch.nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=768,
        dropout=0.1,
        activation="relu",
    ),
)
# Only train the question answering head
model.qa_start.train()
model.qa_end.train()

# Adam tends to be a pretty solid optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use device
model = model.to(device)

# Let's split the data into training and validation sets
training_info, validation_info = train_test_split(training_info,
                                                  test_size=0.2,
                                                  shuffle=True)


def loss(pred, true):
    # return torch.nn.functional.l1_loss(pred, true)
    # return torch.nn.functional.mse_loss(pred, true)
    # return torch.nn.functional.huber_loss(pred, true)
    return torch.nn.functional.smooth_l1_loss(pred, true)


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
        items += 1
        # Convert the input to tensors
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        token_type_ids_tensor = torch.tensor(token_type_ids).unsqueeze(0)
        # Get the output from the model
        pred = model(input_ids_tensor.to(device),
                     token_type_ids=token_type_ids_tensor.to(device))
        last_hidden_state = pred[0]
        pooler_output = pred[1]
        # Get the start and end outputs of qa_start and qa_end on the
        # pooler output
        start_pred = model.qa_start(pooler_output).squeeze(0)
        end_pred = model.qa_end(pooler_output).squeeze(0)
        ground_truth_start = last_hidden_state.squeeze(0)[start_index]
        ground_truth_end = last_hidden_state.squeeze(0)[end_index]
        # Get the start and end loss
        start_loss = loss(start_pred, ground_truth_start)
        end_loss = loss(end_pred, ground_truth_end)

        # Backpropagate the loss for start_pred and end_pred
        start_loss.backward(retain_graph=True)
        optimizer.zero_grad()

        end_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss = start_loss + end_loss
        # Keep track of the loss
        epoch_loss += total_loss.item()
    print(f"\rLoss: {epoch_loss / items}")
    # Also print validation loss
    validation_loss = 0
    items = 1
    for input_ids, token_type_ids, answer_ids in training_info:
        print(f"Loss: {validation_loss / items}", end="\r")

        # Strip out the [CLS] and [SEP] tokens from the answer
        answer_ids = answer_ids[1:-1]
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
        items += 1
        # Convert the input to tensors
        input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        token_type_ids_tensor = torch.tensor(token_type_ids).unsqueeze(0)
        # Get the output from the model
        pred = model(input_ids_tensor.to(device),
                     token_type_ids=token_type_ids_tensor.to(device))
        last_hidden_state = pred[0]
        pooler_output = pred[1]

        # Get the start and end outputs of qa_start and qa_end on the
        # pooler output
        start_pred = model.qa_start(pooler_output).squeeze(0)
        end_pred = model.qa_end(pooler_output).squeeze(0)
        if epoch % 9 == 0:
            # Print out what the question was, which is the input_ids where the
            # token type ids are 0
            question = [
                input_ids[i] for i in range(len(input_ids))
                if token_type_ids[i] == 0
            ]
            print(f"Question: {token(question)}")
            # Print out the observed answer
            print(f"Observed answer: {token(answer_ids)}")

            # Find the tokens in the hidden state that are most similar to the
            # start and end predictions
            start_pred_np = start_pred.cpu().detach().numpy()
            end_pred_np = end_pred.cpu().detach().numpy()
            last_hidden_state_np = last_hidden_state.squeeze(
                0).cpu().detach().numpy()
            start_similarities = cosine_similarity_matrix(
                start_pred_np, last_hidden_state_np)
            end_similarities = cosine_similarity_matrix(
                end_pred_np, last_hidden_state_np)
            # Get the indices of the most similar tokens
            start_index_pred = np.argmax(start_similarities)
            end_index_pred = np.argmax(end_similarities)
            print(f"Start index: {start_index_pred}")
            print(f"End index: {end_index_pred}")
            predicted_answer = input_ids[start_index_pred:end_index_pred + 1]
            print(f"Predicted answer: {token(predicted_answer)}")

        start_pred = model.qa_start(pooler_output).squeeze(0)
        end_pred = model.qa_end(pooler_output).squeeze(0)
        ground_truth_start = last_hidden_state.squeeze(0)[start_index]
        ground_truth_end = last_hidden_state.squeeze(0)[end_index]
        # Get the start and end loss
        start_loss = loss(start_pred, ground_truth_start)
        end_loss = loss(end_pred, ground_truth_end)
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