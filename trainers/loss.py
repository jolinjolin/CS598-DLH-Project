import torch
import torch.nn as nn

"""
This module defines a custom loss function for language modeling tasks.

Classes:
    LanguageModelCriterion(nn.Module):
        A custom loss function that computes the negative log likelihood loss
        for a language model, taking into account a mask to ignore padding tokens.

Functions:
    compute_loss(output, reports_ids, reports_masks):
        Computes the loss for a language model using the LanguageModelCriterion.

        Args:
            output (torch.Tensor): The predicted output from the model, 
                with shape (batch_size, sequence_length, vocab_size).
            reports_ids (torch.Tensor): The ground truth token IDs, 
                with shape (batch_size, sequence_length).
            reports_masks (torch.Tensor): A mask tensor indicating valid tokens 
                (1 for valid, 0 for padding), with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The computed loss value as a scalar tensor.
"""

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq > 0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output