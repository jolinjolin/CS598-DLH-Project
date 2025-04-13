import numpy as np
import cv2
import torch

"""
Utility functions for various tasks including penalty building, tensor manipulation, 
and heatmap generation.

Functions:
----------
penalty_builder(penalty_config: str) -> Callable:
    Builds a penalty function based on the given configuration string.

length_wu(length: float, logprobs: float, alpha: float = 0.) -> float:
    Computes the length re-ranking score as described in "Google's Neural Machine 
    Translation System".

length_average(length: float, logprobs: float, alpha: float = 0.) -> float:
    Computes the average probability of tokens in a sequence.

split_tensors(n: int, x: Union[torch.Tensor, list, tuple, None]) -> Union[torch.Tensor, list, tuple, None]:
    Splits a tensor or collection into `n` parts along the batch dimension.

repeat_tensors(n: int, x: Union[torch.Tensor, list, tuple]) -> Union[torch.Tensor, list, tuple]:
    Repeats a tensor or collection `n` times along the batch dimension.

generate_heatmap(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    Generates a heatmap overlay on an image using the given weights.

Notes:
------
- The `penalty_builder` function supports two penalty types: 'wu' and 'avg'.
- The `split_tensors` and `repeat_tensors` functions handle nested collections 
  (lists or tuples) recursively.
- The `generate_heatmap` function assumes the input image is in CHW format and 
  converts it to HWC format for processing.
"""

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result