# test_nlp_intro.py

import torch
import torch.nn as nn
from torchtyping import TensorType
from torch.nn.utils.rnn import pad_sequence
from typing import List

# Intro to Natural Language Processing
# Inputs:

# positive - a list of strings, each with positive emotion
# negative - a list of strings, each with negative emotion

# Constraints:

# positive.length = n
# negative.length = n

import torch
import torch.nn as nn
from torchtyping import TensorType
from torch.nn.utils.rnn import pad_sequence
from typing import List

# Intro to Natural Language Processing
# In this problem, you will load in a raw body of text and set it up for training. ChatGPT uses the entire text of the internet for training, but in this problem we will use Amazon product reviews and Tweets from X.

# Your task is to encode the input dataset of strings as an integer tensor of size 
# 2
# ⋅
# 𝑁
# ×
# 𝑇
# 2⋅N×T, where 
# 𝑇
# T is the length of the longest string. The lexicographically first word should be represented as 1, the second should be 2, and so on. In the final tensor, list the positive encodings, in order, before the negative encodings.

# Inputs:

# positive - a list of strings, each with positive emotion
# negative - a list of strings, each with negative emotion

# Constraints:

# positive.length = n
# negative.length = n

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_words = set()
        combined = positive + negative
        for phrase in combined:
            words = phrase.split()
            for word in words:
                all_words.add(word)
                
        sorted_words = sorted(list(all_words))
        sorted_map = {}
        for i, w in enumerate(sorted_words):
            if w not in sorted_map:
                sorted_map[w] = float(i + 1)

        tokenized_tensor_ls = []
        for phrase in combined:
            words = phrase.split()
            tokenized = []
            for word in words:
                tokenized.append(sorted_map[word])
            tokenized_tensor_ls.append(torch.tensor(tokenized, dtype=torch.float32))
    
        dataset = pad_sequence(tokenized_tensor_ls,
                               batch_first = True,
                               padding_value = 0.)
        return dataset


class BasicSolution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_words = []
        for phrase in positive:
            words = phrase.split()
            for word in words:
                all_words.append(word)
        for phrase in negative:
            words = phrase.split()
            for word in words:
                all_words.append(word)
                
        all_words.sort()
        sorted = all_words
        sorted_map = {}
        for i, w in enumerate(sorted):
            sorted_map[w] = i + 1

        print(f"sorted_map: {sorted_map}")

        # tokenized_positive_ls = []
        for phrase in positive:
            words = phrase.split()
            tokenized_positive = []
            for i, word in enumerate(words):
                tokenized_positive.append(sorted_map[word])
            # tokenized_positive_ls.append(tokenized_positive)
            tokenized_positive_t = torch.tensor(tokenized_positive, dtype=torch.float32)

        # tokenized_negative_ls = []
        for phrase in negative:
            words = phrase.split()
            tokenized_negative = []
            for i, word in enumerate(words):
                tokenized_negative.append(sorted_map[word])
            # tokenized_negative_ls.append(tokenized_negative)
            tokenized_negative_t = torch.tensor(tokenized_negative, dtype=torch.float32)

        dataset = pad_sequence([tokenized_positive_t, tokenized_negative_t],
                               batch_first = True,
                               padding_value = 0.)
        return dataset

def test_get_dataset_1():
    # Example 1:

    # Input:
    positive = ["Dogecoin to the moon"]
    negative = ["I will short Tesla today"]

    # Output: 
    output = [
      [1.0, 7.0, 6.0, 4.0, 0.0],
      [2.0, 9.0, 5.0, 3.0, 8.0]
    ]
    # Explanation: Lexicographically, we have Dogecoin -> 1, I -> 2, Tesla -> 3, etc. 0 is added to pad the first sentence.
    # Note: torch.nn.utils.rnn.pad_sequence() will be useful for padding a list of variable length tensors into a rectangular, non-jagged tensor. Use 0 for padding set batch_first=True.
    solver = Solution()
    data = solver.get_dataset(positive, negative)
    print(f"data={data}")
    assert data is not None
    assert data.shape == torch.Size([2, 5])
    assert torch.allclose(torch.tensor(output), data), f"allclose is False, output={torch.tensor(output)}, data={data}"

def test_get_dataset_2():

    positive=["Good case, Excellent value.","Great for the jawbone.","The mic is great.","If you are Razr owner...you must have this!","Highly recommend for any one who has a blue tooth phone"]
    negative=["So there is no way for me to plug it in here in the US unless I go buy a converter.","Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!!","I have to jiggle the plug to get it to line up right to get decent volume.","Needless to say, I wasted my money.","What a waste of money and time!"]

    output = [[3.0,22.0,2.0,67.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[4.0,27.0,59.0,37.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[12.0,42.0,35.0,30.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[7.0,73.0,19.0,10.0,52.0,47.0,32.0,61.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[5.0,55.0,27.0,18.0,51.0,72.0,31.0,16.0,20.0,64.0,53.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[11.0,60.0,35.0,49.0,71.0,27.0,41.0,63.0,54.0,36.0,34.0,33.0,34.0,59.0,14.0,65.0,6.0,29.0,21.0,16.0,25.0],[13.0,63.0,23.0,27.0,24.0,39.0,46.0,58.0,1.0,43.0,9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[6.0,32.0,63.0,38.0,59.0,54.0,63.0,28.0,36.0,63.0,40.0,66.0,56.0,63.0,28.0,26.0,68.0,0.0,0.0,0.0,0.0],[8.0,63.0,57.0,6.0,70.0,48.0,45.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[15.0,16.0,69.0,50.0,44.0,17.0,62.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    output_t = torch.tensor(output, dtype=torch.float32)

    solver = Solution()
    data = solver.get_dataset(positive, negative)
    print(f"data={data}")
    assert data is not None
    assert data.shape == output_t.shape
    assert torch.allclose(output_t, data), f"allclose is False, output={output_t}, data={data}"

test_get_dataset_2()
test_get_dataset_1()
