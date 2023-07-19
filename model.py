import torch
import torch.nn.functional as F

class BoWClassifier(torch.nn.Module):

    def __init__(self, vocab_size, num_labels):
        super(BoWClassifier, self).__init__()

        # Initialize simple linear map
        # UNKs are handled here: Size of BOW vector is one larger than vocabulary
        self.linear = torch.nn.Linear(vocab_size +1, num_labels)

    def forward(self, bow_vec):

        features = self.linear(bow_vec)

        return F.log_softmax(features, dim=1)