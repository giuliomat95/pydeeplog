import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.random.set_seed(42)


class DataPreprocess:
    def __init__(self, vocab: list):
        """
        Description:
        + Encodes log keys with additional values for padding, unknown keys and
        ending key
        + Randomly splits the dataset as usually: train (70%), validation (15$)
        and test (15$)
        + Splits into chunks of size `WINDOW_SIZE`. Note that, the longer the
        window size is, the more accurate the model is. However, workflows are
        less accurate.
        Attributes:
        :param vocab (list): List of unique keys in the dataset
        """
        self.special_tokens = ['[PAD]', '[UNK]', '[END]']
        # PAD is the token to pad the sequences such that they all have the same
        # length,
        # UNK replace unknown tokens, END labels the end of the sequences fed
        # into the encoder.
        self.vocab = vocab
        self.num_tokens = len(self.vocab) + len(self.special_tokens)
        # Build dictionaries of tokens
        self.dict_idx2token = self.special_tokens + vocab
        self.dict_token2idx = {value: key for key, value in
                               enumerate(self.dict_idx2token)}

        # PAD token must be in the position 0
        assert self.dict_token2idx['[PAD]'] == 0

    def get_num_tokens(self):
        """
        Returns number of log keys + the number of spacial tokens
        """
        return self.num_tokens

    def encode_dataset(self, dataset):
        """
        Encodes values of the dataset. The unknown tokens are replaced by the
        the corresponding index of the token 'UNK'.
        """
        for i, seq in enumerate(dataset):
            dataset[i] = [self.dict_token2idx[x] if x in self.dict_token2idx
                          else self.dict_token2idx['[UNK]'] for x in seq]
        return dataset

    def chunks(self, dataset, window_size):
        """
        Splits the dataset in smaller chunks with window_size as maximum length.
        """
        chunks = []
        for seq in dataset:
            chunks += self.chunks_seq(seq, window_size)
        return chunks

    @staticmethod
    def chunks_seq(seq, window_size):
        """
        Splits a sequence in smaller chunks with window_size as maximum length.
        Return a list of lists of size (len(seq)-window_size+1,window_size)
        """
        chunks = []
        if len(seq) > window_size:
            # If the sequence is longer than the window size, drag the window
            # and split as many sequences as possible
            i = 0
            while i + window_size <= len(seq):
                x = seq[i:(i + window_size)]
                chunks.append(x)
                i += 1
        else:
            chunks = [seq]
        return chunks

    def transform(self, dataset, add_padding=0):
        """
        Prepares the data to be consumed by the ML model. If used, it should
        come after chunks method.
        """
        # Split into input and target values
        X_data = []
        y_data = []
        for seq in dataset:
            X_data.append(seq[:-1] + [self.dict_token2idx['[END]']])
            y_data.append(seq[-1])

        # Add padding if necessary
        if add_padding > 0:
            X_data = pad_sequences(X_data, maxlen=add_padding,
                                   value=self.dict_token2idx['[PAD]'],
                                   padding='post')
            # Pads input sequences. The ones whose length is smaller than
            # 'maxlen' padded with 'value' until they reach all the same length
        # One hot encoding: Return vectors with all zeros but 1 in the in the
        # indices position passed in input.
        X_data = np.array(tf.one_hot(X_data, self.num_tokens))
        y_data = np.array(tf.one_hot(y_data, self.num_tokens))

        return X_data, y_data

    @staticmethod
    def split_idx(dataset_size, train_ratio, val_ratio):
        """
        Splits indices of the data into the usual three subsets: training,
        validation and testing.

        Arguments:
        - dataset_size
        - train_ratio (float): defines the subset for training.
        - val_ratio (float): defines the subset for validation (must be greater
        than train_ratio).

        Returns: three subsets of the input dataset.
        """
        train_idx, val_idx, test_idx = \
            np.split(np.arange(dataset_size),
                     [int(train_ratio * dataset_size),
                      int(val_ratio * dataset_size)])
        return train_idx, val_idx, test_idx

    def get_dictionaries(self):
        return self.dict_idx2token, self.dict_token2idx
