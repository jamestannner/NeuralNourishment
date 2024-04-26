# Data
BATCH_SIZE = 64 # Batch size we train on
MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
SEQ_LEN = 128  # Length of training sequences, in tokens. AKA the context size

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 2048  # Limits parameters in model.

# Training
EPOCHS = 3

# Inference
NUM_TOKENS_TO_GENERATE = 80

# Special tokens
START_OF_RECIPE = "<|recipe_start|>"
END_OF_RECIPE = "<|recipe_end|>"
PAD = "<|pad|>"
OOV = "<|oov|>"
SPECIAL_TOKENS = [PAD, START_OF_RECIPE, END_OF_RECIPE, OOV]

# File names
VOCAB_FILE = "vocab.pickle"