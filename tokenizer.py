"""
Defines a Tokenizer class and some helper functions. This 
class can be used to train, encode, and decode a byte-pair 
encoding tokenizer. It also supports special tokens, which
can be helpful in dividing different parts of our training 
courpus (e.g., recipe start/end, ingredients start/end, etc.).
Draws heavily from Andrej Karpathy's minBPE project.
"""
import regex as re

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# Matches contractions (e.g., 's, 'd, 'll, 've, 're), sequences of 
# letters (words), individual digits, non-alphanumeric 
# characters (excluding whitespace), line breaks, and trailing whitespace.

def get_stats(ids: list[int], counts: dict[tuple[int, int], int] = None) -> dict[tuple[int, int], int]:
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class Tokenizer:
    def __init__(self):
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.inverse_special_tokens = {} # int -> str, e.g. {100257: '<|endoftext|>'}

        self.pattern = SPLIT_PATTERN # regex pattern for tokenization
        self.compiled_pattern = re.compile(self.pattern) # compiled regex pattern

        self.merges = {} # (int, int) -> int
        self.vocab = self._build_vocab() # int -> bytes
    

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)
    

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256: raise ValueError("Vocab size must be at least 256")

        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text) # divide based on regex pattern
        ids = [list(text_chunk.encode("utf-8")) for text_chunk in text_chunks] # convert each chunk into it's byte representation

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats) # counts the number of each consecutive pair's occurence
            if not stats: break # no more merges possible

            pair = max(stats, key=stats.get) # find the pair with the highest count

            idx = 256 + i # mint a new token and assign it to the next available id

            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids] # replace all occurences of `pair` in `ids` with the newly minted token
            
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()


    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


    def _encode_chunk(self, text: str) -> list[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges: # edge case where no more merges exist
                break # nothing else can be merged anymore

            # otherwise merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        else: 
            special_pattern = "^$" # match nothing

        # splits special tokens from the normal text
        # special_chunks = [match.group() for match in re.finditer(special_pattern, text)]
        special_chunks = re.split(special_pattern, text)
        special_chunks = [special_chunk for special_chunk in special_chunks if special_chunk]

        ids = []
        for chunk in special_chunks:
            if chunk in self.special_tokens: # found a special token
                ids.append(self.special_tokens[chunk]) # add the special token's encoding
            else: # ordinary text
                chunks = re.findall(self.compiled_pattern, chunk) # divide normal text into tokens
                for text_chunk in chunks:
                    ids.extend(self._encode_chunk(text_chunk)) # encode each token
        return ids


    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab: 
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens: 
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else: 
                raise ValueError(f"Unknown token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab


if __name__ == "__main__":
    text = """Once upon a time, in a faraway land nestled between towering mountains and vast oceans, there existed a quaint little village known as Lumindor. It was a place where magic danced through the air like a symphony of light, where every corner held a tale waiting to be discovered, and where the extraordinary was simply a part of everyday life.  In Lumindor, the sky was forever painted with hues of lavender and rose, as if the heavens themselves were in a perpetual state of bliss. The streets were lined with cobblestones worn smooth by the footsteps of generations, and the houses were adorned with vibrant flowers that seemed to bloom year-round. It was a place of wonder and enchantment, where dreams took flight and possibilities knew no bounds.  At the heart of Lumindor stood a majestic castle, its spires reaching toward the heavens like fingers stretching for the stars. It was home to the royal family of Lumindor, rulers beloved by their people for generations untold. King Aldric and Queen Elara ruled with wisdom and grace, their kindness known throughout the land.  But Lumindor was not without its challenges. For beyond the borders of the village lay the Dark Forest, a place shrouded in mystery and shadow. Legends spoke of dark creatures lurking within its depths, waiting to ensnare the unwary in their grasp. Despite the danger, brave adventurers would often venture into the forest in search of treasure or glory, their tales becoming the stuff of legend.  One such adventurer was a young woman named Aria. With hair as dark as the midnight sky and eyes that sparkled like the stars themselves, she possessed a spirit as fiery as the sun. From a young age, Aria had longed to explore the world beyond Lumindor, to uncover its secrets and unravel its mysteries.  Armed with nothing but her wit and her courage, Aria set out into the Dark Forest, determined to carve her own destiny. Along the way, she encountered all manner of creatures, from mischievous fairies to fearsome dragons. Each encounter tested her resolve, but she pressed onward, her determination unwavering.  As she delved deeper into the forest, Aria stumbled upon a hidden grove, bathed in a soft, ethereal light. In the center of the grove stood a magnificent tree, its branches reaching toward the sky like outstretched arms. Intrigued, Aria approached the tree, her heart pounding with anticipation.  As she drew closer, she heard a soft, melodic voice whispering on the wind. It spoke of ancient secrets and forgotten wisdom, of a power that lay dormant within the heart of the forest. Drawn by the voice, Aria reached out and placed her hand upon the tree, feeling a surge of energy course through her veins.  In that moment, she understood. The magic of Lumindor was not confined to the village alone; it flowed through the very land itself, connecting all living things in a delicate web of light and shadow. And Aria, with her courage and her spirit, was destined to be its guardian.  With newfound purpose, Aria returned to Lumindor, her heart brimming with hope and determination. She shared her discovery with King Aldric and Queen Elara, who listened with rapt attention. Together, they vowed to protect the magic of Lumindor, to ensure that its light would shine for generations to come.  And so, the tale of Aria and the magic of Lumindor became legend, passed down through the ages as a testament to the power of courage, friendship, and the enduring magic of the world around us. And though many years have passed since that fateful day, the spirit of adventure lives on in the hearts of all who call Lumindor home."""
    
    T = Tokenizer()

    MAX_VOCAB_SIZE = 2048
    special_tokens = {
        "<|startrecipe|>": MAX_VOCAB_SIZE - 1,
        "<|endrecipe|>": MAX_VOCAB_SIZE - 2,
    }
    
    T.train(text, vocab_size=MAX_VOCAB_SIZE - len(special_tokens))
    T.register_special_tokens(special_tokens)

    example_new_text = "This is new text that the model has never seen before!!"
    print(f"Original Text: {example_new_text}\n")
    encoded = T.encode(example_new_text)
    print(f"Encoded: {encoded}\n")
    decoded = T.decode(encoded)
    print(f"Decoded: {decoded}\n")
    print(f"Original == Decoded: {example_new_text == decoded}")
    print(f"Vocab Size: {T.vocab_size}")
    print(f"Vocab Size == MAX_VOCAB_SIZE: {T.vocab_size == MAX_VOCAB_SIZE}")
    print("\nToken by token:")
    for e in encoded:
        print(f"{e:3d}\t->\t`{T.decode([e])}`")
