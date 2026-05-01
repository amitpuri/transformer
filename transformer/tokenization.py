import json
import collections
import re
from pathlib import Path
from . import trace
from typing import List, Dict, Tuple, Optional

# Constants for special tokens
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class Vocabulary:
    """
    Manages the mapping between string tokens and integer IDs.
    """
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def build(self, tokenized_sentences: List[List[str]]):
        counts = collections.Counter()
        for tokens in tokenized_sentences:
            counts.update(tokens)

        # Start with special tokens
        all_tokens = list(SPECIAL_TOKENS)
        for token, count in sorted(counts.items()):
            if count >= self.min_frequency:
                all_tokens.append(token)

        self.token_to_id = {token: i for i, token in enumerate(all_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    def __getitem__(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<unk>"])

    def to_token(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, "<unk>")

    @property
    def pad_id(self): return self.token_to_id["<pad>"]
    @property
    def unk_id(self): return self.token_to_id["<unk>"]
    @property
    def bos_id(self): return self.token_to_id["<bos>"]
    @property
    def eos_id(self): return self.token_to_id["<eos>"]


class SpacyTokenizer:
    """
    Uses spaCy for rule-based word-level tokenization.
    Useful for languages like English and German.
    """
    def __init__(self):
        import spacy
        self.nlp_en = None
        self.nlp_de = None
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_de = spacy.load("de_core_news_sm")
        except (OSError, ImportError):
            pass
        
        self.vocabs: Dict[str, Vocabulary] = {"en": None, "de": None}

    def tokenize(self, text: str, lang: str = "en") -> List[str]:
        trace.enter("Tokenizer.tokenize", f"lang='{lang}'")
        trace.log(f"Input text: [white]\"{text}\"[/white]")
        nlp = self.nlp_en if lang == "en" else self.nlp_de
        if nlp is None:
            # Fallback to simple whitespace/punctuation split if spacy models are missing
            import re
            trace.log("Using regex fallback (spaCy model not loaded)", style="yellow")
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        else:
            trace.log("Using spaCy rule-based tokenizer")
            tokens = [token.text for token in nlp.tokenizer(text)]
        trace.log(f"Tokens ({len(tokens)}): {tokens}")
        trace.exit(summary=f"{len(tokens)} tokens")
        return tokens


    def build_vocab(self, sentences: List[str], lang: str = "en", min_freq: int = 2):
        trace.enter("Tokenizer.build_vocab", f"lang='{lang}', min_freq={min_freq}, {len(sentences)} sentences")
        tokenized = [self.tokenize(s, lang) for s in sentences]
        vocab = Vocabulary(min_freq)
        vocab.build(tokenized)
        self.vocabs[lang] = vocab
        trace.log(f"Vocabulary size: {len(vocab)} (4 special + {len(vocab)-4} learned tokens)")
        trace.log(f"Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3")
        trace.exit(summary=f"vocab_size={len(vocab)}")
        return vocab

    def encode(self, text: str, lang: str = "en", add_special: bool = True) -> List[int]:
        vocab = self.vocabs[lang]
        if vocab is None:
            raise ValueError(f"Vocabulary for {lang} not built yet.")
        
        trace.enter("Tokenizer.encode", f"add_special={add_special}")
        tokens = self.tokenize(text, lang)
        ids = [vocab[t] for t in tokens]
        trace.log(f"Token->ID mapping: {list(zip(tokens, ids))}")
        
        if add_special:
            ids = [vocab.bos_id] + ids + [vocab.eos_id]
            trace.log(f"With special tokens: [<bos>={vocab.bos_id}] + ids + [<eos>={vocab.eos_id}]")
        trace.log(f"Final IDs: {ids}")
        trace.exit(summary=f"{len(ids)} token IDs")
        return ids

    def decode(self, ids: List[int], lang: str = "en", skip_special: bool = True) -> str:
        vocab = self.vocabs[lang]
        tokens = [vocab.to_token(i) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return " ".join(tokens)

class SubwordTokenizer:
    """
    Learns and applies Byte Pair Encoding (BPE).
    Useful for subword-level tokenization.
    """
    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocabulary: Optional[Vocabulary] = None
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _get_pair_counts(self, words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = collections.defaultdict(int)
        for symbols, freq in words.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_words = {}
        bigram = " ".join(pair)
        # Using a safer approach for merging instead of regex for clarity
        for symbols, freq in words.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                    new_symbols.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_words[tuple(new_symbols)] = freq
        return new_words

    def train(self, corpus: List[str]):
        # 1. Build initial char-level vocab
        word_freqs = collections.defaultdict(int)
        for sentence in corpus:
            for word in sentence.strip().split():
                word_freqs[word] += 1
        
        # Add end-of-word marker </w>
        vocab = {tuple(list(w)[:-1] + [list(w)[-1] + "</w>"]): freq for w, freq in word_freqs.items()}
        
        # 2. Iteratively merge
        all_tokens = set()
        for symbols in vocab:
            all_tokens.update(symbols)
            
        current_tokens = list(SPECIAL_TOKENS) + sorted(list(all_tokens))
        
        while len(current_tokens) < self.vocab_size:
            pairs = self._get_pair_counts(vocab)
            if not pairs: break
            
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_pair(best_pair, vocab)
            self.merges.append(best_pair)
            current_tokens.append(best_pair[0] + best_pair[1])

        self.token_to_id = {tok: i for i, tok in enumerate(current_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        if not self.token_to_id:
            raise ValueError("Tokenizer not trained.")
            
        unk_id = self.token_to_id.get("<unk>", 1)
        ids = []
        
        if add_special:
            ids.append(self.token_to_id.get("<bos>", 2))
            
        for word in text.strip().split():
            symbols = list(list(word)[:-1] + [list(word)[-1] + "</w>"])
            for pair in self.merges:
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                        new_symbols.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            
            for s in symbols:
                ids.append(self.token_to_id.get(s, unk_id))
                
        if add_special:
            ids.append(self.token_to_id.get("<eos>", 3))
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def save(self, path: str):
        state = {"vocab_size": self.vocab_size, "merges": self.merges, "token_to_id": self.token_to_id}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str) -> "SubwordTokenizer":
        state = json.loads(Path(path).read_text())
        tok = cls(vocab_size=state["vocab_size"])
        tok.merges = [tuple(m) for m in state["merges"]]
        tok.token_to_id = state["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        return tok

    @property
    def pad_id(self): return self.token_to_id.get("<pad>", 0)
    @property
    def bos_id(self): return self.token_to_id.get("<bos>", 2)
    @property
    def eos_id(self): return self.token_to_id.get("<eos>", 3)
    def __len__(self): return len(self.token_to_id)

