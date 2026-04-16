"""
Concept word list loading and filtering for U-F2-CBM.

The paper uses the 20K most common English words as concept set Z.
We source words from the google-10000-english list (extended to 20K via
supplementary sources).  A rigorous filtering procedure removes any term
that:
  1. Exactly matches a target class name (case-insensitive)
  2. Is a constituent word of any class name
  3. Is a WordNet synonym of any class name
  4. Is a WordNet hypernym or hyponym of any class name (up to depth 2)

Usage:
    words = load_concept_words(n=20000)
    filtered = filter_concepts(words, imagenet_class_names())
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import List, Set

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Word list sources
# ---------------------------------------------------------------------------

# Primary: google-10000-english (no-swears variant)
_GOOGLE_10K_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/google-10000-english-usa-no-swears.txt"
)

# Secondary: extend to 20K using the full english word list from the same repo
_GOOGLE_20K_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/20k.txt"
)

_CACHE_DIR = Path(__file__).parent / "_word_cache"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def load_concept_words(
    n: int = 20000,
    cache_dir: str | None = None,
) -> List[str]:
    """
    Return up to `n` common English words, deduplicated and lowercased.

    Downloads the google-10000-english word list on first call and caches it.
    If `n` > 10000, attempts to load the 20K list as well.
    """
    cache = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    words: List[str] = []

    # Load 20K list first (covers 10K as well)
    f20k = cache / "20k.txt"
    if not f20k.exists():
        try:
            _download(_GOOGLE_20K_URL, f20k)
        except Exception:
            f20k = None  # fall through to 10K

    if f20k and f20k.exists():
        with open(f20k) as fh:
            for line in fh:
                w = line.strip().lower()
                if w and w.isalpha():
                    words.append(w)
    else:
        f10k = cache / "10k.txt"
        if not f10k.exists():
            _download(_GOOGLE_10K_URL, f10k)
        with open(f10k) as fh:
            for line in fh:
                w = line.strip().lower()
                if w and w.isalpha():
                    words.append(w)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return unique[:n]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_concepts(
    words: List[str],
    class_names: List[str],
    use_wordnet: bool = True,
    wordnet_depth: int = 2,
) -> List[str]:
    """
    Remove from `words` any term that is:
      1. An exact match to a class name (after normalisation)
      2. A constituent token of a class name
      3. A WordNet synonym of any class name (if use_wordnet=True)
      4. A WordNet hypernym / hyponym of any class name up to `wordnet_depth`

    Args:
        words       : raw candidate concept list
        class_names : list of K target class names
        use_wordnet : whether to use WordNet for semantic filtering
        wordnet_depth: max depth for hypernym/hyponym traversal

    Returns:
        Filtered list of concept words.
    """
    banned: Set[str] = _build_banned_set(class_names, use_wordnet, wordnet_depth)

    filtered = [w for w in words if w.lower() not in banned]
    n_removed = len(words) - len(filtered)
    print(
        f"Concept filtering: {len(words)} → {len(filtered)} words "
        f"({n_removed} removed)"
    )
    return filtered


def _build_banned_set(
    class_names: List[str],
    use_wordnet: bool,
    wordnet_depth: int,
) -> Set[str]:
    """Build the complete set of banned terms from class names."""
    banned: Set[str] = set()

    # Normalise class names: lowercase, split on spaces / underscores / hyphens
    normalised_classes = [_normalise(c) for c in class_names]

    for norm in normalised_classes:
        tokens = _tokenise(norm)
        # Rule 1 & 2: exact matches and constituent words
        banned.add(norm)
        banned.update(tokens)

    # Rule 3 & 4: WordNet synonyms, hypernyms, hyponyms
    if use_wordnet:
        _ensure_wordnet()
        from nltk.corpus import wordnet as wn

        for class_name in tqdm(class_names, desc="WordNet filtering", leave=False):
            for token in _tokenise(_normalise(class_name)):
                if not token:
                    continue
                try:
                    synsets = wn.synsets(token)
                except Exception:
                    continue
                for syn in synsets:
                    # synonyms (lemmas in the same synset)
                    for lemma in syn.lemmas():
                        banned.add(lemma.name().lower().replace("_", " "))
                    # hypernyms up to depth
                    _collect_related(syn, banned, direction="hypernyms", depth=wordnet_depth)
                    # hyponyms up to depth
                    _collect_related(syn, banned, direction="hyponyms", depth=wordnet_depth)

    return banned


def _collect_related(synset, banned: Set[str], direction: str, depth: int) -> None:
    """Recursively collect hypernyms or hyponyms up to `depth` levels."""
    if depth == 0:
        return
    try:
        related = getattr(synset, direction)()
    except Exception:
        return
    for rel in related:
        for lemma in rel.lemmas():
            banned.add(lemma.name().lower().replace("_", " "))
        _collect_related(rel, banned, direction, depth - 1)


def _normalise(name: str) -> str:
    """Lowercase and replace non-alpha chars with spaces."""
    return re.sub(r"[^a-z ]", " ", name.lower()).strip()


def _tokenise(name: str) -> List[str]:
    """Split a normalised class name into individual tokens."""
    return [t for t in name.split() if t]


def _ensure_wordnet() -> None:
    """Download WordNet data if not already present."""
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")
    except LookupError:
        print("Downloading NLTK WordNet...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
