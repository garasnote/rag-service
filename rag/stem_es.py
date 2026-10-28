"""Spanish Snowball stemmer in pure Python, and the guard that keeps it off names.

The algorithm is the published Spanish Snowball stemmer
(https://snowballstem.org/algorithms/spanish/), ported from ``algorithms/spanish.sbl``
so this module needs nothing but the standard library. The port is checked
against the reference ``snowballstemmer`` package in the tests (skipped when that
package is absent), because a stemmer that is "roughly right" is worse than none.

Two things matter for a job-posting index:

``stem_es(word)``
    the Snowball stem for one lowercase Spanish-ish word. Accented and
    unaccented spellings both work, because the postlude step unaccents the
    result either way.

``stemmable(token)``
    the entity guard. A stemmer that shreds ``Datavant`` into ``Datav`` is the
    documented reason a previous naive suffix stripper lost 25 points of MRR, so
    the guard refuses tokens an index should keep verbatim. Two policies:

    ``acronyms`` (default)  refuse digits, punctuation, all-caps and the employer
                            allowlist. Measured to match leaving the guard off
                            entirely on both benchmarks.
    ``strict``              additionally refuse anything capitalised. It protects
                            every name, and it measured 15 points of intent sat@1
                            worse, because postings are title-cased lists: the
                            query word ``analista`` stems to ``anal`` while the
                            document's ``Analista`` stays ``analista`` and the two
                            no longer meet.

    The guard is not cosmetic: 247 of the 571 employer names in the corpus change
    under the raw stemmer (``Accenture`` -> ``accentur``). Matching stays symmetric
    so that costs nothing on its own, but a name that collides with a common stem
    would, and the all-caps and allowlist rules keep the obvious cases out for free.
"""
from __future__ import annotations

VOWELS = frozenset("aeiou\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc")

_UNACCENT = {"\u00e1": "a", "\u00e9": "e", "\u00ed": "i", "\u00f3": "o", "\u00fa": "u"}

# "acronyms" won every measurement; "strict" is the rule as first specified and is
# still here so the ablation is a one-word change.
GUARD_POLICY = "acronyms"

# Employers and product names that survive lowercasing in real text (a tracker
# row, a slug, a search box) and that a Spanish stemmer has no business touching.
ENTITY_WORDS = frozenset("""
datavant keepler buynomics cofidis metyis cinte kpmg feverup idom quental arelance
gobernacion accenture deloitte indra repsol iberdrola telefonica santander mckinsey
glovo cabify amadeus everis sopra babel celonis datadog snowflake databricks
""".split())


# --- regions -----------------------------------------------------------------

def _is_vowel(ch: str) -> bool:
    return ch in VOWELS


def region_rv(word: str) -> int:
    """RV: after the next vowel (2nd letter consonant), after the next consonant
    (first two vowels), otherwise after the third letter."""
    n = len(word)
    if n < 2:
        return n
    if _is_vowel(word[1]):
        if _is_vowel(word[0]):
            for i in range(2, n):
                if not _is_vowel(word[i]):
                    return i + 1
            return n
        return min(3, n)
    for i in range(2, n):
        if _is_vowel(word[i]):
            return i + 1
    return n


def _after_vowel_consonant(word: str, start: int) -> int:
    for i in range(start + 1, len(word)):
        if _is_vowel(word[i - 1]) and not _is_vowel(word[i]):
            return i + 1
    return len(word)


def region_r1(word: str) -> int:
    return _after_vowel_consonant(word, 0) if word else 0


def region_r2(word: str, p1: int) -> int:
    return _after_vowel_consonant(word, p1)


def _longest(word: str, suffixes, start_min: int = 0) -> str:
    """Longest suffix match. ``start_min`` is Snowball's ``setlimit``: suffixes
    that would start before it are not candidates at all, so a shorter one wins."""
    best = ""
    for suffix in suffixes:
        if len(suffix) > len(best) and word.endswith(suffix) and len(word) - len(suffix) >= start_min:
            best = suffix
    return best


# --- step 0: attached pronoun -------------------------------------------------

_PRONOUNS = ("me", "se", "sela", "selo", "selas", "selos", "la", "le", "lo",
             "las", "les", "los", "nos")
_PRONOUN_ACCENTED = (("i\u00e9ndo", "iendo"), ("\u00e1ndo", "ando"),
                     ("\u00e1r", "ar"), ("\u00e9r", "er"), ("\u00edr", "ir"))
_PRONOUN_PLAIN = ("ando", "iendo", "ar", "er", "ir")


def _attached_pronoun(word: str, pv: int) -> str:
    suffix = _longest(word, _PRONOUNS)
    if not suffix:
        return word
    stem = word[:-len(suffix)]
    for accented, plain in _PRONOUN_ACCENTED:
        if stem.endswith(accented) and len(stem) - len(accented) >= pv:
            return stem[:-len(accented)] + plain
    for plain in _PRONOUN_PLAIN:
        if stem.endswith(plain) and len(stem) - len(plain) >= pv:
            return stem[:-len(plain)]
    if stem.endswith("uyendo") and len(stem) - 6 >= pv:
        return stem[:-6] + "yendo"
    return word


# --- step 1: standard suffix --------------------------------------------------

_G1 = ("anza", "anzas", "ico", "ica", "icos", "icas", "ismo", "ismos", "able",
       "ables", "ible", "ibles", "ista", "istas", "oso", "osa", "osos", "osas",
       "amiento", "amientos", "imiento", "imientos")
_G2 = ("adora", "ador", "aci\u00f3n", "adoras", "adores", "aciones", "ante",
       "antes", "ancia", "ancias", "acion")
_G3 = ("log\u00eda", "log\u00edas")
_G4 = ("uci\u00f3n", "uciones", "ucion")
_G5 = ("encia", "encias")
_G6 = ("amente",)
_G7 = ("mente",)
_G8 = ("idad", "idades")
_G9 = ("iva", "ivo", "ivas", "ivos")

_STANDARD = (_G1, _G2, _G3, _G4, _G5, _G6, _G7, _G8, _G9)


def _standard_suffix(word: str, p1: int, p2: int):
    """Return the re-stemmed word, or None when the leaf condition fails.

    None is not "unchanged": a failed region test makes Snowball try the next
    step, so the caller must be able to tell the two apart.
    """
    best = ""
    for group in _STANDARD:
        candidate = _longest(word, group)
        if len(candidate) > len(best):
            best = candidate
    if not best:
        return None
    start = len(word) - len(best)

    if best in _G1 or best in _G2:
        if start < p2:
            return None
        word = word[:start]
        if best in _G2 and word.endswith("ic") and len(word) - 2 >= p2:
            word = word[:-2]
        return word
    if best in _G3:
        return word[:start] + "log" if start >= p2 else None
    if best in _G4:
        return word[:start] + "u" if start >= p2 else None
    if best in _G5:
        return word[:start] + "ente" if start >= p2 else None
    if best in _G6:                                    # -amente
        if start < p1:
            return None
        word = word[:start]
        for suffix in ("iv", "os", "ic", "ad"):
            if word.endswith(suffix) and len(word) - len(suffix) >= p2:
                word = word[:len(word) - len(suffix)]
                if suffix == "iv" and word.endswith("at") and len(word) - 2 >= p2:
                    word = word[:-2]
                break
        return word
    if best in _G7:                                    # -mente
        if start < p2:
            return None
        word = word[:start]
        for suffix in ("ante", "able", "ible"):
            if word.endswith(suffix) and len(word) - len(suffix) >= p2:
                word = word[:len(word) - len(suffix)]
                break
        return word
    if best in _G8:                                    # -idad
        if start < p2:
            return None
        word = word[:start]
        for suffix in ("abil", "ic", "iv"):
            if word.endswith(suffix) and len(word) - len(suffix) >= p2:
                word = word[:len(word) - len(suffix)]
                break
        return word
    if start < p2:                                     # _G9: -iva -ivo
        return None
    word = word[:start]
    if word.endswith("at") and len(word) - 2 >= p2:
        word = word[:-2]
    return word


# --- step 2a: verb suffixes ---------------------------------------------------

_Y_FORMS = ("ya", "ye", "yan", "yen", "yeron", "yendo", "yo", "y\u00f3",
            "yas", "yes", "yais", "yamos")


def _y_verb_suffix(word: str, pv: int):
    suffix = _longest(word, _Y_FORMS, start_min=pv)
    if not suffix:
        return None
    start = len(word) - len(suffix)
    if start == 0 or word[start - 1] != "u":
        return None
    return word[:start]


_VERB_GU = ("en", "es", "\u00e9is", "emos")
_VERB_AR = ("ar\u00edan", "ar\u00edas", "ar\u00e1n", "ar\u00e1s", "ar\u00edais",
            "ar\u00eda", "ar\u00e9is", "ar\u00edamos", "aremos", "ar\u00e1",
            "ar\u00e9")
_VERB_ER = ("er\u00edan", "er\u00edas", "er\u00e1n", "er\u00e1s", "er\u00edais",
            "er\u00eda", "er\u00e9is", "er\u00edamos", "eremos", "er\u00e1",
            "er\u00e9")
_VERB_IR = ("ir\u00edan", "ir\u00edas", "ir\u00e1n", "ir\u00e1s", "ir\u00edais",
            "ir\u00eda", "ir\u00e9is", "ir\u00edamos", "iremos", "ir\u00e1",
            "ir\u00e9")
_VERB_OTHER = ("aba", "ada", "ida", "\u00eda", "ara", "iera", "ad", "ed", "id",
               "ase", "iese", "aste", "iste", "an", "aban", "\u00edan", "aran",
               "ieran", "asen", "iesen", "aron", "ieron", "ado", "ido", "ando",
               "iendo", "i\u00f3", "ar", "er", "ir", "as", "abas", "adas",
               "idas", "\u00edas", "aras", "ieras", "ases", "ieses", "\u00eds",
               "\u00e1is", "abais", "\u00edais", "arais", "ierais", "aseis",
               "ieseis", "asteis", "isteis", "ados", "idos", "amos",
               "\u00e1bamos", "\u00edamos", "imos", "\u00e1ramos",
               "i\u00e9ramos", "i\u00e9semos", "\u00e1semos")

_VERB_ALL = _VERB_GU + _VERB_AR + _VERB_ER + _VERB_IR + _VERB_OTHER


def _verb_suffix(word: str, pv: int):
    best = _longest(word, _VERB_ALL, start_min=pv)
    if not best:
        return None
    start = len(word) - len(best)
    if best in _VERB_GU and start >= 2 and word[start - 1] == "u" and word[start - 2] == "g":
        start -= 1                                     # siguen -> sig
    return word[:start]


# --- step 2b: residual suffix -------------------------------------------------

def _residual_suffix(word: str, pv: int):
    if word.endswith("os"):
        start = len(word) - 2
        return word[:start] if start >= pv else None
    for suffix in ("a", "o", "\u00e1", "\u00ed", "\u00f3", "e", "\u00e9"):
        if not word.endswith(suffix):
            continue
        start = len(word) - 1
        if start < pv:
            return None
        word = word[:start]
        if suffix in ("e", "\u00e9") and len(word) >= 2 and word[-1] == "u" and word[-2] == "g":
            if len(word) - 1 >= pv:
                word = word[:-1]                       # llegue -> lleg
        return word
    return None


# --- the public stemmer -------------------------------------------------------

def stem_es(word: str) -> str:
    """Stem one word. Regions are marked once, on the original word, as Snowball does."""
    word = (word or "").lower()
    if not word:
        return ""
    pv = region_rv(word)
    p1 = region_r1(word)
    p2 = region_r2(word, p1)

    word = _attached_pronoun(word, pv)
    stemmed = _standard_suffix(word, p1, p2)
    if stemmed is None:
        stemmed = _y_verb_suffix(word, pv)
    if stemmed is None:
        stemmed = _verb_suffix(word, pv)
    if stemmed is not None:
        word = stemmed
    residual = _residual_suffix(word, pv)
    if residual is not None:
        word = residual
    return "".join(_UNACCENT.get(ch, ch) for ch in word)


# --- the entity guard ---------------------------------------------------------

def stemmable(token: str, sentence_initial: bool = False, policy: str = "") -> bool:
    """True when a token is safe to hand to the Spanish stemmer.

    False for anything that looks like a name, an acronym or a version string, and
    for the employer allowlist. ``sentence_initial`` forgives the one capital that
    is spelling rather than identity, and only matters under ``policy="strict"``.
    """
    if not token or not token.isalpha():
        return False                                   # digits, +, #, -, . inside
    if token.lower() in ENTITY_WORDS:
        return False
    if len(token) > 1 and token.isupper():
        return False                                   # NTT, BBVA, RAG
    if (policy or GUARD_POLICY) == "strict" and token != token.lower():
        return bool(sentence_initial)                  # Datavant, Buynomics
    return True
