import html
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Token:
    text: str
    token_type: str
    start_char: int
    end_char: int


SQL_TOKEN_REGEX = re.compile(
    r"""
    (?P<comment_multi>/\*.*?\*/)|
    (?P<comment_single>--[^\n]*|#[^\n]*)|
    (?P<string>'(?:''|\\'|[^'])*'|"(?:""|\\"|[^"])*")|
    (?P<number>\b\d+(?:\.\d+)?\b)|
    (?P<operator><>|!=|<=|>=|==|:=|[-+*/%=<>&|^!~]+)|
    (?P<identifier>\b[a-zA-Z_][\w$]*\b)|
    (?P<punctuation>[(),.;])
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def normalise_text(text: str) -> str:
    """Apply light-weight normalisation keeping SQL semantics intact."""
    if text is None:
        return ""
    original = text
    text = text.replace("\x00", "")
    text = html.unescape(text)
    text = unquote(text)
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace but preserve single spaces
    text = re.sub(r"\s+", " ", text.strip())
    logger.debug("Normalised text from %r to %r", original[:50], text[:50])
    return text


def lex_query(query: str) -> List[Token]:
    """Tokenise the SQL query using a simple regex lexer."""
    tokens: List[Token] = []
    idx = 0
    length = len(query)
    while idx < length:
        match = SQL_TOKEN_REGEX.match(query, idx)
        if match:
            token_type = match.lastgroup or "unknown"
            token_text = match.group(token_type)
            start, end = match.span()
            tokens.append(Token(token_text, token_type, start, end))
            idx = end
        else:
            # treat the current character as whitespace/unknown token
            next_idx = idx + 1
            tokens.append(Token(query[idx:next_idx], "whitespace", idx, next_idx))
            idx = next_idx
    logger.debug("Lexed %d tokens", len(tokens))
    return tokens


def iter_windows(tokens: Sequence[Token], window_size: int, stride: int) -> Iterable[Tuple[int, int, Sequence[Token]]]:
    if not tokens:
        return
    total = len(tokens)
    i = 0
    while i < total:
        window_tokens = tokens[i : i + window_size]
        if not window_tokens:
            break
        yield i, min(total, i + window_size), window_tokens
        if i + window_size >= total:
            break
        i += stride


def tokens_to_text(tokens: Sequence[Token], source: str) -> str:
    if not tokens:
        return ""
    start = tokens[0].start_char
    end = tokens[-1].end_char
    return source[start:end]


def window_metadata(tokens: Sequence[Token], source: str) -> Dict[str, object]:
    if not tokens:
        return {"text": "", "start_char": 0, "end_char": 0}
    start = tokens[0].start_char
    end = tokens[-1].end_char
    return {
        "text": source[start:end],
        "start_char": int(start),
        "end_char": int(end),
        "token_count": len(tokens),
    }


def save_tokens(row_id: int, tokens: Sequence[Token]) -> None:
    config.TOKEN_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "text": tok.text,
            "token_type": tok.token_type,
            "start_char": tok.start_char,
            "end_char": tok.end_char,
        }
        for tok in tokens
    ]
    path = config.TOKEN_METADATA_DIR / config.TOKEN_META_TEMPLATE.format(row_id=row_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("Saved %d tokens metadata to %s", len(tokens), path)


