import html
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

import sqlparse
from sqlparse.sql import Token as SQLParseToken
from sqlparse.tokens import Token as SQLParseTokenType

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Token:
    text: str
    token_type: str
    start_char: int
    end_char: int


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


def _sqlparse_token_type(token: SQLParseToken) -> str:
    """Return a simplified token type string for sqlparse tokens."""
    ttype = token.ttype
    if ttype is None:
        if token.is_group:
            return token.__class__.__name__.lower()
        return "unknown"
    if isinstance(ttype, SQLParseTokenType):
        # sqlparse TokenType has a string-like repr ``Token.Keyword`` etc.
        return str(ttype).split(".")[-1].lower()
    # fallback for e.g. combined token types (tuples)
    return ":".join(str(part).split(".")[-1].lower() for part in (ttype if isinstance(ttype, tuple) else (ttype,)))


def lex_query(query: str) -> List[Token]:
    """Tokenise the SQL query using sqlparse for robust SQL-aware lexing."""
    tokens: List[Token] = []
    try:
        parsed = sqlparse.parse(query)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("sqlparse failed, falling back to character tokens: %s", exc)
        return [Token(query, "unknown", 0, len(query))] if query else []

    if not parsed:
        return []

    offset = 0
    for tok in parsed[0].flatten():
        text = str(tok)
        if not text:
            continue
        start = offset
        end = start + len(text)
        offset = end
        token_type = _sqlparse_token_type(tok)
        tokens.append(Token(text, token_type, start, end))
    logger.debug("Lexed %d tokens via sqlparse", len(tokens))
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


