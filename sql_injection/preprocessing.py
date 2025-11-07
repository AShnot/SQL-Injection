import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlglot import Tokenizer
from sqlglot.errors import ParseError

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
    """Lowercase + NFKC normalisation with null byte removal."""
    if text is None:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = cleaned.lower()
    return cleaned


_TOKENIZER = Tokenizer()


def _sqlglot_token_type(token) -> str:
    """Return a simplified token type string for sqlglot tokens."""
    token_type = getattr(token, "token_type", None)
    if token_type is None:
        return "unknown"
    # sqlglot exposes enums with ``name`` / ``value`` attributes.
    name = getattr(token_type, "name", None)
    if name:
        return str(name).lower()
    value = getattr(token_type, "value", None)
    if value is not None:
        return str(value).lower()
    return str(token_type).lower()


def lex_query(query: str) -> List[Token]:
    """Tokenise the SQL query using sqlglot for robust SQL-aware lexing."""
    if not query:
        return []

    raw_tokens: List[Token] = []
    try:
        parsed_tokens = _TOKENIZER.tokenize(query)
    except (ParseError, ValueError) as exc:  # pragma: no cover - defensive
        logger.warning("sqlglot failed to tokenise query, using fallback: %s", exc)
        return [Token(query, "unknown", 0, len(query))]
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected sqlglot failure, using fallback")
        return [Token(query, "unknown", 0, len(query))]

    offset = 0
    for tok in parsed_tokens:
        text = getattr(tok, "text", None) or ""
        if not text:
            continue
        # sqlglot tokens provide span info in newer versions; fall back to search
        span = getattr(tok, "span", None)
        if span and len(span) == 2 and all(isinstance(x, int) for x in span):
            start, end = span
        else:
            start = query.find(text, offset)
            if start == -1:
                start = offset
            end = start + len(text)
        offset = end
        token_type = _sqlglot_token_type(tok)
        raw_tokens.append(Token(text, token_type, int(start), int(end)))

    logger.debug("Lexed %d tokens via sqlglot", len(raw_tokens))
    return raw_tokens


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


