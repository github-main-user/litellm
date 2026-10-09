import json
import re
from collections.abc import Iterator, Mapping
from itertools import accumulate
from typing import Final

from pydantic import JsonValue
from xxhash import xxh64

from .common_utils import _CLAUDE_CODE_BILLING_HEADER_PREFIX

# native Claude Code 2.1.283, inspected 2026-10-09: VA 0x3635785 and 0x363d980.
# sha256: 1859583ce32920595c61ef868bee52e1b1594f7486db209935e01f1e5e804ae2
_CCH_SEED: Final = 0x4D659218E32A3268
_ARRAY_TOKENS: Final = re.compile(rb'"(?:[^"\\]|\\[\s\S])*"|[][]')
_PROPERTIES: Final = re.compile(rb'"(?:model":"|fallbacks":\[|fallback_credit_token":"|max_tokens":[0-9]+)')
_CCH_FIELD: Final = re.compile(r"(?<=;)[ \t]*cch=[^;]*(?:;|$)")
_ENTRYPOINT: Final = re.compile(r"\bcc_entrypoint=[^;]*;")


def _array_depth(previous: tuple[int, int], current: tuple[int, int]) -> tuple[int, int]:
    return current[0], previous[1] + current[1]


def _fallback_end(body: bytes, start: int) -> int | None:
    depths: Final = accumulate(
        (
            (token.end(), 1 if body[token.start()] == 91 else -1 if body[token.start()] == 93 else 0)
            for token in _ARRAY_TOKENS.finditer(body, start + len(b'"fallbacks":['))
        ),
        _array_depth,
        initial=(start, 1),
    )
    return next((end for end, depth in depths if depth == 0), None)


def _property_span(body: bytes, start: int, end: int, scan_start: int) -> tuple[int, int]:
    if body[end:end + 1] == b",":
        return start, end + 1
    if start > scan_start and body[start - 1:start] == b",":
        return start - 1, end
    return start, end


def _cch_chunks(body: bytes) -> Iterator[memoryview]:
    view: Final = memoryview(body)
    cursor = 0  # rebind-ok: native byte scanner advances without copying the request
    for marker in _PROPERTIES.finditer(body):
        if marker.start() < cursor:
            continue
        if marker[0] == b'"model":"':
            end: Final = body.find(b'"', marker.end())
            if end >= 0:
                yield view[cursor:marker.end()]
                cursor = end  # rebind-ok: keep the closing quote in the next hash chunk
            continue
        property_end: Final = (
            _fallback_end(body, marker.start()) if marker[0] == b'"fallbacks":['
            else body.find(b'"', marker.end()) + 1 if marker[0] == b'"fallback_credit_token":"'
            else marker.end()
        )
        if property_end is None or property_end == 0:
            continue
        start, stop = _property_span(body, marker.start(), property_end, cursor)
        yield view[cursor:start]
        cursor = stop  # rebind-ok: advance over the excluded property
    yield view[cursor:]


def anthropic_subscription_cch(body: bytes) -> str:
    digest: Final = xxh64(seed=_CCH_SEED)
    for chunk in _cch_chunks(body):
        digest.update(chunk)
    return f"{digest.intdigest() & 0xFFFFF:05x}"


def _billing_placeholder(text: str) -> str:
    without_cch: Final = (
        _CLAUDE_CODE_BILLING_HEADER_PREFIX
        + _CCH_FIELD.sub("", ";" + text[len(_CLAUDE_CODE_BILLING_HEADER_PREFIX):])[1:]
    )
    entrypoint: Final = _ENTRYPOINT.search(without_cch)
    insertion: Final = entrypoint.end() if entrypoint is not None else len(_CLAUDE_CODE_BILLING_HEADER_PREFIX)
    if "cch=00000" in without_cch[:insertion]:
        raise ValueError("Subscription billing attribution has an ambiguous placeholder")
    return without_cch[:insertion] + " cch=00000;" + without_cch[insertion:]


def serialize_anthropic_subscription_request(request: Mapping[str, JsonValue]) -> str:
    system: Final = request.get("system")
    if not isinstance(system, list) or not system or not isinstance(system[0], dict):
        raise ValueError("Subscription billing attribution is missing")
    billing: Final = system[0]
    text: Final = billing.get("text")
    if billing.get("type") != "text" or not isinstance(text, str) or not text.startswith(_CLAUDE_CODE_BILLING_HEADER_PREFIX):
        raise ValueError("Subscription billing attribution is missing")
    billing_with_placeholder: Final = {
        "type": "text",
        "text": _billing_placeholder(text),
        **{name: value for name, value in billing.items() if name not in ("type", "text")},
    }
    wire: Final = json.dumps(
        {
            "system": [billing_with_placeholder, *system[1:]],
            **{name: value for name, value in request.items() if name != "system"},
        },
        separators=(",", ":"),
    ).encode()
    system_start: Final = wire.index(b'"system":[')
    placeholder: Final = wire.find(b"cch=00000", system_start, system_start + 300)
    if placeholder < 0:
        raise ValueError("Subscription billing attribution exceeds the signing window")
    value: Final = anthropic_subscription_cch(wire).encode()
    return (wire[:placeholder + 4] + value + wire[placeholder + 9:]).decode()
