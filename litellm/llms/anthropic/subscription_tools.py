import json
import re
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast

from litellm.llms.anthropic.common_utils import AnthropicError

ANTHROPIC_TOOL_NAME_REVERSE_MAP_KEY: Final = "_anthropic_tool_name_map"

_CLAUDE_CODE_NAMES: Final = MappingProxyType(
    {
        name.lower(): name
        for name in (
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Grep",
            "Glob",
            "AskUserQuestion",
            "EnterPlanMode",
            "ExitPlanMode",
            "KillShell",
            "NotebookEdit",
            "Skill",
            "Task",
            "TaskOutput",
            "TodoWrite",
            "WebFetch",
            "WebSearch",
        )
    }
)
_BLOCK_FIELDS: Final = MappingProxyType(
    {
        "tool_use": (("name",), ()),
        "tool_reference": (("name", "tool_name"), ()),
        "tool_result": ((), ("content",)),
        "tool_search_tool_result": ((), ("content",)),
        "tool_search_tool_search_result": ((), ("tool_references",)),
        "tool_addition": ((), ("tool",)),
        "tool_removal": ((), ("tool",)),
    }
)


def _record(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _items(value: object) -> tuple[object, ...]:
    return tuple(cast(Sequence[object], value)) if isinstance(value, (list, tuple)) else ()


def _block_fields(block: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    kind: Final = block.get("type")
    return _BLOCK_FIELDS.get(kind, ((), ())) if isinstance(kind, str) else ((), ())


def _block_names(value: object) -> Iterator[str]:
    if isinstance(value, (list, tuple)):
        for item in cast(Sequence[object], value):
            yield from _block_names(item)
        return
    block: Final = _record(value)
    name_fields, child_fields = _block_fields(block)
    for field in name_fields:
        if isinstance(name := block.get(field), str):
            yield name
    for field in child_fields:
        yield from _block_names(block.get(field))


def _rewrite_name(value: object, names: Mapping[str, str]) -> object:
    return names.get(value, value) if isinstance(value, str) else value


def _rewrite_block(value: object, names: Mapping[str, str]) -> object:
    if isinstance(value, list):
        return [_rewrite_block(item, names) for item in cast(Sequence[object], value)]
    block: Final = _record(value)
    name_fields, child_fields = _block_fields(block)
    if not name_fields and not child_fields:
        return value
    return {
        key: _rewrite_name(item, names)
        if key in name_fields
        else _rewrite_block(item, names)
        if key in child_fields
        else item
        for key, item in block.items()
    }


def prepare_subscription_tools(body: Mapping[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    tools: Final = tuple(_record(tool) for tool in _items(body.get("tools")))
    reserved: Final = frozenset(
        name for tool in tools if tool.get("type") not in (None, "custom") if isinstance(name := tool.get("name"), str)
    )
    declared: Final = tuple(name for tool in tools if isinstance(name := tool.get("name"), str))
    messages: Final = _items(body.get("messages"))
    history: Final = tuple(name for message in messages for name in _block_names(_record(message).get("content")))
    choice: Final = _record(body.get("tool_choice"))
    selected: Final = choice.get("name") if choice.get("type") == "tool" else None
    originals: Final = (*declared, *history, *((selected,) if isinstance(selected, str) else ()))
    targets: Final = {
        name: name if name in reserved else _CLAUDE_CODE_NAMES.get(name.lower(), name) for name in originals
    }
    if len(set(targets.values())) != len(targets):
        raise AnthropicError(
            status_code=400,
            message="Claude subscription tool names collide after normalization; use distinct names, not read and Read.",
        )
    forward: Final = {original: wire for original, wire in targets.items() if original != wire}
    reverse: Final = {wire: original for original, wire in forward.items()}
    if not forward:
        return dict(body), reverse
    return {
        **body,
        **(
            {
                "tools": [
                    {**tool, "name": _rewrite_name(tool["name"], forward)} if "name" in tool else tool for tool in tools
                ]
            }
            if "tools" in body
            else {}
        ),
        "messages": [
            {**_record(message), "content": _rewrite_block(_record(message).get("content"), forward)}
            if "content" in _record(message)
            else message
            for message in messages
        ],
        **({"tool_choice": {**choice, "name": _rewrite_name(selected, forward)}} if isinstance(selected, str) else {}),
    }, reverse


def tool_name_reverse_map(value: object) -> dict[str, str]:
    return {key: item for key, item in _record(value).items() if isinstance(item, str)}


def restore_subscription_tools(body: Mapping[str, object], names: Mapping[str, str]) -> dict[str, object]:
    if not names:
        return dict(body)
    return {
        **body,
        **({"content": _rewrite_block(body["content"], names)} if "content" in body else {}),
        **({"content_block": _rewrite_block(body["content_block"], names)} if "content_block" in body else {}),
        **(
            {"message": restore_subscription_tools(_record(body["message"]), names)}
            if body.get("type") == "message_start" and isinstance(body.get("message"), Mapping)
            else {}
        ),
    }


def _restore_event_data(data: str, names: Mapping[str, str]) -> str:
    try:
        payload: Final = cast(object, json.loads(data))
    except ValueError:
        return data
    if not isinstance(payload, dict):
        return data
    restored: Final = restore_subscription_tools(cast(Mapping[str, object], payload), names)
    return json.dumps(restored) if restored != payload else data


_SSE_LINE_END: Final = re.compile(rb"\r\n|[\r\n]")


def _sse_frame_ends(data: bytes) -> Iterator[int]:
    line_start = 0  # rebind-ok: tracks the start of each line while scanning one buffered event
    for match in _SSE_LINE_END.finditer(data):
        if match.start() == line_start:
            yield match.end()
        line_start = match.end()


def _split_sse_frames(data: bytes) -> tuple[tuple[bytes, ...], bytes]:
    ends: Final = tuple(_sse_frame_ends(data))
    if not ends:
        return (), data
    starts: Final = (0, *ends[:-1])
    return tuple(data[start:end] for start, end in zip(starts, ends)), data[ends[-1] :]


def _sse_lines(frame: bytes) -> Iterator[tuple[int, int, int]]:
    line_start = 0  # rebind-ok: tracks byte spans without decoding or changing line endings
    for match in _SSE_LINE_END.finditer(frame):
        yield line_start, match.start(), match.end()
        line_start = match.end()
    if line_start < len(frame):
        yield line_start, len(frame), len(frame)


def _data_value_start(line: bytes) -> int | None:
    if line == b"data":
        return len(line)
    if not line.startswith(b"data:"):
        return None
    value_start: Final = len(b"data:")
    return value_start + 1 if line[value_start:].startswith(b" ") else value_start


def _restore_sse_frame(frame: bytes, names: Mapping[str, str]) -> bytes:
    lines: Final = tuple(_sse_lines(frame))
    data_lines: Final = tuple(
        (start, content_end, line_end, value_start)
        for start, content_end, line_end in lines
        if (value_start := _data_value_start(frame[start:content_end])) is not None
    )
    if not data_lines:
        return frame
    try:
        data: Final = b"\n".join(
            frame[start + value_start : content_end]
            for start, content_end, _line_end, value_start in data_lines
        ).decode("utf-8")
    except UnicodeDecodeError:
        return frame
    rewritten: Final = _restore_event_data(data, names)
    if rewritten == data:
        return frame
    first_data_start, first_content_end, first_line_end, first_value_start = data_lines[0]
    data_starts: Final = frozenset(start for start, _content_end, _line_end, _value_start in data_lines)
    return b"".join(
        (
            frame[start : start + first_value_start]
            + rewritten.encode()
            + frame[first_content_end:first_line_end]
            if start == first_data_start
            else b""
            if start in data_starts
            else frame[start:line_end]
        )
        for start, _content_end, line_end in lines
    )


async def restore_subscription_tool_stream(
    stream: AsyncIterator[bytes], names: Mapping[str, str]
) -> AsyncIterator[bytes]:
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import aclose_if_supported

    pending = b""  # rebind-ok: retains at most one unterminated SSE event across transport chunks
    try:
        async for chunk in stream:
            if not names:
                yield chunk
                continue
            frames, pending = _split_sse_frames(pending + chunk)
            for frame in frames:
                yield _restore_sse_frame(frame, names)
        if pending:
            yield pending
    finally:
        await aclose_if_supported(stream)
