from collections.abc import Mapping
from typing import Final, TypeVar

from mcp.types import CallToolRequestParams, CallToolResult, TextContent
from pydantic import BaseModel, SecretStr

from litellm.experimental_mcp_client.client import MCPClient
from litellm.llms.kling.videos.protocol import Capabilities, Submission, TaskResult
from litellm.types.mcp import MCPAuth

T = TypeVar('T', bound=BaseModel)


def decode_result(result: CallToolResult, schema: type[T]) -> T:
    if result.is_error:
        raise ValueError("Kling rejected the tool call")
    if result.structured_content is not None:
        return schema.model_validate(result.structured_content)
    if len(result.content) == 1 and isinstance(result.content[0], TextContent):
        return schema.model_validate_json(result.content[0].text)
    raise ValueError("Kling returned an unexpected tool response")


class KlingVideoClient:
    def __init__(self, token: SecretStr):
        self._client: Final = MCPClient(
            server_url="https://kling.ai/mcp", auth_type=MCPAuth.bearer_token, auth_value=token.get_secret_value()
        )

    async def capabilities(self) -> Capabilities:
        result: Final = await self._client.call_tool(
            CallToolRequestParams(name="who_am_i", arguments={}), raise_on_error=True
        )
        return decode_result(result, Capabilities)

    async def submit(self, model: str, arguments: Mapping[str, str]) -> Submission:
        result: Final = await self._client.call_tool(
            CallToolRequestParams(
                name="text_to_video",
                arguments={
                    "model": model,
                    "arguments": [{"name": key, "value": value} for key, value in arguments.items()],
                    "inputs": [],
                },
            ),
            raise_on_error=True,
        )
        return decode_result(result, Submission)

    async def query(self, generation_id: str) -> TaskResult:
        result: Final = await self._client.call_tool(
            CallToolRequestParams(name="query_tasks", arguments={"generationId": generation_id}), raise_on_error=True
        )
        task: Final = decode_result(result, TaskResult)
        if task.generationId != generation_id:
            raise ValueError("Kling returned a different generation ID")
        return task
