from decimal import Decimal

import pytest

from litellm.llms.kling.videos.protocol import TaskResult, VideoQuote


@pytest.mark.parametrize(
    ("status", "work_status", "content_type", "expected"),
    [
        ("PROCESSING", None, "video", None),
        ("FAILED", "completed", "video", Decimal(0)),
        ("cancelled", None, "video", Decimal(0)),
        ("PARTIAL_COMPLETED", "FAILED", "video", None),
        ("COMPLETED", "completed", "image", None),
        ("COMPLETED", "completed", "video", Decimal("0.615")),
        ("PARTIAL_COMPLETED", "completed", "video", Decimal("0.615")),
        ("unknown", "completed", "video", None),
    ],
)
def test_bill_only_confirmed_video_at_original_rate(status, work_status, content_type, expected):
    quote = VideoQuote(requested_seconds=5, usd_per_second=Decimal("0.123"))
    task = TaskResult.model_validate(
        {
            "generationId": "job",
            "status": status,
            "createTime": 1000,
            "finishTime": 93000,
            "creditsConsumed": 400,
            "works": [{"status": work_status, "contentType": content_type, "url": "https://example.com/output"}],
        }
    )
    assert quote.final_cost(task) == expected
