ALTER TABLE "LiteLLM_CanonicalModel"
    ADD COLUMN video_price_per_second DECIMAL(24,12),
    ALTER COLUMN input_price_per_million_tokens DROP NOT NULL,
    ALTER COLUMN output_price_per_million_tokens DROP NOT NULL,
    ALTER COLUMN cache_read_price_per_million_tokens DROP NOT NULL,
    ALTER COLUMN cache_write_price_per_million_tokens DROP NOT NULL,
    ADD CONSTRAINT canonical_pricing_unit CHECK (
        (video_price_per_second IS NULL AND num_nonnulls(input_price_per_million_tokens, output_price_per_million_tokens, cache_read_price_per_million_tokens, cache_write_price_per_million_tokens) = 4)
        OR (video_price_per_second IS NOT NULL AND video_price_per_second > 0 AND num_nonnulls(input_price_per_million_tokens, output_price_per_million_tokens, cache_read_price_per_million_tokens, cache_write_price_per_million_tokens) = 0)
    );

CREATE TABLE "LiteLLM_VideoGenerationJob" (
    id TEXT PRIMARY KEY,
    credential TEXT NOT NULL REFERENCES "LiteLLM_CredentialsTable"(credential_name) ON UPDATE CASCADE,
    provider_id TEXT,
    api_key TEXT NOT NULL,
    user_id TEXT,
    team_id TEXT,
    model_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    requested_seconds INTEGER NOT NULL CHECK (requested_seconds > 0),
    usd_per_second DECIMAL(24,12) NOT NULL CHECK (usd_per_second > 0),
    status TEXT NOT NULL DEFAULT 'submitting',
    output_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    next_poll TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (credential, provider_id)
);
CREATE INDEX video_generation_due ON "LiteLLM_VideoGenerationJob"(next_poll) WHERE status='in_progress';
