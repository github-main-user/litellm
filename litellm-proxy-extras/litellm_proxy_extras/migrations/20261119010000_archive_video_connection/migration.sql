ALTER TABLE "LiteLLM_VideoGenerationJob"
    ALTER COLUMN credential DROP NOT NULL,
    ADD COLUMN archived_credential TEXT,
    ADD CONSTRAINT video_job_detached_only_if_terminal CHECK (
        credential IS NOT NULL OR (archived_credential IS NOT NULL AND status IN ('completed','failed'))
    );
CREATE INDEX video_job_connection_idx ON "LiteLLM_VideoGenerationJob"(credential);
