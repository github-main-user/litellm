ALTER TABLE "LiteLLM_VideoGenerationJob" ADD COLUMN "request_hash" TEXT;
ALTER TABLE "LiteLLM_VideoGenerationJob" ADD COLUMN "prepaid" BOOLEAN NOT NULL DEFAULT false;
