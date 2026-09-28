CREATE TABLE "LiteLLM_PublicInferenceId" (
    "public_id" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "fingerprint" TEXT NOT NULL,
    "value" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "LiteLLM_PublicInferenceId_pkey" PRIMARY KEY ("public_id")
);

CREATE UNIQUE INDEX "LiteLLM_PublicInferenceId_owner_kind_fingerprint_key"
ON "LiteLLM_PublicInferenceId"("owner", "kind", "fingerprint");

CREATE INDEX "LiteLLM_PublicInferenceId_expires_at_idx"
ON "LiteLLM_PublicInferenceId"("expires_at");
