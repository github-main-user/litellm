BEGIN;

LOCK TABLE "LiteLLM_CanonicalModel" IN ACCESS EXCLUSIVE MODE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM "LiteLLM_CanonicalModel"
        WHERE "base_model" IS NULL
           OR length("base_model") > 255
           OR "base_model" !~ '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$'
    ) THEN
        RAISE EXCEPTION 'Cannot require base_model: existing canonical models have null, empty or invalid base_model values. Assign valid provider/model IDs explicitly before retrying; no IDs are backfilled.';
    END IF;
END $$;

ALTER TABLE "LiteLLM_CanonicalModel" ALTER COLUMN "base_model" SET NOT NULL;
ALTER TABLE "LiteLLM_CanonicalModel" ADD CONSTRAINT "LiteLLM_CanonicalModel_base_model_valid"
    CHECK (length("base_model") <= 255 AND "base_model" ~ '^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$');

COMMIT;
