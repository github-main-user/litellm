CREATE TABLE "LiteLLM_CanonicalModel" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "group" TEXT,
    "input_price_per_million_tokens" DECIMAL(24,12) NOT NULL,
    "output_price_per_million_tokens" DECIMAL(24,12) NOT NULL,
    "cache_read_price_per_million_tokens" DECIMAL(24,12) NOT NULL,
    "cache_write_price_per_million_tokens" DECIMAL(24,12) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT NOT NULL,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT NOT NULL,
    CONSTRAINT "LiteLLM_CanonicalModel_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "LiteLLM_CanonicalModel_name_key" ON "LiteLLM_CanonicalModel"("name");

CREATE TABLE "LiteLLM_CanonicalModelConnection" (
    "id" TEXT NOT NULL,
    "canonical_model_id" TEXT NOT NULL,
    "provider_connection" TEXT,
    "provider_model" TEXT NOT NULL,
    "mode" TEXT NOT NULL,
    "position" INTEGER NOT NULL DEFAULT 0,
    "deployment_id" TEXT NOT NULL,
    CONSTRAINT "LiteLLM_CanonicalModelConnection_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "LiteLLM_CanonicalModelConnection_deployment_id_key" ON "LiteLLM_CanonicalModelConnection"("deployment_id");
CREATE INDEX "LiteLLM_CanonicalModelConnection_canonical_model_id_idx" ON "LiteLLM_CanonicalModelConnection"("canonical_model_id");
CREATE INDEX "LiteLLM_CanonicalModelConnection_provider_connection_idx" ON "LiteLLM_CanonicalModelConnection"("provider_connection");
ALTER TABLE "LiteLLM_CanonicalModelConnection" ADD CONSTRAINT "LiteLLM_CanonicalModelConnection_canonical_model_id_fkey" FOREIGN KEY ("canonical_model_id") REFERENCES "LiteLLM_CanonicalModel"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_CanonicalModelConnection" ADD CONSTRAINT "LiteLLM_CanonicalModelConnection_provider_connection_fkey" FOREIGN KEY ("provider_connection") REFERENCES "LiteLLM_CredentialsTable"("credential_name") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "LiteLLM_CanonicalModelConnection" ADD CONSTRAINT "LiteLLM_CanonicalModelConnection_deployment_id_fkey" FOREIGN KEY ("deployment_id") REFERENCES "LiteLLM_ProxyModelTable"("model_id") ON DELETE RESTRICT ON UPDATE CASCADE;

CREATE FUNCTION litellm_lock_canonical_name() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended('litellm_canonical_name:' ||
        CASE WHEN TG_TABLE_NAME = 'LiteLLM_CanonicalModel' THEN to_jsonb(NEW)->>'name' ELSE to_jsonb(NEW)->>'model_name' END, 0));
    RETURN NEW;
END;
$$;

CREATE TRIGGER canonical_name_lock BEFORE INSERT OR UPDATE OF name ON "LiteLLM_CanonicalModel"
FOR EACH ROW EXECUTE FUNCTION litellm_lock_canonical_name();
CREATE TRIGGER deployment_name_lock BEFORE INSERT OR UPDATE OF model_name ON "LiteLLM_ProxyModelTable"
FOR EACH ROW EXECUTE FUNCTION litellm_lock_canonical_name();

CREATE FUNCTION litellm_check_canonical_name() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_TABLE_NAME = 'LiteLLM_CanonicalModel' THEN
        IF EXISTS (
            SELECT 1 FROM "LiteLLM_CanonicalModel" current_model
            JOIN "LiteLLM_ProxyModelTable" d ON d.model_name = current_model.name
            WHERE current_model.id = NEW.id AND current_model.name = to_jsonb(NEW)->>'name'
            AND NOT EXISTS (
                SELECT 1 FROM "LiteLLM_CanonicalModelConnection" c
                WHERE c.deployment_id = d.model_id AND c.canonical_model_id = current_model.id
            )
        ) THEN
            RAISE EXCEPTION 'Canonical name is already used by a legacy deployment' USING ERRCODE = '23505';
        END IF;
    ELSIF EXISTS (
        SELECT 1 FROM "LiteLLM_ProxyModelTable" current_deployment
        JOIN "LiteLLM_CanonicalModel" m ON m.name = current_deployment.model_name
        WHERE current_deployment.model_id = NEW.model_id
        AND current_deployment.model_name = to_jsonb(NEW)->>'model_name'
        AND NOT EXISTS (
            SELECT 1 FROM "LiteLLM_CanonicalModelConnection" c
            WHERE c.deployment_id = current_deployment.model_id AND c.canonical_model_id = m.id
        )
    ) THEN
        RAISE EXCEPTION 'Deployment name is reserved by a canonical model' USING ERRCODE = '23505';
    END IF;
    RETURN NULL;
END;
$$;

CREATE CONSTRAINT TRIGGER canonical_name_check AFTER INSERT OR UPDATE OF name ON "LiteLLM_CanonicalModel"
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION litellm_check_canonical_name();
CREATE CONSTRAINT TRIGGER deployment_name_check AFTER INSERT OR UPDATE OF model_name ON "LiteLLM_ProxyModelTable"
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION litellm_check_canonical_name();
