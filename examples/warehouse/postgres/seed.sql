-- Seed a tiny table and a SELECT-only role for the warehouse-ingestion demo.

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    snapshot_date TIMESTAMPTZ   NOT NULL,
    amount        NUMERIC(18, 4) NOT NULL,
    label         TEXT          NOT NULL
);

INSERT INTO events (id, snapshot_date, amount, label) VALUES
    (1, '2026-01-01T00:00:00Z', 12.3400, 'a'),
    (2, '2026-02-01T12:00:00Z', 99.9900, 'b'),
    (3, '2026-03-01T06:00:00Z', 41.5000, 'a')
ON CONFLICT (id) DO NOTHING;

-- Read-only role: the real read-only guarantee (the client also sends intent).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'demo_ro') THEN
        CREATE ROLE demo_ro LOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO demo_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO demo_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO demo_ro;
