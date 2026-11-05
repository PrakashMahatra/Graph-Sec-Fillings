from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_engine():
    return engine


def init_db():
    from app import models  # noqa: F401  (register models on Base.metadata)
    # Extensions must exist before create_all; chunks.embedding needs the vector type and pgcrypto backs gen_random_uuid().
    _run_pg("CREATE EXTENSION IF NOT EXISTS vector")
    _run_pg("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    # app/models.py is the single source of truth for schema and index definitions.
    models.Base.metadata.create_all(engine)
    _ensure_row_level_security()


def _run_pg(stmt: str) -> None:
    # Run one DDL statement in its own transaction. PostgreSQL aborts the entire transaction on error, so isolating each statement means one failure can't poison the rest. All statements here are idempotent, so a skip is benign; genuine failures (e.g. missing permissions) are surfaced as a one-line warning instead of a silent `pass`.
    try:
        with engine.begin() as conn:
            conn.execute(text(stmt))
    except Exception as exc:
        print(f"[db] DDL skipped ({type(exc).__name__}): {stmt.splitlines()[0][:90]}")


def _ensure_row_level_security():
    # Enable row-level security and create tenant-isolation policies, idempotently, safe to run on every startup. `CREATE EXTENSION` and RLS policies are the only DDL here because they are the only DDL that SQLAlchemy has no declarative representation for -- tables, columns, computed columns and indexes all live in app/models.py and are created by create_all(). Mirrors db/init.sql (which the docker entrypoint runs on first volume init).
    rls_tables = ["filings", "documents", "sections", "chunks", "graph_nodes", "graph_edges"]
    with engine.begin() as conn:
        # PostgreSQL policies have no IF NOT EXISTS check, so preflight the catalog to avoid duplicate policy errors.
        policies = {
            row[0]
            for row in conn.execute(
                text("SELECT policyname FROM pg_policies WHERE schemaname = current_schema()")
            )
        }
    for t in rls_tables:
        _run_pg(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY")  # no-op if on
        if f"tenant_isolation_{t}" in policies:
            continue
        # Ingestion runs without tenant context, so this policy stays permissive until SET LOCAL is used.
        _run_pg(
            f"CREATE POLICY tenant_isolation_{t} ON {t} "
            f"USING (current_setting('app.tenant_id', true) = '' "
            f"OR tenant_id::text = current_setting('app.tenant_id', true))"
        )


@contextmanager
def session_scope(tenant_id: str | None = None):
    """Yields a session. Sets app.tenant_id (SET LOCAL) for RLS when given."""
    session = SessionLocal()
    try:
        if tenant_id:
            session.execute(text("SET LOCAL app.tenant_id = :tid"), {"tid": tenant_id})
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
