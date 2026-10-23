"""
Tenant scope MUST come from authenticated server-side context, never from a
client-supplied field. This module is the single choke point: every
retrieval/query function below takes a `tenant_id` resolved here, and every
DB query filters on it explicitly (no reliance on a global session variable
that a bug could leave unset).

For the demo, auth is a stand-in API-key -> tenant lookup. Swap
`resolve_tenant` for real auth (JWT/OAuth) without touching callers.
"""
from fastapi import HTTPException
from sqlalchemy import select

from app.db import session_scope
from app.models import Tenant


class TenantContext:
    def __init__(self, tenant_id: str, slug: str):
        self.tenant_id = tenant_id
        self.slug = slug


def get_or_create_tenant(slug: str, name: str | None = None) -> str:
    with session_scope() as db:
        row = db.execute(select(Tenant).where(Tenant.slug == slug)).scalar_one_or_none()
        if row:
            return row.id
        t = Tenant(slug=slug, name=name or slug)
        db.add(t)
        db.flush()
        return t.id


def resolve_tenant_by_slug(slug: str) -> TenantContext:
    """Server-side resolution. `slug` here stands in for a verified API key
    or session claim -- in production this never comes from a client-typed
    'tenant_id' query param."""
    with session_scope() as db:
        row = db.execute(select(Tenant).where(Tenant.slug == slug)).scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail=f"unknown tenant '{slug}'")
        return TenantContext(tenant_id=row.id, slug=row.slug)