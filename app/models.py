import uuid
import datetime as dt

from sqlalchemy import (
    Column, String, Text, ForeignKey, DateTime, Integer, Float, Index,
    Computed,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, TSVECTOR

from pgvector.sqlalchemy import Vector

from app.config import EMBED_DIM

Base = declarative_base()


def now():
    return dt.datetime.utcnow()


def _pk():
    # Native UUID PK (matches db/init.sql UUID + gen_random_uuid()).
    return Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _tenant_fk(nullable=False):
    return Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=nullable, index=True)


def _uuid_fk(target, nullable=True):
    return Column(PG_UUID(as_uuid=True), ForeignKey(target), nullable=nullable)


def _uuid_nullable():
    # Nullable UUID WITHOUT a ForeignKey (self-refs, duplicate_of).
    return Column(PG_UUID(as_uuid=True), nullable=True)


def _jsonb(default_factory):
    from sqlalchemy.dialects.postgresql import JSONB as _JSONB
    return Column(_JSONB, default=default_factory)


class Tenant(Base):
    __tablename__ = "tenants"
    id = _pk()
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=now)


class User(Base):
    __tablename__ = "users"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    email = Column(String, nullable=False)
    role = Column(String, default="member")
    created_at = Column(DateTime, default=now)


class Company(Base):
    __tablename__ = "companies"
    id = _pk()
    cik = Column(String, index=True)
    name = Column(String, nullable=False)
    ticker = Column(String)


class Filing(Base):
    __tablename__ = "filings"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    company_id = _uuid_fk("companies.id", nullable=False)
    accession_number = Column(String, nullable=False)
    cik = Column(String)
    ticker = Column(String)
    form_type = Column(String)
    filing_date = Column(String)
    period_end_date = Column(String)
    filing_url = Column(String)
    source_hash = Column(String)
    language = Column(String, default="en")
    status = Column(String, default="ingested")
    quality_status = Column(String, default="ok")
    supersedes_filing_id = _uuid_fk("filings.id", nullable=True)
    created_at = Column(DateTime, default=now)

    company = relationship("Company")

    __table_args__ = (
        Index("ux_filing_idem", "tenant_id", "accession_number", unique=True),
    )


class Document(Base):
    __tablename__ = "documents"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    filing_id = _uuid_fk("filings.id", nullable=False)
    document_type = Column(String)
    raw_uri = Column(String)
    content_hash = Column(String, index=True)
    duplicate_of = _uuid_nullable()
    source_language = Column(String, default="en")
    metadata_json = _jsonb(dict)
    created_at = Column(DateTime, default=now)


class Section(Base):
    __tablename__ = "sections"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    document_id = _uuid_fk("documents.id", nullable=False)
    parent_section_id = _uuid_nullable()
    heading = Column(String)
    section_path = Column(String)
    ordinal = Column(Integer, default=0)
    page_start = Column(Integer, default=1)
    page_end = Column(Integer, default=1)


class Chunk(Base):
    __tablename__ = "chunks"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    document_id = _uuid_fk("documents.id", nullable=False)
    section_id = _uuid_fk("sections.id", nullable=True)
    chunk_index = Column(Integer, default=0)
    text = Column(Text, nullable=False)
    text_hash = Column(String, index=True)
    token_count = Column(Integer, default=0)
    page_start = Column(Integer, default=1)
    page_end = Column(Integer, default=1)
    source_locator = _jsonb(dict)
    metadata_json = _jsonb(dict)
    embedding = Column(Vector(EMBED_DIM), nullable=True)
    search_tsv = Column(TSVECTOR, Computed("to_tsvector('english', coalesce(text,''))",
                                           persisted=True), nullable=True)
    quality_flags = _jsonb(list)

    __table_args__ = (
        Index("ix_chunks_tenant_doc", "tenant_id", "document_id"),
        # pgvector HNSW ANN index -- cosine distance over the 384-d MiniLM
        # embeddings, declared here so create_all() owns every index.
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN full-text index over the generated `search_tsv` column.
        Index("chunks_tsv_idx", "search_tsv", postgresql_using="gin"),
    )


class GraphNode(Base):
    __tablename__ = "graph_nodes"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    node_type = Column(String, nullable=False)  # Company, SecurityClass, Person, ...
    canonical_name = Column(String, nullable=False, index=True)
    external_id = Column(String, nullable=True)
    properties = _jsonb(dict)
    valid_from = Column(String, nullable=True)
    valid_to = Column(String, nullable=True)

    __table_args__ = (
        Index("ux_node_identity", "tenant_id", "node_type", "canonical_name", unique=True),
    )


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    id = _pk()
    tenant_id = _tenant_fk(nullable=False)
    source_node_id = _uuid_fk("graph_nodes.id", nullable=False)
    target_node_id = _uuid_fk("graph_nodes.id", nullable=False)
    relation_type = Column(String, nullable=False)
    valid_from = Column(String, nullable=True)
    valid_to = Column(String, nullable=True)
    filing_id = _uuid_fk("filings.id", nullable=True)
    evidence_chunk_id = _uuid_fk("chunks.id", nullable=True)
    properties = _jsonb(dict)
    # Every edge MUST originate from an extraction rule + evidence span --
    # never written because a model "felt" the relation was true.
    confidence = Column(Float, default=1.0)

    __table_args__ = (
        Index("ix_edge_tenant_src", "tenant_id", "source_node_id"),
        Index("ix_edge_tenant_tgt", "tenant_id", "target_node_id"),
    )
