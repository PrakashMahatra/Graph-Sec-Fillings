import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.extract import extract_facts

SAMPLE = (
    "Each share of our Class A Common Stock is entitled to 1 vote per share. "
    "Each share of our Class B Common Stock is entitled to 10 votes per share. "
    "John Carter, our Chief Executive Officer, founded the company in 2015. "
    "Jane Carter beneficially owns approximately 42% of our Class B Common Stock. "
    "Our peer group includes Beta Industries, Gamma Corp and Delta Technologies."
)


def test_votes_per_share_extracted_with_evidence():
    facts = extract_facts(SAMPLE, self_company_name="Acme Holdings")
    voting = [f for f in facts if f.relation_type == "HAS_VOTING_RIGHTS"]
    votes = {f.source: f.properties["votes_per_share"] for f in voting}
    assert votes["Class A Common Stock"] == 1
    assert votes["Class B Common Stock"] == 10
    for f in voting:
        assert f.evidence_span  # every fact carries its literal source span


def test_role_extraction():
    facts = extract_facts(SAMPLE, self_company_name="Acme Holdings")
    roles = [f for f in facts if f.relation_type == "HOLDS_ROLE"]
    assert any(f.source == "John Carter" and "Chief Executive Officer" in f.target for f in roles)


def test_ownership_extraction():
    facts = extract_facts(SAMPLE, self_company_name="Acme Holdings")
    owns = [f for f in facts if f.relation_type == "OWNS"]
    assert any(f.source == "Jane Carter" and f.properties["percent"] == 42.0 for f in owns)


def test_peer_extraction_splits_all_names():
    facts = extract_facts(SAMPLE, self_company_name="Acme Holdings")
    peers = {f.target for f in facts if f.relation_type == "PEER_OF"}
    assert peers == {"Beta Industries", "Gamma Corp", "Delta Technologies"}


def test_no_fact_without_evidence_span():
    facts = extract_facts(SAMPLE, self_company_name="Acme Holdings")
    assert all(f.evidence_span and len(f.evidence_span) > 0 for f in facts)
