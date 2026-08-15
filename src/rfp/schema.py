from dataclasses import dataclass


@dataclass
class Requirement:
    req_id: str
    text: str


@dataclass
class Chapter:
    id: str
    title: str
    requirements: list[Requirement]


@dataclass
class RFP:
    rfp_id: str
    title: str
    industry: str
    chapters: list[Chapter]


@dataclass
class Sentence:
    sent_id: str
    proposal_id: str
    responds_to: str
    section: str
    text: str
    claim_kind: str
    flags: dict
