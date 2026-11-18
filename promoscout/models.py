from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Kind = Literal["camera", "action", "sfx", "dialogue"]


class Block(BaseModel):
    block_id: str
    episode: int
    scene: int
    kind: Kind
    speaker_raw: Optional[str] = None
    character: Optional[str] = None      # canonical name via alias table
    hidden_identity: bool = False        # audience does not know this person's name yet
    tone: Optional[str] = None           # emotion label written in the script
    start: float
    end: float
    text: str                            # verbatim screenplay text
    src_line: int                        # 1-based line in the original file


class Scene(BaseModel):
    episode: int
    index: int
    heading: str
    start: float
    end: float
    blocks: list[Block] = Field(default_factory=list)


class Episode(BaseModel):
    number: int
    filename: str
    logline: str = ""
    continuity: str = ""
    duration: float = 0.0
    scenes: list[Scene] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def blocks(self) -> list[Block]:
        return [b for s in self.scenes for b in s.blocks]


class Segment(BaseModel):
    episode: int
    start: float
    end: float
    block_ids: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start


class DetFeatures(BaseModel):
    hook: float
    cliffhanger: float
    clarity: float
    emotion: float
    signals: dict[str, list[str]] = Field(default_factory=dict)   # dimension -> reasons
    hook_text: str = ""
    cliff_text: str = ""
    hook_block: str = ""
    cliff_block: str = ""


class Evidence(BaseModel):
    score: float = Field(ge=0, le=10)
    block_id: str = ""
    evidence_quote: str = ""
    reason: str = ""
    open_question: str = ""


class Clarity(BaseModel):
    score: float = Field(ge=0, le=10)
    who: str = ""
    what: str = ""
    why_it_matters: str = ""
    missing_context: list[str] = Field(default_factory=list)


class LLMJudgment(BaseModel):
    candidate_id: str
    hook: Evidence
    cliffhanger: Evidence
    clarity: Clarity
    emotion: Evidence
    one_line_why: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    verdict: Literal["post", "maybe", "reject"] = "maybe"
    pair_coherent: Optional[bool] = None
    pair_reason: str = ""


class ScoreCard(BaseModel):
    hook: float = 0
    cliffhanger: float = 0
    clarity: float = 0
    emotion: float = 0
    total: float = 0                      # weighted, out of 100
    confidence: float = 0
    source: Literal["llm", "deterministic"] = "deterministic"
    evidence: dict[str, str] = Field(default_factory=dict)


class AuditFlag(BaseModel):
    code: str
    severity: Literal["info", "warn", "error"]
    message: str


class Candidate(BaseModel):
    candidate_id: str
    segments: list[Segment]
    origin: Literal["window", "joined"] = "window"
    duration: float
    det: DetFeatures
    llm: Optional[LLMJudgment] = None
    scores: ScoreCard = Field(default_factory=ScoreCard)
    flags: list[AuditFlag] = Field(default_factory=list)
    status: Literal["approved", "rejected", "duplicate", "needs_review", "candidate"] = "candidate"
    rank: Optional[int] = None
    tier: Optional[str] = None
    duplicate_of: Optional[str] = None
    hook: str = ""
    cliffhanger: str = ""
    why: str = ""
    text: str = ""                        # verbatim text of all included blocks, for audit/dedupe


class IngestReport(BaseModel):
    files_seen: int = 0
    episodes_parsed: int = 0
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
