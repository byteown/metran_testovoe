from dataclasses import dataclass, field

from fin_agent.schemas import Calculation, Source


@dataclass
class ToolResult:
    rows: list[dict]
    calculations: list[Calculation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
