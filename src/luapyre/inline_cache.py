from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class CacheState(Enum):
    EMPTY = auto()
    MONOMORPHIC = auto()
    POLYMORPHIC = auto()
    MEGAMORPHIC = auto()


@dataclass(slots=True)
class InlineCacheSite:
    """Bounded identity PIC with feedback-driven megamorphic fallback."""

    limit: int = 4
    entries: list[tuple[object, ...]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    megamorphic: bool = False

    @property
    def state(self) -> CacheState:
        if self.megamorphic:
            return CacheState.MEGAMORPHIC
        if not self.entries:
            return CacheState.EMPTY
        if len(self.entries) == 1:
            return CacheState.MONOMORPHIC
        return CacheState.POLYMORPHIC

    def match(self, key: tuple[object, ...]) -> tuple[object, ...] | None:
        if self.megamorphic:
            return None
        for entry in self.entries:
            if entry[0] is key[0] and entry[1:len(key)] == key[1:]:
                self.hits += 1
                return entry
        self.misses += 1
        return None

    def install(self, entry: tuple[object, ...], *, key_size: int = 1) -> None:
        if self.megamorphic:
            return
        for index, old in enumerate(self.entries):
            if old[0] is entry[0] and old[1:key_size] == entry[1:key_size]:
                self.entries[index] = entry
                return
        if len(self.entries) >= self.limit:
            self.entries.clear()
            self.megamorphic = True
            return
        self.entries.append(entry)


@dataclass(slots=True)
class DeoptSite:
    count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> int:
        self.count += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return self.count


class InlineCacheFeedback:
    """Per-bytecode-site caches and compact deoptimization feedback."""

    def __init__(self, *, polymorphic_limit: int = 4, retire_after: int = 3):
        self.polymorphic_limit = polymorphic_limit
        self.retire_after = retire_after
        self.calls: dict[tuple[int, int], InlineCacheSite] = {}
        self.tables: dict[tuple[int, int, str], InlineCacheSite] = {}
        self.deopts: dict[tuple[int, int], DeoptSite] = {}

    def call_site(self, proto: object, pc: int) -> InlineCacheSite:
        key = (id(proto), pc)
        return self.calls.setdefault(key, InlineCacheSite(self.polymorphic_limit))

    def table_site(self, proto: object, pc: int, operation: str) -> InlineCacheSite:
        key = (id(proto), pc, operation)
        return self.tables.setdefault(key, InlineCacheSite(self.polymorphic_limit))

    def record_deopt(self, proto: object, pc: int, reason: str) -> bool:
        site = self.deopts.setdefault((id(proto), pc), DeoptSite())
        return site.record(reason) >= self.retire_after

    def snapshot(self) -> dict[str, object]:
        states: dict[str, int] = {state.name.lower(): 0 for state in CacheState}
        sites = (*self.calls.values(), *self.tables.values())
        for site in sites:
            states[site.state.name.lower()] += 1
        return {
            "call_sites": len(self.calls),
            "table_sites": len(self.tables),
            "states": states,
            "hits": sum(site.hits for site in sites),
            "misses": sum(site.misses for site in sites),
            "invalidations": sum(site.invalidations for site in sites),
            "deopt_sites": len(self.deopts),
            "deopt_reasons": {
                reason: sum(site.reasons.get(reason, 0) for site in self.deopts.values())
                for reason in sorted({r for site in self.deopts.values() for r in site.reasons})
            },
        }
