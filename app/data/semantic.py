"""Semantic role detection (revenue, profit, customer, ...).

Evidence = column-name match + type compatibility + statistical sanity check.
A name match alone is never enough: an incompatible type or implausible values
(e.g. an 'age' column ranging 500-900) removes or lowers the candidate.
The type-only fallback (base > 0) lets a datetime column be recognised as the
date axis even when it is called 'when'.
"""
from collections.abc import Callable
from dataclasses import dataclass

from app.config.settings import settings
from app.data.schemas import ColumnProfile, SemanticMatch, SemanticResult
from app.utils.text import stem, tokenize_name

MIN_STAT_SCORE = 0.3


def _fs(*words: str) -> frozenset[str]:
    return frozenset(words)


@dataclass(frozen=True)
class RoleSpec:
    role: str
    keywords: frozenset[str]
    types: frozenset[str]
    check: Callable[[ColumnProfile], float]
    exclude: frozenset[str] = frozenset()
    base: float = 0.0                     # confidence from type+stats alone (no name hint)
    boost: frozenset[str] = frozenset()   # extra tokens that add a small bonus


# ---------- statistical checks (return 0..1) ----------
def _num(p: ColumnProfile) -> float | None:
    return float(p.min_value) if isinstance(p.min_value, (int, float)) else None


def _always(p: ColumnProfile) -> float:
    return 1.0


def _nonneg(p: ColumnProfile) -> float:
    lo = _num(p)
    return 0.0 if lo is None else (1.0 if lo >= 0 else 0.4)


def _signed(p: ColumnProfile) -> float:
    return 0.0 if _num(p) is None else 1.0


def _quantity(p: ColumnProfile) -> float:
    lo = _num(p)
    if lo is None:
        return 0.0
    if lo < 0:
        return 0.4
    return 1.0 if p.integer_like else 0.6


def _share(p: ColumnProfile) -> float:
    lo, hi = _num(p), p.max_value
    if lo is None or not isinstance(hi, (int, float)):
        return 0.0
    return 1.0 if lo >= 0 and hi <= 100 else 0.3


def _age(p: ColumnProfile) -> float:
    lo, hi = _num(p), p.max_value
    if lo is None or not isinstance(hi, (int, float)):
        return 0.0
    return 1.0 if lo >= 0 and hi <= 120 else 0.0


def _entity(p: ColumnProfile) -> float:
    return 1.0 if p.unique_count >= 2 else 0.0


def _dimension(p: ColumnProfile) -> float:
    """Grouping dimensions should have fewer unique values than rows."""
    return 1.0 if p.cardinality_ratio <= 0.5 else 0.6


_ENTITY_TYPES = _fs("id", "categorical", "text")
_NUMERIC = _fs("numeric")
_RATE_WORDS = _fs("rate", "pct", "percent", "percentage", "ratio", "growth")

# Order matters: it breaks confidence ties (earlier = higher priority).
ROLE_SPECS: list[RoleSpec] = [
    RoleSpec("date", _fs("date", "datetime", "timestamp", "time", "dt", "period"),
             _fs("datetime"), _always, base=0.6,
             boost=_fs("order", "transaction", "invoice", "purchase", "sale", "txn")),
    RoleSpec("order_id", _fs("order", "invoice", "transaction", "txn", "receipt", "basket"),
             _fs("id"), _always),
    RoleSpec("customer", _fs("customer", "client", "cust", "buyer", "user", "member", "account"),
             _ENTITY_TYPES, _entity,
             exclude=_fs("age", "gender", "satisfaction", "segment", "tenure", "since", "count",
                         "lifetime", "ltv", "clv", "type")),
    RoleSpec("product", _fs("product", "item", "sku", "article", "merchandise"),
             _ENTITY_TYPES, _entity,
             exclude=_fs("category", "subcategory", "price", "cost", "quantity", "qty", "count")),
    RoleSpec("category", _fs("category", "subcategory", "categories", "segment", "genre"),
             _fs("categorical", "id", "text"), _dimension),
    RoleSpec("region", _fs("region", "state", "province", "country", "city", "location", "area",
                           "zone", "territory", "market", "store", "branch"),
             _ENTITY_TYPES, _dimension,
             exclude=_fs("manager", "rep", "representative", "person", "head")),
    RoleSpec("channel", _fs("channel", "medium", "platform", "source"),
             _fs("categorical", "text"), _dimension),
    RoleSpec("campaign", _fs("campaign", "adgroup", "creative"), _ENTITY_TYPES, _entity),
    RoleSpec("revenue", _fs("revenue", "sales", "turnover", "gmv", "earnings"), _NUMERIC, _nonneg,
             exclude=_RATE_WORDS | _fs("margin", "tax", "rep", "representative", "manager", "person",
                                       "team", "target", "forecast", "per", "count")),
    RoleSpec("profit", _fs("profit", "pnl", "netprofit"), _NUMERIC, _signed,
             exclude=_RATE_WORDS | _fs("margin")),
    RoleSpec("margin", _fs("margin"), _NUMERIC, _signed),
    RoleSpec("cost", _fs("cost", "cogs", "expense", "expenditure"), _NUMERIC, _nonneg,
             exclude=_RATE_WORDS | _fs("acquisition", "cac", "click", "per")),
    RoleSpec("spend", _fs("spend", "spent", "budget", "adspend", "investment"), _NUMERIC, _nonneg),
    RoleSpec("price", _fs("price", "mrp", "unitprice"), _NUMERIC, _nonneg,
             exclude=_fs("change", "index", "growth")),
    RoleSpec("quantity", _fs("quantity", "qty", "units", "volume", "pieces"), _NUMERIC, _quantity,
             exclude=_fs("price", "cost", "rate")),
    RoleSpec("discount", _fs("discount", "disc", "rebate"), _NUMERIC, _nonneg),
    RoleSpec("impressions", _fs("impressions", "views", "reach"), _NUMERIC, _nonneg,
             exclude=_RATE_WORDS),
    RoleSpec("clicks", _fs("clicks"), _NUMERIC, _nonneg, exclude=_RATE_WORDS | _fs("through")),
    RoleSpec("conversions", _fs("conversions", "leads", "signups"), _NUMERIC, _nonneg,
             exclude=_RATE_WORDS),
    RoleSpec("amount", _fs("amount", "amt"), _NUMERIC, _signed, exclude=_fs("tax", "discount")),
    RoleSpec("salary", _fs("salary", "wage", "compensation", "ctc", "pay", "income"),
             _NUMERIC, _nonneg, exclude=_RATE_WORDS),
    RoleSpec("employee", _fs("employee", "emp", "staff", "worker", "personnel"),
             _ENTITY_TYPES, _entity),
    RoleSpec("department", _fs("department", "dept", "division", "team"),
             _fs("categorical", "text"), _dimension),
    RoleSpec("tenure", _fs("tenure", "seniority", "yearsatcompany", "experience"),
             _NUMERIC, _nonneg),
    RoleSpec("age", _fs("age"), _NUMERIC, _age),
    RoleSpec("target", _fs("churn", "churned", "attrition", "exited", "left", "fraud", "isfraud",
                           "default", "defaulted", "converted", "target", "label", "outcome"),
             _fs("boolean"), _always),
]


def _name_score(tokens: list[str], spec: RoleSpec) -> tuple[float | None, str | None]:
    """Return (score, matched_keyword). score None means an exclusion word was found."""
    stems = {stem(t) for t in tokens}
    compact = "".join(tokens)
    for ex in spec.exclude:
        e = stem(ex)
        if e in stems or (len(e) >= 5 and e in compact):
            return None, None
    best, hit = 0.0, None
    for kw in spec.keywords:
        k = stem(kw)
        if k in stems:
            return 1.0, kw
        if len(k) >= 5 and k in compact:
            best, hit = 0.8, kw
    return best, hit


def detect_semantic_roles(columns: list[ColumnProfile]) -> SemanticResult:
    candidates: dict[str, list[SemanticMatch]] = {}
    for p in columns:
        tokens = tokenize_name(p.name)
        scored: list[tuple[float, int, SemanticMatch]] = []
        for prio, spec in enumerate(ROLE_SPECS):
            if p.inferred_type not in spec.types:
                continue
            name_s, kw = _name_score(tokens, spec)
            if name_s is None:
                continue
            stat = spec.check(p)
            if stat < MIN_STAT_SCORE:
                continue
            if name_s > 0:
                conf = 0.65 * name_s + 0.35 * stat
                if spec.boost & {stem(t) for t in tokens} | (spec.boost & set(tokens)):
                    conf += 0.05
                evidence = [f"name matches '{kw}'"]
            elif spec.base > 0:
                conf = spec.base * stat
                evidence = ["no name hint; matched on data type alone"]
            else:
                continue
            conf = round(min(conf, 1.0), 3)
            if conf < settings.semantic_min_confidence:
                continue
            evidence += [f"type '{p.inferred_type}' compatible", f"statistical check {stat:.2f}"]
            scored.append((conf, prio, SemanticMatch(
                column=p.name, role=spec.role, confidence=conf, evidence=evidence)))
        if scored:
            scored.sort(key=lambda t: (-t[0], t[1]))
            candidates[p.name] = [m for _, _, m in scored]

    primary = {col: ms[0] for col, ms in candidates.items()}
    roles: dict[str, list[str]] = {}
    for col, m in sorted(primary.items(), key=lambda kv: -kv[1].confidence):
        roles.setdefault(m.role, []).append(col)
    return SemanticResult(
        primary=primary, candidates=candidates, roles=roles,
        unmatched_columns=[p.name for p in columns if p.name not in primary],
    )
