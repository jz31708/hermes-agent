"""Centralized adaptive quota controller, sliding-window pacer, and daily circuit breaker for Gemini.

Provides:
- Concurrency-safe sliding-window RPM pacer based on time.monotonic() with boundary margin.
- SQLite-backed persistent state tracking observed attempts, learned/configured quotas, and daily blocks.
- Exact Pacific midnight (America/Los_Angeles) calculation handling DST transitions.
- Structured Google RPC 429 QuotaFailure parsing (distinguishing per-minute vs per-day).
- Automatic sticky circuit-breaker for daily exhaustion with seamless fallback to Hermes fallback_providers.
- Read-only snapshot & formatting for /quota, /gquota, and CLI.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_DB_PATH = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "gemini_quota.sqlite3"


class QuotaViolationType(str, Enum):
    REQUESTS_PER_MINUTE = "REQUESTS_PER_MINUTE"
    INPUT_TOKENS_PER_MINUTE = "INPUT_TOKENS_PER_MINUTE"
    REQUESTS_PER_DAY = "REQUESTS_PER_DAY"
    INPUT_TOKENS_PER_DAY = "INPUT_TOKENS_PER_DAY"
    OTHER_RATE_LIMIT = "OTHER_RATE_LIMIT"


@dataclass
class QuotaPolicy:
    model: str
    project_id: Optional[str] = None
    tier: str = "free"  # "free", "paid", "unknown"
    rpm_limit: Optional[int] = None  # None = unknown, never unlimited
    input_tpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    input_tpd_limit: Optional[int] = None
    source: str = "unknown"  # "runtime_evidence" > "configured" > "learned" > "unknown"
    observed_at: float = 0.0


@dataclass
class QuotaViolation:
    violation_type: QuotaViolationType
    quota_metric: Optional[str] = None
    quota_id: Optional[str] = None
    quota_dimensions: Optional[Dict[str, str]] = None
    quota_value: Optional[int] = None
    retry_delay: Optional[float] = None
    is_daily: bool = False
    raw_message: Optional[str] = None


@dataclass
class QuotaStatus:
    model: str
    project_id: Optional[str]
    tier: str
    rpd_limit: Optional[int]
    rpd_source: str
    observed_attempts_today: int
    estimated_remaining_today: Optional[int]
    rpm_limit: Optional[int]
    rpm_used_current_window: int
    tpm_limit: Optional[int]
    is_daily_blocked: bool
    daily_blocked_until_utc: Optional[float]
    daily_blocked_until_pacific_str: Optional[str]
    daily_blocked_until_paris_str: Optional[str]
    next_pacific_reset_utc: float
    next_pacific_reset_pacific_str: str
    next_pacific_reset_paris_str: str
    active_provider: str = "gemini"
    fallback_model: Optional[str] = None


def get_pacific_now(now_utc: Optional[datetime] = None) -> datetime:
    """Return the current datetime localized in America/Los_Angeles."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return now_utc.astimezone(PACIFIC_TZ)


def get_pacific_day_str(now_utc: Optional[datetime] = None) -> str:
    """Return the current calendar day in America/Los_Angeles as YYYY-MM-DD."""
    return get_pacific_now(now_utc).strftime("%Y-%m-%d")


def get_next_pacific_midnight_utc(now_utc: Optional[datetime] = None) -> datetime:
    """Calculate the next midnight in America/Los_Angeles and return it in UTC.
    Correctly accounts for PST/PDT Daylight Saving transitions.
    """
    pac_now = get_pacific_now(now_utc)
    next_pac_date = pac_now.date() + timedelta(days=1)
    # Combine with ZoneInfo ensures correct DST fold / offset calculation
    next_midnight_pac = datetime.combine(next_pac_date, dtime(0, 0, 0), tzinfo=PACIFIC_TZ)
    return next_midnight_pac.astimezone(timezone.utc)


def parse_google_rpc_retry_delay(val: Any) -> Optional[float]:
    """Parse Google's retryDelay (e.g. '4.234s' or float or dict)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    if isinstance(val, str):
        val_str = val.strip()
        if val_str.endswith("s"):
            val_str = val_str[:-1]
        try:
            return max(0.0, float(val_str))
        except (ValueError, TypeError):
            return None
    if isinstance(val, dict):
        seconds = float(val.get("seconds", 0))
        nanos = float(val.get("nanos", 0)) / 1e9
        return max(0.0, seconds + nanos)
    return None


class GeminiQuotaStorage:
    """Thread-safe SQLite storage for Gemini quota policy, attempts, and circuit-breaker."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quota_policy (
                    model TEXT PRIMARY KEY,
                    project_id TEXT,
                    tier TEXT,
                    rpm_limit INTEGER,
                    input_tpm_limit INTEGER,
                    rpd_limit INTEGER,
                    input_tpd_limit INTEGER,
                    source TEXT,
                    observed_at REAL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quota_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    day_pacific TEXT NOT NULL,
                    model TEXT NOT NULL,
                    success INTEGER NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_day ON quota_attempts (day_pacific, model);")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_blocks (
                    model TEXT PRIMARY KEY,
                    blocked_until_utc REAL NOT NULL,
                    day_pacific TEXT NOT NULL,
                    reason TEXT,
                    updated_at REAL NOT NULL
                );
            """)

    def record_attempt(self, model: str, success: bool, timestamp: float, day_pacific: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                "INSERT INTO quota_attempts (timestamp, day_pacific, model, success) VALUES (?, ?, ?, ?)",
                (timestamp, day_pacific, model, 1 if success else 0),
            )
            # Prune attempts older than 3 days to keep table compact
            conn.execute(
                "DELETE FROM quota_attempts WHERE timestamp < ?",
                (timestamp - 3 * 86400.0,),
            )

    def count_attempts_today(self, model: str, day_pacific: str) -> int:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM quota_attempts WHERE day_pacific = ? AND model = ?",
                (day_pacific, model),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_policy(self, model: str) -> Optional[QuotaPolicy]:
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT model, project_id, tier, rpm_limit, input_tpm_limit, rpd_limit, input_tpd_limit, source, observed_at FROM quota_policy WHERE model = ?",
                (model,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return QuotaPolicy(
                model=row[0],
                project_id=row[1],
                tier=row[2] or "free",
                rpm_limit=row[3],
                input_tpm_limit=row[4],
                rpd_limit=row[5],
                input_tpd_limit=row[6],
                source=row[7] or "unknown",
                observed_at=float(row[8] or 0.0),
            )

    def save_policy(self, policy: QuotaPolicy) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO quota_policy (model, project_id, tier, rpm_limit, input_tpm_limit, rpd_limit, input_tpd_limit, source, observed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model) DO UPDATE SET
                    project_id = COALESCE(excluded.project_id, quota_policy.project_id),
                    tier = COALESCE(excluded.tier, quota_policy.tier),
                    rpm_limit = COALESCE(excluded.rpm_limit, quota_policy.rpm_limit),
                    input_tpm_limit = COALESCE(excluded.input_tpm_limit, quota_policy.input_tpm_limit),
                    rpd_limit = COALESCE(excluded.rpd_limit, quota_policy.rpd_limit),
                    input_tpd_limit = COALESCE(excluded.input_tpd_limit, quota_policy.input_tpd_limit),
                    source = excluded.source,
                    observed_at = excluded.observed_at
            """, (
                policy.model,
                policy.project_id,
                policy.tier,
                policy.rpm_limit,
                policy.input_tpm_limit,
                policy.rpd_limit,
                policy.input_tpd_limit,
                policy.source,
                policy.observed_at,
            ))

    def set_daily_block(self, model: str, blocked_until_utc: float, day_pacific: str, reason: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                INSERT INTO daily_blocks (model, blocked_until_utc, day_pacific, reason, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model) DO UPDATE SET
                    blocked_until_utc = excluded.blocked_until_utc,
                    day_pacific = excluded.day_pacific,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
            """, (model, blocked_until_utc, day_pacific, reason, time.time()))

    def get_daily_block(self, model: str) -> Optional[Tuple[float, str, str]]:
        """Returns (blocked_until_utc, day_pacific, reason) or None."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT blocked_until_utc, day_pacific, reason FROM daily_blocks WHERE model = ?",
                (model,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return (float(row[0]), str(row[1]), str(row[2]))

    def clear_daily_block(self, model: str) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM daily_blocks WHERE model = ?", (model,))


class GeminiQuotaManager:
    """Centralized quota manager for Gemini models in Hermes.

    Manages:
    - 60-second true sliding-window RPM pacer.
    - Persistent SQLite storage surviving restarts.
    - Pacific-midnight daily reset calculation.
    - Structured 429 QuotaFailure parsing.
    - Daily circuit-breaker coordination with Hermes fallback providers.
    """

    def __init__(
        self,
        *,
        storage: Optional[GeminiQuotaStorage] = None,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        boundary_margin_ms: int = 50,
        clock: Optional[Callable[[], float]] = None,
        wall_clock: Optional[Callable[[], datetime]] = None,
    ):
        self.storage = storage or GeminiQuotaStorage(db_path)
        self.boundary_margin_s = max(0.0, boundary_margin_ms / 1000.0)
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

        # In-memory sliding window deque of monotonic timestamps: {model: deque([ts, ...])}
        self._window_history: Dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)

        # Last 429 tracking for observability
        self.last_429_at: Optional[float] = None
        self.last_429_classification: Optional[str] = None
        self.last_quota_evidence_source: Optional[str] = None

    # ── Clock Helpers ────────────────────────────────────────────────────────

    def monotonic(self) -> float:
        return self._clock()

    def now_utc(self) -> datetime:
        return self._wall_clock()

    def current_pacific_day(self) -> str:
        return get_pacific_day_str(self.now_utc())

    def next_pacific_midnight_utc(self) -> datetime:
        return get_next_pacific_midnight_utc(self.now_utc())

    # ── Policy & Discovery ───────────────────────────────────────────────────

    def get_effective_policy(self, model: str) -> QuotaPolicy:
        """Retrieve policy from storage or return default unknown policy."""
        policy = self.storage.get_policy(model)
        if policy is not None:
            return policy
        # Fallback default: unknown policy (None = unknown, not unlimited)
        return QuotaPolicy(
            model=model,
            project_id=None,
            tier="free",
            rpm_limit=None,
            input_tpm_limit=None,
            rpd_limit=None,
            input_tpd_limit=None,
            source="unknown",
            observed_at=self.now_utc().timestamp(),
        )

    def learn_quota_limits(
        self,
        model: str,
        *,
        rpm_limit: Optional[int] = None,
        input_tpm_limit: Optional[int] = None,
        rpd_limit: Optional[int] = None,
        input_tpd_limit: Optional[int] = None,
        project_id: Optional[str] = None,
        tier: Optional[str] = None,
        source: str = "runtime_evidence",
    ) -> None:
        """Update learned policy in persistent store if source has higher or equal precedence.
        Precedence: runtime_evidence > configured > learned > unknown
        """
        precedence = {"runtime_evidence": 4, "configured": 3, "learned": 2, "unknown": 1}
        current = self.get_effective_policy(model)
        if precedence.get(source, 0) < precedence.get(current.source, 0) and current.source != "unknown":
            logger.debug(
                "Skipping quota update for %s: current source '%s' has higher precedence than '%s'",
                model, current.source, source,
            )
            return

        updated = QuotaPolicy(
            model=model,
            project_id=project_id or current.project_id,
            tier=tier or current.tier,
            rpm_limit=rpm_limit if rpm_limit is not None else current.rpm_limit,
            input_tpm_limit=input_tpm_limit if input_tpm_limit is not None else current.input_tpm_limit,
            rpd_limit=rpd_limit if rpd_limit is not None else current.rpd_limit,
            input_tpd_limit=input_tpd_limit if input_tpd_limit is not None else current.input_tpd_limit,
            source=source,
            observed_at=self.now_utc().timestamp(),
        )
        self.storage.save_policy(updated)
        self.last_quota_evidence_source = source
        logger.info(
            "Learned Gemini quota policy for %s: RPM=%s RPD=%s source=%s",
            model, updated.rpm_limit, updated.rpd_limit, source,
        )

    # ── Daily Circuit Breaker ────────────────────────────────────────────────

    def is_daily_blocked(self, model: str = "gemini-3.8-flash") -> bool:
        """Check whether the model is currently under a daily quota exhaustion block."""
        block = self.storage.get_daily_block(model)
        if not block:
            return False
        blocked_until_utc, _day_pac, _reason = block
        current_utc_ts = self.now_utc().timestamp()
        if current_utc_ts >= blocked_until_utc:
            # Expired: clear block automatically
            self.storage.clear_daily_block(model)
            logger.info("Daily block expired for %s (reset passed); clearing block", model)
            return False
        return True

    def block_until_pacific_midnight(self, model: str, reason: str) -> float:
        """Block Gemini model until next midnight America/Los_Angeles."""
        next_midnight = self.next_pacific_midnight_utc()
        blocked_until_utc = next_midnight.timestamp()
        day_pac = self.current_pacific_day()
        self.storage.set_daily_block(model, blocked_until_utc, day_pac, reason)
        logger.warning(
            "Gemini %s DAILY BLOCKED until %s UTC (%s Pacific) | Reason: %s",
            model,
            next_midnight.isoformat(),
            next_midnight.astimezone(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S %Z"),
            reason,
        )
        return blocked_until_utc

    # ── Pacer (Sliding Window RPM) ───────────────────────────────────────────

    def acquire(self, model: str = "gemini-3.8-flash") -> float:
        """Acquire permission to send an API request to Gemini.

        - If daily blocked: raises GeminiDailyQuotaExhaustedError immediately.
        - If RPM limit is known (N):
          The first N requests in any 60-second window proceed with 0 wait.
          Request N+1 waits the exact millisecond delta until the oldest slot expires.
        - Records the attempt in SQLite (observed_attempts_today).

        Returns:
            Wait time in seconds (0.0 if no wait was needed).
        """
        if self.is_daily_blocked(model):
            block = self.storage.get_daily_block(model)
            blocked_until = block[0] if block else self.next_pacific_midnight_utc().timestamp()
            raise GeminiDailyQuotaExhaustedError(
                f"Gemini {model} daily quota exhausted until Pacific midnight",
                model=model,
                blocked_until_utc=blocked_until,
            )

        policy = self.get_effective_policy(model)
        rpm_limit = policy.rpm_limit

        with self._lock:
            now = self.monotonic()
            history = self._window_history[model]

            # Evict timestamps older than 60.0s
            cutoff = now - 60.0
            while history and history[0] <= cutoff:
                history.popleft()

            # If RPM limit is known, enforce true sliding-window pacing
            if rpm_limit is not None and rpm_limit > 0:
                if len(history) < rpm_limit:
                    wait_needed = 0.0
                    scheduled_time = now
                else:
                    # Slot that becomes free is rpm_limit slots behind
                    oldest_active_slot = history[-rpm_limit]
                    target_time = oldest_active_slot + 60.0 + self.boundary_margin_s
                    scheduled_time = max(now, target_time)
                    wait_needed = max(0.0, scheduled_time - now)

                history.append(scheduled_time)
            else:
                wait_needed = 0.0
                history.append(now)

        # Actual wait happens outside the lock to prevent lock starvation
        if wait_needed > 0.0:
            logger.debug(
                "Pacer throttle for %s: %s in window (limit %s), waiting %.3fs",
                model, len(history), rpm_limit, wait_needed,
            )
            time.sleep(wait_needed)

        # Record attempt in SQLite for daily counting
        self.storage.record_attempt(
            model=model,
            success=True,
            timestamp=self.now_utc().timestamp(),
            day_pacific=self.current_pacific_day(),
        )
        return wait_needed

    # ── 429 Error Classifier & Parser ────────────────────────────────────────

    def parse_429_error(
        self,
        *,
        status_code: int,
        headers: Any = None,
        body_json: Any = None,
        body_text: Optional[str] = None,
        model: str = "gemini-3.8-flash",
    ) -> Optional[QuotaViolation]:
        """Structured parser for HTTP 429 responses from Google Generative Language API.

        Extracts:
        - google.rpc.QuotaFailure violations[]
        - quotaMetric, quotaId, quotaDimensions, quotaValue
        - google.rpc.RetryInfo.retryDelay
        - HTTP Retry-After header
        - Classifies as REQUESTS_PER_MINUTE, INPUT_TOKENS_PER_MINUTE, REQUESTS_PER_DAY, etc.
        """
        if status_code != 429:
            return None

        now_ts = self.now_utc().timestamp()
        self.last_429_at = now_ts

        # Extract Retry-After header if present
        header_retry_delay: Optional[float] = None
        if headers and hasattr(headers, "get"):
            ra = headers.get("retry-after") or headers.get("Retry-After")
            if ra:
                try:
                    header_retry_delay = max(0.0, float(ra))
                except (ValueError, TypeError):
                    pass

        # Parse body JSON if needed
        data = body_json
        if data is None and body_text:
            try:
                data = json.loads(body_text)
            except Exception:
                data = {}
        data = data or {}

        err_obj = data.get("error") if isinstance(data, dict) else {}
        err_message = str(err_obj.get("message", "") if isinstance(err_obj, dict) else "")
        details = err_obj.get("details", []) if isinstance(err_obj, dict) and isinstance(err_obj.get("details"), list) else []

        body_retry_delay: Optional[float] = None
        violations_list: List[Dict[str, Any]] = []

        for detail in details:
            if not isinstance(detail, dict):
                continue
            type_url = str(detail.get("@type", ""))
            if type_url.endswith("/google.rpc.RetryInfo"):
                body_retry_delay = parse_google_rpc_retry_delay(detail.get("retryDelay"))
            elif type_url.endswith("/google.rpc.QuotaFailure"):
                for v in detail.get("violations", []):
                    if isinstance(v, dict):
                        violations_list.append(v)

        # Effective retry delay is the maximum of server-suggested delays
        effective_retry_delay = max(
            [d for d in [header_retry_delay, body_retry_delay] if d is not None],
            default=None,
        )

        parsed_violations: List[QuotaViolation] = []

        for v in violations_list:
            q_metric = str(v.get("quotaMetric") or "")
            q_id = str(v.get("quotaId") or "")
            q_dims = v.get("quotaDimensions") or {}
            desc = str(v.get("description") or err_message)

            # Extract numerical quota value directly from quotaValue if present, or regex on desc
            q_val: Optional[int] = None
            raw_q_val = v.get("quotaValue")
            if raw_q_val is not None:
                try:
                    q_val = int(str(raw_q_val).strip())
                except (ValueError, TypeError):
                    q_val = None

            if q_val is None:
                val_match = re.search(r"limit:\s*(\d+)", desc) or re.search(r"quota:\s*(\d+)", desc)
                if val_match:
                    try:
                        q_val = int(val_match.group(1))
                    except Exception:
                        pass

            # Structured classification:
            # 1. Inspect quotaId first for explicit temporal granularity
            # 2. Inspect quotaMetric
            # 3. Inspect description / message
            q_id_lower = q_id.lower()
            q_metric_lower = q_metric.lower()
            desc_lower = desc.lower()

            is_daily = False
            v_type = QuotaViolationType.OTHER_RATE_LIMIT

            if any(k in q_id_lower for k in ["perday", "requests_per_day", "per_day", "daily", "day"]):
                is_daily = True
                if any(k in q_id_lower or k in q_metric_lower for k in ["token", "tpd"]):
                    v_type = QuotaViolationType.INPUT_TOKENS_PER_DAY
                else:
                    v_type = QuotaViolationType.REQUESTS_PER_DAY
            elif any(k in q_id_lower for k in ["perminute", "requests_per_minute", "per_minute", "minute", "rpm"]):
                is_daily = False
                if any(k in q_id_lower or k in q_metric_lower for k in ["token", "tpm"]):
                    v_type = QuotaViolationType.INPUT_TOKENS_PER_MINUTE
                else:
                    v_type = QuotaViolationType.REQUESTS_PER_MINUTE
            elif any(k in q_metric_lower for k in ["per_day", "perday", "daily", "requests_per_day"]):
                is_daily = True
                v_type = QuotaViolationType.REQUESTS_PER_DAY
            elif any(k in q_metric_lower for k in ["token_per_day", "tokens_per_day", "tpd"]):
                is_daily = True
                v_type = QuotaViolationType.INPUT_TOKENS_PER_DAY
            elif any(k in q_metric_lower for k in ["token", "tpm", "tokens_per_minute"]):
                is_daily = False
                v_type = QuotaViolationType.INPUT_TOKENS_PER_MINUTE
            elif any(k in q_metric_lower for k in ["per_minute", "perminute", "rpm", "requests_per_minute", "free_tier_requests"]):
                is_daily = False
                v_type = QuotaViolationType.REQUESTS_PER_MINUTE
            else:
                combined_tag = f"{q_id_lower} {q_metric_lower} {desc_lower}"
                if any(k in combined_tag for k in ["perday", "daily", "day"]):
                    is_daily = True
                    v_type = QuotaViolationType.REQUESTS_PER_DAY
                else:
                    is_daily = False
                    v_type = QuotaViolationType.REQUESTS_PER_MINUTE

            parsed_violations.append(QuotaViolation(
                violation_type=v_type,
                quota_metric=q_metric,
                quota_id=q_id,
                quota_dimensions=q_dims,
                quota_value=q_val,
                retry_delay=effective_retry_delay,
                is_daily=is_daily,
                raw_message=desc,
            ))

        # Textual fallback if no structured QuotaFailure was found
        if not parsed_violations:
            combined_msg = f"{err_message} {body_text or ''}".lower()
            is_daily = "free_tier" in combined_msg and any(d in combined_msg for d in ["day", "daily", "perday"])
            v_type = QuotaViolationType.REQUESTS_PER_DAY if is_daily else QuotaViolationType.REQUESTS_PER_MINUTE
            q_val = None
            val_match = re.search(r"limit:\s*(\d+)", combined_msg)
            if val_match:
                try:
                    q_val = int(val_match.group(1))
                except Exception:
                    pass
            parsed_violations.append(QuotaViolation(
                violation_type=v_type,
                quota_value=q_val,
                retry_delay=effective_retry_delay,
                is_daily=is_daily,
                raw_message=err_message or (body_text or "")[:300],
            ))

        # Rule: A daily violation DOMINATES all transient violations
        daily_violations = [pv for pv in parsed_violations if pv.is_daily]
        selected_violation = daily_violations[0] if daily_violations else parsed_violations[0]

        # Learn limit from Google evidence if provided, respecting quotaDimensions model
        target_model = (selected_violation.quota_dimensions or {}).get("model") or model
        if selected_violation.quota_value is not None and selected_violation.quota_value > 0:
            if selected_violation.is_daily:
                self.learn_quota_limits(target_model, rpd_limit=selected_violation.quota_value, source="runtime_evidence")
            else:
                self.learn_quota_limits(target_model, rpm_limit=selected_violation.quota_value, source="runtime_evidence")

        self.last_429_classification = selected_violation.violation_type.value
        logger.warning(
            "Gemini 429 classified as %s (is_daily=%s, retry_delay=%ss, quota_value=%s) for %s",
            selected_violation.violation_type.value,
            selected_violation.is_daily,
            selected_violation.retry_delay,
            selected_violation.quota_value,
            target_model,
        )
        return selected_violation

    # ── Reporting & Status for /quota and CLI ─────────────────────────────────

    def get_status(self, model: str = "gemini-3.8-flash", fallback_model: Optional[str] = "gpt-5.6-luna") -> QuotaStatus:
        """Produce honest, comprehensive quota status descriptor."""
        policy = self.get_effective_policy(model)
        attempts_today = self.storage.count_attempts_today(model, self.current_pacific_day())
        rpd_limit = policy.rpd_limit

        estimated_remaining: Optional[int] = None
        if rpd_limit is not None:
            estimated_remaining = max(0, rpd_limit - attempts_today)

        # Count current window requests
        now_mono = self.monotonic()
        history = self._window_history[model]
        cutoff = now_mono - 60.0
        current_rpm_count = sum(1 for ts in history if ts > cutoff)

        block = self.storage.get_daily_block(model)
        is_blocked = self.is_daily_blocked(model)
        blocked_until_utc = block[0] if (block and is_blocked) else None

        blocked_pac_str: Optional[str] = None
        blocked_paris_str: Optional[str] = None
        if blocked_until_utc:
            dt_utc = datetime.fromtimestamp(blocked_until_utc, timezone.utc)
            blocked_pac_str = dt_utc.astimezone(PACIFIC_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
            blocked_paris_str = dt_utc.astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z")

        next_reset_dt = self.next_pacific_midnight_utc()
        next_reset_utc = next_reset_dt.timestamp()
        next_reset_pac_str = next_reset_dt.astimezone(PACIFIC_TZ).strftime("%H:%M %Z (%Y-%m-%d)")
        next_reset_paris_str = next_reset_dt.astimezone(ZoneInfo("Europe/Paris")).strftime("%H:%M %Z (%Y-%m-%d)")

        active_provider = "openai-codex" if is_blocked else "gemini"

        return QuotaStatus(
            model=model,
            project_id=policy.project_id,
            tier=policy.tier,
            rpd_limit=rpd_limit,
            rpd_source=policy.source,
            observed_attempts_today=attempts_today,
            estimated_remaining_today=estimated_remaining,
            rpm_limit=policy.rpm_limit,
            rpm_used_current_window=current_rpm_count,
            tpm_limit=policy.input_tpm_limit,
            is_daily_blocked=is_blocked,
            daily_blocked_until_utc=blocked_until_utc,
            daily_blocked_until_pacific_str=blocked_pac_str,
            daily_blocked_until_paris_str=blocked_paris_str,
            next_pacific_reset_utc=next_reset_utc,
            next_pacific_reset_pacific_str=next_reset_pac_str,
            next_pacific_reset_paris_str=next_reset_paris_str,
            active_provider=active_provider,
            fallback_model=fallback_model,
        )

    def render_markdown(self, model: str = "gemini-3.8-flash", fallback_model: Optional[str] = "gpt-5.6-luna") -> str:
        """Render honest terminal/chat display without invoking any LLM."""
        st = self.get_status(model=model, fallback_model=fallback_model)

        lines = [
            f"📊 Gemini API — {st.model}",
            "",
            f"Projet : {st.project_id or 'non spécifié'}",
            f"Tier détecté : {st.tier.title()}",
        ]

        # RPD line: honest labeling
        if st.rpd_limit is not None:
            source_tag = f"[{st.rpd_source}]" if st.rpd_source else ""
            lines.append(f"Limite RPD effective : {st.rpd_limit} {source_tag}".strip())
            lines.append(f"Tentatives locales observées : {st.observed_attempts_today}")
            lines.append(f"Restant estimé : {st.estimated_remaining_today} / {st.rpd_limit}")
        else:
            lines.append("Limite RPD effective : inconnue")
            lines.append(f"Tentatives locales observées : {st.observed_attempts_today}")

        # RPM / TPM line
        if st.rpm_limit is not None:
            lines.append(f"RPM : {st.rpm_used_current_window} / {st.rpm_limit}")
        else:
            lines.append(f"RPM : {st.rpm_used_current_window} / inconnue")

        if st.tpm_limit is not None:
            lines.append(f"TPM : limite connue ({st.tpm_limit})")
        else:
            lines.append("TPM : inconnue")

        lines.append("")
        lines.append(f"Provider actif : {st.active_provider}")
        if st.fallback_model:
            lines.append(f"Fallback Codex : {st.fallback_model}")

        if st.is_daily_blocked:
            lines.append("Circuit Gemini : 🔴 BLOQUÉ — quota journalier épuisé")
            if st.daily_blocked_until_paris_str:
                lines.append(f"Retour automatique : {st.daily_blocked_until_paris_str}")
        else:
            lines.append("Circuit Gemini : 🟢 disponible")

        lines.append("")
        lines.append("Reset quotidien :")
        lines.append(f"• {st.next_pacific_reset_pacific_str} (America/Los_Angeles)")
        lines.append(f"• {st.next_pacific_reset_paris_str} (Europe/Paris)")

        return "\n".join(lines)


class GeminiDailyQuotaExhaustedError(RuntimeError):
    """Raised when Gemini daily quota is exhausted and fallback should be activated."""

    def __init__(self, message: str, model: str, blocked_until_utc: float):
        super().__init__(message)
        self.model = model
        self.blocked_until_utc = blocked_until_utc


_GLOBAL_MANAGER: Optional[GeminiQuotaManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_quota_manager() -> GeminiQuotaManager:
    """Singleton getter for the global GeminiQuotaManager."""
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = GeminiQuotaManager()
        return _GLOBAL_MANAGER
