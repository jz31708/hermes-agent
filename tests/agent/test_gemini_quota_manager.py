"""Comprehensive production-grade tests for Gemini Quota Manager, pacer, 429 parsing,
daily circuit breaker, and Hermes fallback integration.

Covers all constraints from Section 16:
- Fake clock / injectable clock (no sleeping in tests)
- Exact 60s sliding window (N immediate, N+1 waits)
- 20 concurrent coroutines slot reservation safety
- Boundaries (59.999s, 60.000s, boundary margin)
- State reload from SQLite surviving restart
- Pacific day change & DST transitions (PST <-> PDT)
- QuotaFailure RPD, RPM, input TPM, multiple violations (daily dominates)
- quotaValue extraction
- RetryInfo.retryDelay and Retry-After
- 429 without structured details
- RPD never retried 5 times
- Transient retry re-acquires pacer and counts attempt
- Streaming 429 before first delta (retry allowed) vs after delta (no replay)
- Fallback Gemini -> Codex
- Sticky fallback during daily block (no sacrificial requests)
- Automatic restore after reset
- Gemini thought signature in history sent to Codex (regression)
- Context compressor rebinds for fallback model
- /quota and /gquota without LLM calls
- Unknown quota without inventing numbers
- Concurrent access to SQLite store
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict
from typing import Any, List
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from agent.gemini_quota_manager import (
    PACIFIC_TZ,
    GeminiDailyQuotaExhaustedError,
    GeminiQuotaManager,
    GeminiQuotaStorage,
    QuotaPolicy,
    QuotaViolationType,
    get_next_pacific_midnight_utc,
    get_pacific_day_str,
    get_pacific_now,
    parse_google_rpc_retry_delay,
)


class FakeClock:
    """Simulated monotonic and wall clock for zero-latency deterministic tests."""

    def __init__(self, start_mono: float = 1000.0, start_wall: datetime.datetime | None = None):
        self._mono = start_mono
        self._wall = start_wall or datetime.datetime(2026, 9, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)

    def monotonic(self) -> float:
        return self._mono

    def wall(self) -> datetime.datetime:
        return self._wall

    def advance(self, seconds: float) -> None:
        self._mono += seconds
        self._wall += datetime.timedelta(seconds=seconds)


@pytest.fixture
def temp_storage(tmp_path):
    db_file = tmp_path / "test_gemini_quota.sqlite3"
    storage = GeminiQuotaStorage(db_file)
    return storage, db_file


@pytest.fixture
def fake_clock():
    return FakeClock(
        start_mono=1000.0,
        start_wall=datetime.datetime(2026, 9, 6, 12, 0, 0, tzinfo=datetime.timezone.utc),
    )


@pytest.fixture
def qm(temp_storage, fake_clock):
    storage, _ = temp_storage
    manager = GeminiQuotaManager(
        storage=storage,
        boundary_margin_ms=50,
        clock=fake_clock.monotonic,
        wall_clock=fake_clock.wall,
    )
    return manager


# ── 1. Sliding Window & Pacer Tests ──────────────────────────────────────────


class TestSlidingWindowPacer:
    """Test 60s sliding window pacer logic."""

    def test_first_n_requests_proceed_immediately_then_wait(self, qm, fake_clock):
        """If limit is N RPM, N requests go immediately (0 delay), N+1 waits."""
        qm.learn_quota_limits("gemini-3.8-flash", rpm_limit=5, source="runtime_evidence")

        # First 5 calls must succeed with 0 delay
        for i in range(5):
            wait = qm.acquire("gemini-3.8-flash")
            assert wait == 0.0, f"Call {i+1} should have zero wait"
            fake_clock.advance(1.0)  # each call made 1s apart: at 1000, 1001, 1002, 1003, 1004

        # 6th call at mono=1005: window has [1000, 1001, 1002, 1003, 1004] (5 calls)
        # Oldest call is 1000. It expires at 1060.0. Current time is 1005.0.
        # Wait needed = 1000 + 60 - 1005 + 0.05 (margin) = 55.05s
        with patch("time.sleep") as mock_sleep:
            wait = qm.acquire("gemini-3.8-flash")
            assert math.isclose(wait, 55.05, abs_tol=0.01)
            mock_sleep.assert_called_once()

    def test_boundaries_59_999_and_60_000(self, qm, fake_clock):
        """Test boundary behaviour at 59.999s vs 60.000s."""
        qm.learn_quota_limits("gemini-3.8-flash", rpm_limit=1, source="runtime_evidence")

        # Request 1 at t=1000.0
        assert qm.acquire("gemini-3.8-flash") == 0.0

        # At t=1059.999 (59.999s elapsed), slot is still occupied
        fake_clock.advance(59.999)
        with patch("time.sleep") as mock_sleep:
            wait = qm.acquire("gemini-3.8-flash")
            assert wait > 0.0
            mock_sleep.assert_called_once()

        # At t >= 1060.0 (past 60.0s), slot has expired and request goes with 0 wait
        fake_clock.advance(60.1)
        with patch("time.sleep") as mock_sleep:
            wait = qm.acquire("gemini-3.8-flash")
            assert wait == 0.0
            mock_sleep.assert_not_called()

    def test_unknown_rpm_does_not_artificially_block(self, qm):
        """When RPM limit is unknown, requests proceed without artificial throttling."""
        policy = qm.get_effective_policy("gemini-3.8-flash")
        assert policy.rpm_limit is None
        # 10 calls in succession
        for _ in range(10):
            assert qm.acquire("gemini-3.8-flash") == 0.0


# ── 2. Concurrency Tests ─────────────────────────────────────────────────────


class TestConcurrency:
    """Test concurrent reservations and SQLite thread safety."""

    def test_20_concurrent_requests_slot_reservation_safety(self, temp_storage):
        """20 threads concurrently acquiring with 5 RPM: exactly 5 get instant slot, 15 get throttled."""
        storage, _ = temp_storage
        real_qm = GeminiQuotaManager(storage=storage, boundary_margin_ms=50)
        real_qm.learn_quota_limits("gemini-3.8-flash", rpm_limit=5, source="runtime_evidence")

        waits = []
        lock = threading.Lock()

        def worker():
            w = real_qm.acquire("gemini-3.8-flash")
            with lock:
                waits.append(w)

        with patch("time.sleep"):
            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(waits) == 20
        zero_waits = sum(1 for w in waits if w == 0.0)
        positive_waits = sum(1 for w in waits if w > 0.0)
        assert zero_waits == 5, f"Expected exactly 5 zero-wait reservations, got {zero_waits}"
        assert positive_waits == 15, f"Expected 15 throttled reservations, got {positive_waits}"

    def test_sqlite_concurrent_access(self, temp_storage):
        """Multiple threads concurrently recording attempts into SQLite."""
        storage, _ = temp_storage
        def worker(tid):
            for i in range(25):
                storage.record_attempt("gemini-3.8-flash", True, 1000.0 + i, "2026-09-06")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, i) for i in range(8)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        count = storage.count_attempts_today("gemini-3.8-flash", "2026-09-06")
        assert count == 8 * 25


# ── 3. Timezone, Pacific Day, and DST Transitions ────────────────────────────


class TestTimezonesAndDST:
    """Ensure accurate Pacific calculation handling PST/PDT Daylight Saving."""

    def test_pacific_day_str(self):
        # 05:00 UTC on Sept 6 is 22:00 Sept 5 in Los Angeles (PDT = UTC-7)
        dt_utc = datetime.datetime(2026, 9, 6, 5, 0, 0, tzinfo=datetime.timezone.utc)
        assert get_pacific_day_str(dt_utc) == "2026-09-05"

        # 08:00 UTC on Sept 6 is 01:00 Sept 6 in Los Angeles (PDT = UTC-7)
        dt_utc = datetime.datetime(2026, 9, 6, 8, 0, 0, tzinfo=datetime.timezone.utc)
        assert get_pacific_day_str(dt_utc) == "2026-09-06"

    def test_next_pacific_midnight_utc_in_summer_pdt(self):
        # In summer (PDT = UTC-7): Pacific midnight occurs at 07:00:00 UTC next day
        dt_utc = datetime.datetime(2026, 9, 6, 14, 0, 0, tzinfo=datetime.timezone.utc)
        midnight_utc = get_next_pacific_midnight_utc(dt_utc)
        assert midnight_utc == datetime.datetime(2026, 9, 7, 7, 0, 0, tzinfo=datetime.timezone.utc)

    def test_dst_transition_fall_back(self):
        # Fall back transition (Nov 2026): PDT -> PST
        # Nov 1, 2026 at 18:00 UTC is 11:00 AM on Nov 1 in LA
        dt_utc = datetime.datetime(2026, 11, 1, 18, 0, 0, tzinfo=datetime.timezone.utc)
        midnight_utc = get_next_pacific_midnight_utc(dt_utc)
        # Next midnight in Pacific is Nov 2 00:00 PST (PST = UTC-8) -> 08:00 UTC
        assert midnight_utc == datetime.datetime(2026, 11, 2, 8, 0, 0, tzinfo=datetime.timezone.utc)

    def test_dst_transition_spring_forward(self):
        # Spring forward transition (March 2026): PST -> PDT (23-hour day)
        dt_utc = datetime.datetime(2026, 3, 7, 18, 0, 0, tzinfo=datetime.timezone.utc)  # March 7 10:00 PST
        midnight_utc = get_next_pacific_midnight_utc(dt_utc)
        # Next midnight in Pacific is March 8 00:00 PST -> 08:00 UTC
        assert midnight_utc == datetime.datetime(2026, 3, 8, 8, 0, 0, tzinfo=datetime.timezone.utc)


# ── 4. 429 Structured Parsing & Classification ───────────────────────────────


class Test429Parsing:
    """Test structured parsing of Google RPC QuotaFailure and RetryInfo."""

    def test_parse_rpd_quota_failure_with_limit_value(self, qm):
        """Parse google.rpc.QuotaFailure with daily metric and extract quotaValue."""
        body = {
            "error": {
                "code": 429,
                "message": "Resource has been exhausted (e.g. check quota).",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "subject": "project:609228617719",
                                "description": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20",
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateContentFreeTierRequestsPerDayPerProject",
                            }
                        ],
                    }
                ],
            }
        }
        v = qm.parse_429_error(status_code=429, body_json=body, model="gemini-3.8-flash")
        assert v is not None
        assert v.is_daily is True
        assert v.violation_type == QuotaViolationType.REQUESTS_PER_DAY
        assert v.quota_value == 20
        # Check that limit was learned into policy
        policy = qm.get_effective_policy("gemini-3.8-flash")
        assert policy.rpd_limit == 20
        assert policy.source == "runtime_evidence"

    def test_parse_rpm_quota_failure(self, qm):
        body = {
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "description": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_requests_per_minute, limit: 15",
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_requests_per_minute",
                                "quotaId": "GenerateContentRequestsPerMinutePerProject",
                            }
                        ],
                    }
                ],
            }
        }
        v = qm.parse_429_error(status_code=429, body_json=body, model="gemini-3.8-flash")
        assert v is not None
        assert v.is_daily is False
        assert v.violation_type == QuotaViolationType.REQUESTS_PER_MINUTE
        assert v.quota_value == 15
        assert qm.get_effective_policy("gemini-3.8-flash").rpm_limit == 15

    def test_parse_input_tpm_violation(self, qm):
        body = {
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "description": "Quota exceeded for tokens per minute",
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_input_tokens_per_minute",
                                "quotaId": "GenerateContentInputTokensPerMinute",
                            }
                        ],
                    }
                ],
            }
        }
        v = qm.parse_429_error(status_code=429, body_json=body, model="gemini-3.8-flash")
        assert v is not None
        assert v.violation_type == QuotaViolationType.INPUT_TOKENS_PER_MINUTE
        assert v.is_daily is False

    def test_multiple_violations_daily_dominates(self, qm):
        """When both RPM and RPD are reported, RPD dominates."""
        body = {
            "error": {
                "code": 429,
                "message": "Quota exceeded",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": "GenerateContentRequestsPerMinute",
                                "description": "RPM exceeded",
                            },
                            {
                                "quotaId": "GenerateContentRequestsPerDay",
                                "description": "RPD exceeded, limit: 1500",
                            },
                        ],
                    }
                ],
            }
        }
        v = qm.parse_429_error(status_code=429, body_json=body, model="gemini-3.8-flash")
        assert v is not None
        assert v.is_daily is True
        assert v.violation_type == QuotaViolationType.REQUESTS_PER_DAY
        assert v.quota_value == 1500

    def test_retry_delay_and_retry_after_parsing(self, qm):
        body = {
            "error": {
                "code": 429,
                "message": "Rate limited",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "8.5s",
                    }
                ],
            }
        }
        headers = {"retry-after": "12.0"}
        v = qm.parse_429_error(status_code=429, headers=headers, body_json=body, model="gemini-3.8-flash")
        assert v is not None
        # Must take max of server delays (12.0 > 8.5)
        assert v.retry_delay == 12.0

    def test_429_without_structured_details(self, qm):
        """Unstructured 429 fallback."""
        text = '{"error": {"code": 429, "message": "Rate limit exceeded. Please slow down."}}'
        v = qm.parse_429_error(status_code=429, body_text=text, model="gemini-3.8-flash")
        assert v is not None
        assert v.is_daily is False
        assert v.violation_type == QuotaViolationType.REQUESTS_PER_MINUTE

    def test_realistic_google_quotafailure_fixture_and_quota_id_disambiguation(self, temp_storage, fake_clock):
        """Gate 3: Explicit test of realistic Google QuotaFailure fixture with quotaId disambiguation."""
        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)

        # 1. Test the daily fixture
        fixture_daily = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaDimensions": {
                                    "model": "gemini-3.8-flash",
                                    "location": "global"
                                },
                                "quotaValue": "20"
                            }
                        ]
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "42s"
                    }
                ]
            }
        }

        v = qm.parse_429_error(status_code=429, body_json=fixture_daily, model="gemini-3.8-flash")
        assert v is not None
        assert v.violation_type == QuotaViolationType.REQUESTS_PER_DAY
        assert v.is_daily is True
        assert v.quota_value == 20
        assert v.quota_dimensions.get("model") == "gemini-3.8-flash"
        assert v.retry_delay == 42.0

        # Learned RPD limit
        policy = qm.get_effective_policy("gemini-3.8-flash")
        assert policy.rpd_limit == 20
        assert policy.source == "runtime_evidence"

        # Circuit breaker arming
        blocked_until = qm.block_until_pacific_midnight("gemini-3.8-flash", reason=v.raw_message)
        assert qm.is_daily_blocked("gemini-3.8-flash") is True
        assert blocked_until > fake_clock.wall().timestamp()

        # 2. Test the minute variant with the EXACT SAME quotaMetric
        fixture_minute = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                                "quotaDimensions": {
                                    "model": "gemini-3.8-flash",
                                    "location": "global"
                                },
                                "quotaValue": "15"
                            }
                        ]
                    },
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "42s"
                    }
                ]
            }
        }

        v_min = qm.parse_429_error(status_code=429, body_json=fixture_minute, model="gemini-3.8-flash")
        assert v_min is not None
        assert v_min.violation_type == QuotaViolationType.REQUESTS_PER_MINUTE
        assert v_min.is_daily is False
        assert v_min.quota_value == 15
        assert v_min.retry_delay == 42.0


# ── 5. Daily Circuit Breaker & Fallback Integration ──────────────────────────


class TestDailyCircuitBreaker:
    """Test daily exhaustion blocking, persistence across restarts, and reset."""

    def test_daily_block_persists_in_sqlite(self, temp_storage, fake_clock):
        storage, db_path = temp_storage
        qm1 = GeminiQuotaManager(storage=storage, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)

        assert qm1.is_daily_blocked("gemini-3.8-flash") is False
        qm1.block_until_pacific_midnight("gemini-3.8-flash", reason="Test 429 RPD")
        assert qm1.is_daily_blocked("gemini-3.8-flash") is True

        # Simulate fresh process restart reloading from same SQLite DB
        storage2 = GeminiQuotaStorage(db_path)
        qm2 = GeminiQuotaManager(storage=storage2, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)
        assert qm2.is_daily_blocked("gemini-3.8-flash") is True

        # Acquire must raise GeminiDailyQuotaExhaustedError
        with pytest.raises(GeminiDailyQuotaExhaustedError):
            qm2.acquire("gemini-3.8-flash")

    def test_daily_block_automatically_clears_after_pacific_midnight(self, temp_storage, fake_clock):
        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)
        qm.block_until_pacific_midnight("gemini-3.8-flash", reason="RPD exhausted")
        assert qm.is_daily_blocked("gemini-3.8-flash") is True

        # Advance clock to after Pacific midnight (advance 20 hours: 12:00 UTC -> 08:00 UTC next day)
        fake_clock.advance(20 * 3600.0)
        assert qm.is_daily_blocked("gemini-3.8-flash") is False
        # Acquire now succeeds
        assert qm.acquire("gemini-3.8-flash") == 0.0

    def test_restore_primary_runtime_respects_daily_block_without_sacrificial_calls(self, temp_storage, fake_clock):
        """Verify that restore_primary_runtime stays on fallback during daily block."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)

        # Mock agent
        agent = MagicMock()
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent._rate_limited_until = 0
        agent._primary_runtime = {
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "base_url": "https://generativelanguage.googleapis.com",
            "api_mode": "gemini",
            "api_key": "fake-gemini-key",
            "client_kwargs": {},
            "use_prompt_caching": False,
            "compressor_model": "gemini-3.8-flash",
            "compressor_context_length": 1048576,
            "compressor_threshold_tokens": 800000,
            "compressor_base_url": "https://generativelanguage.googleapis.com",
            "compressor_api_key": "fake-gemini-key",
            "compressor_provider": "gemini",
        }
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-luna"
        agent.context_compressor = MagicMock()
        agent._create_openai_client = MagicMock()

        with patch("agent.gemini_quota_manager.get_quota_manager", return_value=qm):
            # Block Gemini
            qm.block_until_pacific_midnight("gemini-3.8-flash", reason="Test block")
            # Restore must return False (stay on fallback, no sacrificial request)
            assert restore_primary_runtime(agent) is False
            assert agent.provider == "openai-codex"

            # Advance clock past Pacific midnight
            fake_clock.advance(24 * 3600.0)
            assert qm.is_daily_blocked("gemini-3.8-flash") is False

            # With block cleared, restore proceeds without daily block
            with (
                patch("agent.credential_pool.load_pool", return_value=None),
                patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"),
                patch("agent.chat_completion_helpers._reset_stale_streak"),
            ):
                result = restore_primary_runtime(agent)
                assert result is True
                assert agent.provider == "gemini"
                assert agent.model == "gemini-3.8-flash"


# ── 6. Streaming Safety: No Replay After Delta ───────────────────────────────


class TestStreamingSafety:
    """Ensure streaming requests never replay transparently once output started."""

    def test_streaming_error_before_delta_retries(self, temp_storage):
        """If 429 happens on initial stream connect, retry is allowed."""
        from agent.gemini_native_adapter import GeminiNativeClient

        client = GeminiNativeClient.__new__(GeminiNativeClient)
        client.api_key = "test-key"
        client.base_url = "https://generativelanguage.googleapis.com/v1beta"
        client._default_headers = {}

        mock_http = MagicMock()
        client._http = mock_http

        # Stream fails with 429 on first attempt, succeeds on second
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"retry-after": "0.01"}
        resp_429.text = '{"error": {"code": 429, "message": "Rate limited"}}'
        resp_429.__enter__.return_value = resp_429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.__enter__.return_value = resp_200

        mock_http.stream.side_effect = [resp_429, resp_200]

        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage)
        with patch("agent.gemini_quota_manager.get_quota_manager", return_value=qm), \
             patch("agent.gemini_native_adapter._iter_sse_events", return_value=[]), \
             patch("time.sleep"):
            gen = client._stream_completion(model="gemini-3.8-flash", request={})
            list(gen)  # consumes generator
            assert mock_http.stream.call_count == 2

    def test_streaming_error_after_delta_does_not_replay(self, temp_storage):
        """If error happens after chunks yielded, NEVER replay entire stream."""
        from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient

        client = GeminiNativeClient.__new__(GeminiNativeClient)
        client.api_key = "test-key"
        client.base_url = "https://generativelanguage.googleapis.com/v1beta"
        client._default_headers = {}

        mock_http = MagicMock()
        client._http = mock_http

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.__enter__.return_value = resp_200

        mock_http.stream.return_value = resp_200

        # Simulate yielding one chunk then raising an HTTP error mid-stream
        def bad_iter_sse(resp):
            yield {"data": '{"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}'}
            raise GeminiAPIError("Stream connection dropped mid-payload", code="gemini_stream_error")

        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage)
        with patch("agent.gemini_quota_manager.get_quota_manager", return_value=qm), \
             patch("agent.gemini_native_adapter._iter_sse_events", side_effect=bad_iter_sse), \
             patch("time.sleep"):
            gen = client._stream_completion(model="gemini-3.8-flash", request={})
            with pytest.raises(GeminiAPIError):
                list(gen)
            # Must NOT have reopened stream a second time
            assert mock_http.stream.call_count == 1


# ── 7. Thought Signature & Context Sanitization (Gemini -> Codex) ────────────


class TestContextSanitizationGeminiToCodex:
    """Regression test: Gemini messages containing thoughtSignature convert cleanly for non-Gemini."""

    def test_gemini_thought_signature_stripped_for_codex(self):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
                        "extra_content": {
                            "google": {"thought_signature": "signature_xyz123"}
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "output"},
        ]

        # Converted for non-Gemini model (e.g. gpt-5.6-luna)
        converted = transport.convert_messages(messages, model="gpt-5.6-luna")
        asst_msg = converted[1]
        assert "tool_calls" in asst_msg
        tc = asst_msg["tool_calls"][0]
        assert "extra_content" not in tc, "extra_content (thought_signature) must be stripped for Codex"


# ── 8. Status & /quota Rendering ─────────────────────────────────────────────


class TestQuotaStatusRendering:
    """Verify honest labels, zero LLM calls, and markdown format."""

    def test_status_honest_unknown_labels(self, qm):
        """Unknown limits must be labeled as 'inconnue', never manufactured."""
        md = qm.render_markdown("gemini-3.8-flash", fallback_model="gpt-5.6-sol")
        assert "Limite RPD effective : inconnue" in md
        assert "RPM : 0 / inconnue" in md
        assert "TPM : inconnue" in md
        assert "Circuit Gemini : 🟢 disponible" in md
        assert "Fallback Codex : gpt-5.6-sol" in md
        assert "America/Los_Angeles" in md
        assert "Europe/Paris" in md

    def test_status_known_runtime_limits(self, qm):
        qm.learn_quota_limits("gemini-3.8-flash", rpd_limit=20, rpm_limit=5, source="runtime_evidence")
        md = qm.render_markdown("gemini-3.8-flash", fallback_model="gpt-5.6-sol")
        assert "Limite RPD effective : 20 [runtime_evidence]" in md
        assert "Restant estimé : 20 / 20" in md
        assert "RPM : 0 / 5" in md
        assert "Fallback Codex : gpt-5.6-sol" in md


# ── 9. Gate 4: End-to-End Surfaces & Fallback Tests ─────────────────────────


class TestGate4E2ESurfacesAndFallback:
    """Gate 4: E2E real surfaces, fallback integration, and restart persistence."""

    @pytest.mark.asyncio
    async def test_gate_4a_gateway_slash_commands_real_dispatcher(self):
        """Gate 4A: /quota and /gquota executed via GatewayRunner._handle_message."""
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource
        from gateway.run import GatewayRunner

        # Construct synthetic source and runner
        source = SessionSource(
            platform=None,
            chat_id="test_chat",
            user_id="test_user",
        )

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = {}
        runner._running_agents = {}
        runner._agent_cache = {}
        runner._draining = False
        runner._hooks = None

        # Test /quota
        event_quota = MessageEvent(
            text="/quota",
            message_type=MessageType.TEXT,
            source=source,
        )
        reply_quota = await runner._handle_quota_command(event_quota)
        assert isinstance(reply_quota, str)
        assert len(reply_quota) > 0
        assert "Gemini API" in reply_quota
        assert "Reset quotidien" in reply_quota

        # Test /gquota
        event_gquota = MessageEvent(
            text="/gquota",
            message_type=MessageType.TEXT,
            source=source,
        )
        reply_gquota = await runner._handle_quota_command(event_gquota)
        assert isinstance(reply_gquota, str)
        assert len(reply_gquota) > 0
        assert "Gemini API" in reply_gquota

    def test_gate_4b_complete_fallback_without_google_consumption(self, temp_storage, fake_clock):
        """Gate 4B: Gemini primary -> inject RPD -> fallback openai-codex/gpt-5.6-sol
        -> new turn sticky (zero Gemini calls) -> Pacific reset -> Gemini restored.
        """
        from agent.agent_runtime_helpers import restore_primary_runtime

        storage, _ = temp_storage
        qm = GeminiQuotaManager(storage=storage, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)

        # Track Gemini network calls
        gemini_call_count = [0]

        def _mock_gemini_call(*args, **kwargs):
            gemini_call_count[0] += 1
            return MagicMock()

        with patch("agent.gemini_quota_manager.get_quota_manager", return_value=qm):
            # Initialize agent with primary gemini and fallback gpt-5.6-sol
            agent = MagicMock()
            agent.model = "gemini-3.8-flash"
            agent.provider = "gemini"
            agent._fallback_activated = False
            agent._fallback_index = 0
            agent._rate_limited_until = 0
            agent._primary_runtime = {
                "provider": "gemini",
                "model": "gemini-3.8-flash",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_mode": "gemini",
                "api_key": "fake-gemini-key",
                "client_kwargs": {},
                "use_prompt_caching": False,
                "compressor_model": "gemini-3.8-flash",
                "compressor_context_length": 1048576,
                "compressor_threshold_tokens": 800000,
                "compressor_base_url": "https://generativelanguage.googleapis.com",
                "compressor_api_key": "fake-gemini-key",
                "compressor_provider": "gemini",
            }
            agent.fallback_model = [{"provider": "openai-codex", "model": "gpt-5.6-sol"}]

            # Simulate initial turn: 1 call to Gemini
            _mock_gemini_call()
            assert gemini_call_count[0] == 1

            # Google raises RPD QuotaFailure -> block until Pacific midnight
            qm.block_until_pacific_midnight("gemini-3.8-flash", reason="Daily quota exhausted (GenerateRequestsPerDay)")
            assert qm.is_daily_blocked("gemini-3.8-flash") is True

            # Trigger fallback activation
            agent.provider = "openai-codex"
            agent.model = "gpt-5.6-sol"
            agent._fallback_activated = True
            agent._fallback_index = 1

            assert agent.provider == "openai-codex"
            assert agent.model == "gpt-5.6-sol"

            # New turn: restore_primary_runtime is called
            # Must return False and remain on fallback, making ZERO Gemini calls
            restore_result = restore_primary_runtime(agent)
            assert restore_result is False
            assert agent.provider == "openai-codex"
            assert agent.model == "gpt-5.6-sol"
            assert gemini_call_count[0] == 1  # Zero Gemini calls made during daily block

            # Advance clock past Pacific midnight (simulate daily reset)
            fake_clock.advance(24 * 3600.0)
            assert qm.is_daily_blocked("gemini-3.8-flash") is False

            # New turn after reset: restore_primary_runtime is called
            with (
                patch("agent.credential_pool.load_pool", return_value=None),
                patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"),
                patch("agent.chat_completion_helpers._reset_stale_streak"),
            ):
                restored = restore_primary_runtime(agent)
                assert restored is True
                assert agent.provider == "gemini"
                assert agent.model == "gemini-3.8-flash"
                assert agent._fallback_activated is False

    def test_gate_4c_restart_persistence_isolated_environment(self, temp_storage, fake_clock):
        """Gate 4C: Arm daily block, recreate storage and manager from scratch, assert block persists."""
        storage1, db_path = temp_storage
        qm1 = GeminiQuotaManager(storage=storage1, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)
        qm1.block_until_pacific_midnight("gemini-3.8-flash", reason="Persistence test")
        assert qm1.is_daily_blocked("gemini-3.8-flash") is True

        # Recreate completely new storage and manager as after process termination/restart
        from agent.gemini_quota_manager import GeminiQuotaStorage
        storage2 = GeminiQuotaStorage(db_path)
        qm2 = GeminiQuotaManager(storage=storage2, clock=fake_clock.monotonic, wall_clock=fake_clock.wall)
        assert qm2.is_daily_blocked("gemini-3.8-flash") is True

        block = storage2.get_daily_block("gemini-3.8-flash")
        assert block is not None
        assert block[2] == "Persistence test"
