"""Bounded, opt-in HTTPS capture and a separate resumable request cache."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.economic_event_sources import safe_file
from ashare_quant.utils.manifest import atomic_write_json

HOSTS = frozenset({"www.cninfo.com.cn", "static.cninfo.com.cn"})
REDIRECTS = {301, 302, 303, 307, 308}


class AnnouncementError(ValueError):
    """Stable acquisition error code; never include remote credentials or bodies."""


def check(condition: bool, code: str) -> None:
    if not condition:
        raise AnnouncementError(code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def official_url(url: str) -> str:
    try:
        p = urlsplit(url)
        valid = (
            p.scheme == "https"
            and p.hostname in HOSTS
            and p.port in (None, 443)
            and not p.username
            and not p.password
            and not p.fragment
            and "\\" not in url
            and not any(ord(c) < 33 for c in url)
        )
    except ValueError:
        valid = False
    check(valid, "ANNOUNCEMENT_UNSAFE_URL")
    return url


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    connect_timeout: float = Field(default=10, gt=0, le=60)
    read_timeout: float = Field(default=30, gt=0, le=120)
    retries: int = Field(default=2, ge=0, le=3)
    interval: float = Field(default=1, ge=0.1, le=60)
    backoff: float = Field(default=1, ge=0, le=10)
    max_redirects: int = Field(default=3, ge=0, le=5)
    max_requests: int = Field(default=200, ge=1, le=1000)
    response_bytes: int = Field(default=4_000_000, ge=128, le=20_000_000)
    pdf_bytes: int = Field(default=50_000_000, ge=128, le=100_000_000)


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes


class Transport(Protocol):
    def exchange(
        self, method: str, url: str, form: dict[str, str], limit: int, policy: Limits
    ) -> Reply: ...


class PublicHTTPSConnection(http.client.HTTPSConnection):
    """Resolve once and connect to a checked public IP; TLS still verifies hostname."""

    def connect(self) -> None:
        addresses = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
        check(bool(addresses), "ANNOUNCEMENT_DNS_EMPTY")
        check(
            all(ipaddress.ip_address(a[4][0]).is_global for a in addresses),
            "ANNOUNCEMENT_NON_PUBLIC_ADDRESS",
        )
        family, socktype, proto, _, address = addresses[0]
        sock = socket.socket(family, socktype, proto)
        try:
            sock.settimeout(self.timeout)
            sock.connect(address)
            self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class HTTPSWire:
    def __init__(self, *, authorized: bool = False) -> None:
        self.authorized = authorized

    def exchange(
        self, method: str, url: str, form: dict[str, str], limit: int, policy: Limits
    ) -> Reply:
        check(self.authorized, "ANNOUNCEMENT_NETWORK_NOT_AUTHORIZED")
        p = urlsplit(official_url(url))
        connection = PublicHTTPSConnection(str(p.hostname), timeout=policy.connect_timeout)
        try:
            connection.connect()
            assert connection.sock is not None
            connection.sock.settimeout(policy.read_timeout)
            target = p.path or "/"
            if p.query:
                target += "?" + p.query
            connection.request(
                method,
                target,
                body=urlencode(form).encode() if method == "POST" else None,
                headers={
                    "User-Agent": "ashare-quant-announcement/1",
                    "Accept": "application/json, application/pdf",
                    "Accept-Encoding": "identity",
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                },
            )
            response = connection.getresponse()
            headers = {k.lower(): v for k, v in response.getheaders()}
            check(headers.get("content-encoding", "identity") == "identity", "ENCODING_UNSUPPORTED")
            if "content-length" in headers:
                check(0 <= int(headers["content-length"]) <= limit, "RESPONSE_TOO_LARGE")
            data = bytearray()
            deadline = time.monotonic() + policy.read_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65536, limit + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                check(len(data) <= limit, "RESPONSE_TOO_LARGE")
            if "content-length" in headers:
                check(len(data) == int(headers["content-length"]), "RESPONSE_TRUNCATED")
            return Reply(response.status, headers, bytes(data))
        finally:
            connection.close()


def request_spec(url: str, form: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "url": official_url(url),
        "method": "POST" if form is not None else "GET",
        "form": form or {},
    }


def outcome(attempt: dict[str, Any]) -> str:
    if attempt["error"]:
        return str(attempt["error"])
    code = attempt["http_status"]
    if code == 200:
        return "CAPTURED"
    if code in (401, 403):
        return "PERMISSION_REQUIRED"
    if code == 429:
        return "RATE_LIMITED"
    if code >= 500:
        return "SERVER_REJECTED"
    return "HTTP_ERROR"


def validate_capture(capture: dict[str, Any], blobs: dict[str, bytes], policy: Limits) -> None:
    spec = capture["request"]
    check(
        spec == request_spec(spec["url"], spec["form"] if spec["method"] == "POST" else None),
        "REQUEST_SCHEMA",
    )
    check(capture["request_id"] == payload_hash(spec), "REQUEST_ID")
    attempts = capture["attempts"]
    check(0 < len(attempts) <= (policy.retries + 1) * (policy.max_redirects + 1), "ATTEMPT_LIMIT")
    expected_url = spec["url"]
    retries, redirects = 0, 0
    for i, attempt in enumerate(attempts):
        check(attempt["url"] == expected_url, "REDIRECT_CHAIN")
        official_url(attempt["url"])
        check(
            attempt["error"]
            in (
                None,
                "TIMEOUT",
                "NETWORK_ERROR",
                "RESPONSE_TOO_LARGE",
                "RESPONSE_TRUNCATED",
                "ENCODING_UNSUPPORTED",
                "ANNOUNCEMENT_NON_PUBLIC_ADDRESS",
                "ANNOUNCEMENT_DNS_EMPTY",
                "REQUEST_LIMIT",
                "ANNOUNCEMENT_UNSAFE_URL",
            ),
            "CAPTURE_ERROR",
        )
        if attempt["body_sha256"] is not None:
            check(
                type(attempt["http_status"]) is int
                and 100 <= attempt["http_status"] <= 599
                and attempt["error"] is None,
                "HTTP_STATUS_SCHEMA",
            )
            body = blobs[attempt["body_sha256"]]
            check(
                digest(body) == attempt["body_sha256"] and len(body) == attempt["size"],
                "RESPONSE_HASH",
            )
            cap = policy.pdf_bytes if spec["method"] == "GET" else policy.response_bytes
            check(len(body) <= cap, "RESPONSE_TOO_LARGE")
        else:
            check(
                attempt["error"] is not None and attempt["http_status"] is None, "MISSING_RESPONSE"
            )
        if i + 1 < len(attempts):
            if attempt["http_status"] in REDIRECTS:
                redirects += 1
                check(redirects <= policy.max_redirects, "REDIRECT_LIMIT")
                check(
                    spec["method"] == "GET" or attempt["http_status"] in (307, 308),
                    "POST_REDIRECT_REWRITE",
                )
                expected_url = official_url(urljoin(attempt["url"], attempt["location"]))
            else:
                retries += 1
                redirects = 0
                check(retries <= policy.retries, "RETRY_LIMIT")
                check(
                    attempt["error"] in ("TIMEOUT", "NETWORK_ERROR")
                    or attempt["http_status"] in (500, 502, 504),
                    "RETRY_NOT_ALLOWED",
                )
                expected_url = spec["url"]
    check(
        capture["status"] in (outcome(attempts[-1]), "IN_PROGRESS", "REDIRECT_REJECTED"),
        "CAPTURE_STATUS",
    )
    if capture["status"] == "REDIRECT_REJECTED":
        check(attempts[-1]["http_status"] in REDIRECTS, "REDIRECT_STATUS")


class CaptureCache:
    """Hash-checked request journal. Explicit retry preserves every earlier capture."""

    def __init__(self, root: Path, policy: Limits) -> None:
        self.root, self.policy = root, policy

    def register_worklist(self, body: dict[str, Any]) -> None:
        path = self.root / "worklists" / (payload_hash(body) + ".json")
        if path.exists():
            check(read_json(path) == body, "CACHE_WORKLIST_CORRUPT")
        else:
            atomic_write_json(path, body)

    def save(self, capture: dict[str, Any], blobs: dict[str, bytes]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for h, data in blobs.items():
            check(digest(data) == h, "CACHE_BODY_HASH")
            path = self.root / "blobs" / (h + ".bin")
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                check(file_hash(path) == h, "CACHE_CORRUPT")
            else:
                with path.open("xb") as output:
                    output.write(data)
        h = payload_hash(capture)
        target = self.root / "captures" / (h + ".json")
        if not target.exists():
            atomic_write_json(target, capture)
        pointer = self.root / "requests" / (capture["request_id"] + ".json")
        history = read_json(pointer)["history"] if pointer.exists() else []
        if h not in history:
            atomic_write_json(pointer, {"history": [*history, h]})

    def load(self, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]] | None:
        pointer = self.root / "requests" / (payload_hash(spec) + ".json")
        if not pointer.exists():
            return None
        history = read_json(pointer)["history"]
        check(bool(history), "CACHE_CORRUPT")
        latest: tuple[dict[str, Any], dict[str, bytes]] | None = None
        for h in history:
            capture = read_json(safe_file(self.root, f"captures/{h}.json"))
            check(payload_hash(capture) == h and capture["request"] == spec, "CACHE_CORRUPT")
            blobs = {
                a["body_sha256"]: safe_file(self.root, f"blobs/{a['body_sha256']}.bin").read_bytes()
                for a in capture["attempts"]
                if a["body_sha256"] is not None
            }
            validate_capture(capture, blobs, self.policy)
            latest = capture, blobs
        return latest


class SafeHTTPClient:
    def __init__(
        self,
        *,
        authorized: bool = False,
        policy: Limits | None = None,
        transport: Transport | None = None,
        cache: CaptureCache | None = None,
        retry_failed: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.authorized = authorized
        self.policy = policy or Limits()
        self.transport = transport or HTTPSWire(authorized=authorized)
        self.cache, self.retry_failed, self.sleep = cache, retry_failed, sleep
        self.requests = 0

    def fetch(self, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
        official_url(spec["url"])
        cached = self.cache.load(spec) if self.cache else None
        if cached and (cached[0]["status"] == "CAPTURED" or not self.retry_failed):
            return cached
        check(self.authorized, "ANNOUNCEMENT_NETWORK_NOT_AUTHORIZED")
        attempts: list[dict[str, Any]] = []
        blobs: dict[str, bytes] = {}
        capture = {
            "request": spec,
            "request_id": payload_hash(spec),
            "attempts": attempts,
            "status": "IN_PROGRESS",
        }
        for retry in range(self.policy.retries + 1):
            url = spec["url"]
            for hop in range(self.policy.max_redirects + 1):
                a: dict[str, Any] = {
                    "url": url,
                    "http_status": None,
                    "body_sha256": None,
                    "size": 0,
                    "location": None,
                    "error": None,
                }
                try:
                    check(self.requests < self.policy.max_requests, "REQUEST_LIMIT")
                    self.sleep(
                        self.policy.interval
                        + (self.policy.backoff * 2 ** (retry - 1) if retry else 0)
                    )
                    self.requests += 1
                    cap = (
                        self.policy.pdf_bytes
                        if spec["method"] == "GET"
                        else self.policy.response_bytes
                    )
                    reply = self.transport.exchange(
                        spec["method"], url, spec["form"], cap, self.policy
                    )
                    check(len(reply.body) <= cap, "RESPONSE_TOO_LARGE")
                    h = digest(reply.body)
                    blobs[h] = reply.body
                    a.update(
                        http_status=reply.status,
                        body_sha256=h,
                        size=len(reply.body),
                        location=reply.headers.get("location"),
                    )
                except TimeoutError:
                    a["error"] = "TIMEOUT"
                except (OSError, http.client.HTTPException):
                    a["error"] = "NETWORK_ERROR"
                except AnnouncementError as error:
                    a["error"] = str(error)
                attempts.append(a)
                if self.cache:
                    capture["status"] = "IN_PROGRESS"
                    self.cache.save(capture, blobs)
                if a["http_status"] in REDIRECTS:
                    try:
                        check(
                            hop < self.policy.max_redirects and bool(a["location"]),
                            "ANNOUNCEMENT_UNSAFE_URL",
                        )
                        # POST redirection semantics must not silently discard/rewrite a query.
                        check(
                            spec["method"] == "GET" or a["http_status"] in (307, 308),
                            "ANNOUNCEMENT_UNSAFE_URL",
                        )
                        url = official_url(urljoin(url, a["location"]))
                        continue
                    except AnnouncementError:
                        capture["status"] = "REDIRECT_REJECTED"
                        break
                capture["status"] = outcome(a)
                break
            retryable = attempts[-1]["error"] in ("TIMEOUT", "NETWORK_ERROR") or attempts[-1][
                "http_status"
            ] in (500, 502, 504)
            if not retryable:
                break
        validate_capture(capture, blobs, self.policy)
        if self.cache:
            self.cache.save(capture, blobs)
        return capture, blobs


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    check(len({key for key, _ in pairs}) == len(pairs), "JSON_DUPLICATE_KEY")
    return dict(pairs)


def _invalid_constant(value: str) -> None:
    raise AnnouncementError("JSON_NON_FINITE")


def response_json(capture: dict[str, Any], blobs: dict[str, bytes]) -> dict[str, Any] | list[Any]:
    check(capture["status"] == "CAPTURED", "RESPONSE_NOT_CAPTURED")
    try:
        parsed = json.loads(
            blobs[capture["attempts"][-1]["body_sha256"]],
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        check(isinstance(parsed, (dict, list)), "JSON_ROOT_SCHEMA")
        assert isinstance(parsed, (dict, list))
        return parsed
    except (ValueError, UnicodeError) as error:
        raise AnnouncementError("RESPONSE_NOT_JSON_OR_ACCESS_CHALLENGE") from error
