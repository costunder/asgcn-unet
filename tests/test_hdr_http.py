from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from scripts import get_hdr, hdr_http

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
PAYLOAD = HDF5_MAGIC + b"official-example-content" * 3
PUBLIC_URL = "https://public.files.1drv.com/y4m/example?authkey=synthetic-secret"


class Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = (
            headers
            if headers is not None
            else {
                "Content-Length": str(len(payload)),
                "Content-Type": "application/octet-stream",
            }
        )

    def getcode(self) -> int:
        return self.status

    def getheader(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name, default)


class FakeClient:
    def __init__(self, *responses: Callable[[], Response | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[dict[str, Any], int]] = []
        self.refreshes: list[dict[str, Any]] = []

    def open_file(self, item: dict[str, Any], offset: int) -> Response:
        index = min(len(self.calls), len(self.responses) - 1)
        self.calls.append((dict(item), offset))
        assert self.responses, "Existing or rejected local files must not be downloaded"
        result = self.responses[index]()
        if isinstance(result, Exception):
            raise result
        return result

    def refresh(self, item: dict[str, Any]) -> dict[str, Any]:
        self.refreshes.append(dict(item))
        return {**item, "url": PUBLIC_URL + "&fresh=1"}


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hdr_http.time, "sleep", lambda _delay: None)


def _item(payload: bytes = PAYLOAD) -> dict[str, Any]:
    return {
        "split": "train",
        "name": "1.h5",
        "id": "synthetic-item",
        "drive_id": "synthetic-drive",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "url": PUBLIC_URL,
    }


def _partial(target: Path, item: dict[str, Any], payload: bytes) -> tuple[Path, Path]:
    part = target.with_name(target.name + ".part")
    state = target.with_name(target.name + ".part.json")
    part.write_bytes(payload)
    state.write_text(
        json.dumps({"size": item["size"], "sha256": item["sha256"]}),
        encoding="utf-8",
    )
    return part, state


def _range_response(payload: bytes, start: int, total: int) -> Response:
    return Response(
        payload,
        status=206,
        headers={
            "Content-Length": str(len(payload)),
            "Content-Range": f"bytes {start}-{start + len(payload) - 1}/{total}",
            "Content-Type": "application/octet-stream",
        },
    )


@pytest.mark.parametrize("status", [200, 206])
def test_download_accepts_complete_verified_response(tmp_path: Path, status: int) -> None:
    target = tmp_path / "1.h5"
    client = FakeClient(
        lambda: Response(PAYLOAD) if status == 200 else _range_response(PAYLOAD, 0, len(PAYLOAD))
    )

    assert hdr_http.download_file(client, _item(), target) == "downloaded"
    assert target.read_bytes() == PAYLOAD
    assert [offset for _, offset in client.calls] == [0]
    assert not target.with_name("1.h5.part").exists()
    assert not target.with_name("1.h5.part.json").exists()


def test_download_keeps_verified_completed_file_without_network(tmp_path: Path) -> None:
    target = tmp_path / "1.h5"
    target.write_bytes(PAYLOAD)
    client = FakeClient()

    assert hdr_http.download_file(client, _item(), target) == "kept"
    assert target.read_bytes() == PAYLOAD
    assert client.calls == []


@pytest.mark.parametrize("existing", [b"short", PAYLOAD[:-1] + b"!", b"not-hdf5" + PAYLOAD[8:]])
def test_download_refuses_to_overwrite_mismatched_completed_file(
    tmp_path: Path, existing: bytes
) -> None:
    target = tmp_path / "1.h5"
    target.write_bytes(existing)
    client = FakeClient()

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert target.read_bytes() == existing
    assert client.calls == []


def test_download_resumes_identity_matched_partial(tmp_path: Path) -> None:
    target = tmp_path / "1.h5"
    prefix_length = 19
    part, state = _partial(target, _item(), PAYLOAD[:prefix_length])
    client = FakeClient(
        lambda: _range_response(PAYLOAD[prefix_length:], prefix_length, len(PAYLOAD))
    )

    assert hdr_http.download_file(client, _item(), target) == "downloaded"
    assert target.read_bytes() == PAYLOAD
    assert [offset for _, offset in client.calls] == [prefix_length]
    assert not part.exists()
    assert not state.exists()


def test_download_resumes_after_interrupted_response(tmp_path: Path) -> None:
    class InterruptedResponse(Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell() == 0:
                return super().read(19)
            raise OSError("connection interrupted")

    target = tmp_path / "1.h5"
    client = FakeClient(
        lambda: InterruptedResponse(PAYLOAD),
        lambda: _range_response(PAYLOAD[19:], 19, len(PAYLOAD)),
    )

    assert hdr_http.download_file(client, _item(), target, retries=2) == "downloaded"
    assert target.read_bytes() == PAYLOAD
    assert [offset for _, offset in client.calls] == [0, 19]


@pytest.mark.parametrize("identity_problem", ["missing", "size", "sha256", "invalid-json"])
def test_download_refuses_partial_with_missing_or_wrong_identity(
    tmp_path: Path, identity_problem: str
) -> None:
    target = tmp_path / "1.h5"
    part, state = _partial(target, _item(), PAYLOAD[:19])
    if identity_problem == "missing":
        state.unlink()
    elif identity_problem == "invalid-json":
        state.write_text("{incomplete", encoding="utf-8")
    else:
        identity = {"size": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}
        identity[identity_problem] = 1 if identity_problem == "size" else "0" * 64
        state.write_text(json.dumps(identity), encoding="utf-8")
    client = FakeClient()

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert not target.exists()
    assert part.read_bytes() == PAYLOAD[:19]
    assert client.calls == []


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Content-Length": str(len(PAYLOAD) - 19)}, 200),
        (
            {
                "Content-Length": str(len(PAYLOAD) - 19),
                "Content-Range": f"bytes 18-{len(PAYLOAD) - 2}/{len(PAYLOAD)}",
            },
            206,
        ),
        (
            {
                "Content-Length": str(len(PAYLOAD) - 19),
                "Content-Range": f"bytes 19-{len(PAYLOAD) - 1}/{len(PAYLOAD) + 1}",
            },
            206,
        ),
        ({"Content-Length": str(len(PAYLOAD) - 19)}, 206),
        (
            {
                "Content-Length": "1",
                "Content-Range": f"bytes 19-{len(PAYLOAD) - 1}/{len(PAYLOAD)}",
            },
            206,
        ),
    ],
)
def test_download_rejects_unsafe_resume_response(
    tmp_path: Path, headers: dict[str, str], status: int
) -> None:
    target = tmp_path / "1.h5"
    part, _ = _partial(target, _item(), PAYLOAD[:19])
    client = FakeClient(lambda: Response(PAYLOAD[19:], status=status, headers=headers))

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert not target.exists()
    assert part.read_bytes() == PAYLOAD[:19]


@pytest.mark.parametrize(
    "content_range",
    [
        f"bytes 1-{len(PAYLOAD)}/{len(PAYLOAD)}",
        f"bytes 0-{len(PAYLOAD) - 1}/{len(PAYLOAD) + 1}",
        f"bytes 0-{len(PAYLOAD) - 2}/{len(PAYLOAD)}",
        "bytes nonsense",
    ],
)
def test_download_rejects_bad_initial_content_range(tmp_path: Path, content_range: str) -> None:
    target = tmp_path / "1.h5"
    client = FakeClient(
        lambda: Response(
            PAYLOAD,
            status=206,
            headers={
                "Content-Length": str(len(PAYLOAD)),
                "Content-Range": content_range,
            },
        )
    )

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert not target.exists()


@pytest.mark.parametrize(
    "body,headers",
    [
        (PAYLOAD[:-1], {"Content-Length": str(len(PAYLOAD))}),
        (PAYLOAD + b"!", {"Content-Length": str(len(PAYLOAD))}),
        (PAYLOAD[:-1] + b"!", {"Content-Length": str(len(PAYLOAD))}),
        (PAYLOAD, {}),
        (PAYLOAD, {"Content-Length": "not-a-number"}),
    ],
)
def test_download_rejects_bad_size_hash_or_length_headers(
    tmp_path: Path, body: bytes, headers: dict[str, str]
) -> None:
    target = tmp_path / "1.h5"
    client = FakeClient(lambda: Response(body, headers=headers))

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert not target.exists()


def test_download_rejects_html_even_with_matching_size_and_hash(tmp_path: Path) -> None:
    target = tmp_path / "1.h5"
    html = b"<html>sign in to view this file</html>"
    client = FakeClient(
        lambda: Response(
            html,
            headers={"Content-Length": str(len(html)), "Content-Type": "text/html"},
        )
    )

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(html), target, retries=1)

    assert not target.exists()


@pytest.mark.parametrize("error_type", [URLError, OSError])
def test_download_retries_transient_transport_failure(
    tmp_path: Path, error_type: type[Exception]
) -> None:
    target = tmp_path / "1.h5"
    client = FakeClient(lambda: error_type("temporary outage"), lambda: Response(PAYLOAD))

    assert hdr_http.download_file(client, _item(), target, retries=2) == "downloaded"
    assert target.read_bytes() == PAYLOAD
    assert [offset for _, offset in client.calls] == [0, 0]


@pytest.mark.parametrize("status", [401, 403, 408, 429, 500, 503])
def test_download_retries_retryable_http_status(tmp_path: Path, status: int) -> None:
    target = tmp_path / "1.h5"
    client = FakeClient(
        lambda: HTTPError(PUBLIC_URL, status, "temporary failure", {}, None),
        lambda: Response(PAYLOAD),
    )

    assert hdr_http.download_file(client, _item(), target, retries=2) == "downloaded"
    assert target.read_bytes() == PAYLOAD
    assert len(client.calls) == 2


def test_download_failure_does_not_expose_url_token_or_private_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "1.h5"
    private_path = tmp_path / "private-environment" / "tls.pem"
    message = f"download {PUBLIC_URL} with Bearer synthetic-token failed at {private_path}"
    client = FakeClient(lambda: URLError(message))

    with pytest.raises(hdr_http.DownloadError) as error:
        hdr_http.download_file(client, _item(), target, retries=1)

    output = capsys.readouterr()
    public_error = str(error.value) + output.out + output.err
    for secret in (PUBLIC_URL, "synthetic-secret", "synthetic-token", str(private_path)):
        assert secret not in public_error
    assert not target.exists()


@pytest.mark.parametrize("symlink_name", ["1.h5", "1.h5.part", "1.h5.part.json"])
def test_download_refuses_symlinks_without_modifying_referent(
    tmp_path: Path, symlink_name: str
) -> None:
    target = tmp_path / "1.h5"
    referent = tmp_path / "keep-original"
    referent.write_bytes(b"original content")
    link = tmp_path / symlink_name
    try:
        link.symlink_to(referent)
    except OSError:
        if os.name == "nt":
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    client = FakeClient()

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_file(client, _item(), target, retries=1)

    assert link.is_symlink()
    assert referent.read_bytes() == b"original content"
    assert client.calls == []


@pytest.mark.parametrize("split", [None, "train", "eval"])
def test_download_cli_selects_and_checks_requested_official_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    split: str | None,
) -> None:
    destination = tmp_path / "EventHDR"
    calls: list[tuple[Path, dict[str, tuple[str, ...]]]] = []

    def fake_download(target: Path, expected: dict[str, tuple[str, ...]]) -> dict[str, int]:
        calls.append((target, expected))
        for split_name, names in expected.items():
            directory = target / split_name
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_bytes(PAYLOAD)
        return {"downloaded": sum(len(names) for names in expected.values()), "kept": 0}

    monkeypatch.setattr(hdr_http, "download_dataset", fake_download)
    arguments = ["--download", "--destination", str(destination)]
    if split is not None:
        arguments.extend(["--split", split])

    assert get_hdr.main(arguments) == 0

    expected = get_hdr.EXPECTED if split is None else {split: get_hdr.EXPECTED[split]}
    assert calls == [(destination.resolve(), expected)]
    for split_name, names in expected.items():
        assert {path.name for path in (destination / split_name).iterdir()} == set(names)
    output = capsys.readouterr()
    assert "EventHDR download passed" in output.out
    assert str(tmp_path) not in output.out + output.err


def test_download_cli_returns_failure_without_raw_filesystem_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_path = tmp_path / "private-cache" / "token.txt"

    def fail_download(*_args: Any) -> dict[str, int]:
        raise OSError(f"cannot read {private_path} with synthetic-secret")

    monkeypatch.setattr(hdr_http, "download_dataset", fail_download)

    assert get_hdr.main(["--download", "--destination", str(tmp_path / "EventHDR")]) == 1
    output = capsys.readouterr()
    assert "ERROR:" in output.err
    assert str(private_path) not in output.err
    assert "synthetic-secret" not in output.err
    assert "passed" not in output.out


def _metadata(name: str = "1.h5", **updates: Any) -> dict[str, Any]:
    value = {
        "name": name,
        "id": "synthetic-item",
        "parentReference": {"driveId": "synthetic-drive"},
        "size": len(PAYLOAD),
        "file": {"hashes": {"sha256Hash": hashlib.sha256(PAYLOAD).hexdigest().upper()}},
        "@content.downloadUrl": PUBLIC_URL,
    }
    value.update(updates)
    return value


def _inventory_client(
    monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, Any]]
) -> hdr_http.OneDrive:
    client = hdr_http.OneDrive(retries=1)

    def fake_json(url: str, **_kwargs: Any) -> dict[str, Any]:
        if "/shares/" in url:
            return {
                "value": [
                    {
                        "name": "train",
                        "folder": {},
                        "id": "train-folder",
                        "parentReference": {"driveId": "synthetic-drive"},
                    }
                ]
            }
        assert url == f"{hdr_http.API}/drives/synthetic-drive/items/train-folder/children"
        return {"value": entries}

    monkeypatch.setattr(client, "_json", fake_json)
    return client


def test_inventory_normalizes_official_file_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _inventory_client(monkeypatch, [_metadata()])

    assert client.inventory({"train": ("1.h5",)}) == [_item()]


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [_metadata(), _metadata("2.h5")],
        [_metadata(), _metadata()],
        [_metadata("other.h5")],
        [_metadata("../1.h5")],
        [_metadata("1.hdf5")],
    ],
)
def test_inventory_rejects_missing_extra_duplicate_or_renamed_files(
    monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, Any]]
) -> None:
    client = _inventory_client(monkeypatch, entries)

    with pytest.raises(hdr_http.DownloadError):
        client.inventory({"train": ("1.h5",)})


@pytest.mark.parametrize("size", [0, -1, True, "100", 100_000_000_000])
def test_inventory_rejects_invalid_file_sizes(monkeypatch: pytest.MonkeyPatch, size: Any) -> None:
    client = _inventory_client(monkeypatch, [_metadata(size=size)])

    with pytest.raises(hdr_http.DownloadError):
        client.inventory({"train": ("1.h5",)})


def test_inventory_rejects_total_size_at_100gb(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_metadata("1.h5", size=50_000_000_000), _metadata("2.h5", size=50_000_000_000)]
    client = _inventory_client(monkeypatch, entries)

    with pytest.raises(hdr_http.DownloadError):
        client.inventory({"train": ("1.h5", "2.h5")})


@pytest.mark.parametrize("sha256", [None, "", "a" * 63, "g" * 64, "a" * 65])
def test_inventory_rejects_missing_or_invalid_sha256(
    monkeypatch: pytest.MonkeyPatch, sha256: Any
) -> None:
    client = _inventory_client(monkeypatch, [_metadata(file={"hashes": {"sha256Hash": sha256}})])

    with pytest.raises(hdr_http.DownloadError):
        client.inventory({"train": ("1.h5",)})


@pytest.mark.parametrize(
    "url",
    [
        "http://public.files.1drv.com/content",
        "https://1drv.com.attacker.example/content?token=synthetic-secret",
        "https://attacker.example/content",
        "https://fixture-login:synthetic-secret@public.files.1drv.com/content",
        "https://public.files.1drv.com:8443/content",
        "file:///private-environment/token.txt",
        "https://[invalid-ipv6]/content?token=synthetic-secret",
        "https://[public.files.1drv.com/content?token=synthetic-secret",
    ],
)
def test_inventory_rejects_unapproved_download_urls_without_leaking_them(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    client = _inventory_client(monkeypatch, [_metadata(**{"@content.downloadUrl": url})])

    with pytest.raises(hdr_http.DownloadError) as error:
        client.inventory({"train": ("1.h5",)})

    assert url not in str(error.value)
    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://public.files.1drv.com/content",
        "https://my.microsoftpersonalcontent.com/content",
        "https://example.sharepoint.com/content",
    ],
)
def test_inventory_accepts_approved_https_storage_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    client = _inventory_client(monkeypatch, [_metadata(**{"@content.downloadUrl": url})])

    assert client.inventory({"train": ("1.h5",)})[0]["url"] == url


def test_metadata_pagination_collects_all_same_api_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    client = hdr_http.OneDrive(retries=1)
    start = f"{hdr_http.API}/drives/example/items/folder/children"
    next_page = start + "?skiptoken=synthetic-page-token"
    calls: list[str] = []

    def fake_json(url: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        if url == start:
            return {"value": [_metadata("1.h5")], "@odata.nextLink": next_page}
        assert url == next_page
        return {"value": [_metadata("2.h5")]}

    monkeypatch.setattr(client, "_json", fake_json)

    assert [value["name"] for value in client._children(start)] == ["1.h5", "2.h5"]
    assert calls == [start, next_page]


@pytest.mark.parametrize(
    "next_page",
    [
        "https://attacker.example/page?token=synthetic-secret",
        "https://public.files.1drv.com/_api/v2.0/items",
        "https://my.microsoftpersonalcontent.com/not-the-api/items",
        "https://my.microsoftpersonalcontent.com/_api/v2.0evil/items",
        123,
    ],
)
def test_metadata_pagination_rejects_foreign_or_invalid_links_before_request(
    monkeypatch: pytest.MonkeyPatch, next_page: Any
) -> None:
    client = hdr_http.OneDrive(retries=1)
    start = f"{hdr_http.API}/drives/example/items/folder/children"
    calls: list[str] = []

    def fake_json(url: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        return {"value": [], "@odata.nextLink": next_page}

    monkeypatch.setattr(client, "_json", fake_json)

    with pytest.raises(hdr_http.DownloadError) as error:
        client._children(start)

    assert calls == [start]
    assert "synthetic-secret" not in str(error.value)


def test_metadata_pagination_rejects_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    client = hdr_http.OneDrive(retries=1)
    start = f"{hdr_http.API}/drives/example/items/folder/children"
    calls: list[str] = []

    def fake_json(url: str, **_kwargs: Any) -> dict[str, Any]:
        calls.append(url)
        return {"value": [], "@odata.nextLink": start}

    monkeypatch.setattr(client, "_json", fake_json)

    with pytest.raises(hdr_http.DownloadError):
        client._children(start)

    assert calls == [start]


class FakeOpener:
    def __init__(self, *results: bytes | Exception) -> None:
        self.results = results
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: int) -> Response:
        assert timeout > 0
        self.requests.append(request)
        result = self.results[len(self.requests) - 1]
        if isinstance(result, Exception):
            raise result
        return Response(result)


def test_metadata_obtains_anonymous_token_and_reuses_it() -> None:
    opener = FakeOpener(
        b'{"token":"synthetic-token"}',
        b'{"value":[]}',
        b'{"value":[]}',
        b'{"value":[]}',
    )
    client = hdr_http.OneDrive(opener=opener, retries=1)
    url = f"{hdr_http.API}/drives/example/items/folder/children"

    assert client._json(url) == {"value": []}
    assert client._json(url) == {"value": []}

    token_request, redemption_request, metadata_request, repeated_request = opener.requests
    assert token_request.full_url == hdr_http.TOKEN_URL
    assert token_request.get_method() == "POST"
    assert json.loads(token_request.data) == {"appId": hdr_http.APP_ID}
    assert token_request.get_header("Authorization") is None
    assert redemption_request.full_url == client._shared_url()
    assert redemption_request.get_header("Authorization") == "Badger synthetic-token"
    assert redemption_request.get_header("Prefer") == "autoredeem"
    assert metadata_request.full_url == repeated_request.full_url == url
    assert metadata_request.get_header("Authorization") == "Badger synthetic-token"
    assert repeated_request.get_header("Authorization") == "Badger synthetic-token"


def test_shared_root_request_redeems_new_token_without_duplicate_request() -> None:
    opener = FakeOpener(b'{"token":"synthetic-token"}', b'{"value":[]}')
    client = hdr_http.OneDrive(opener=opener, retries=1)

    assert client._json(client._shared_url()) == {"value": []}
    assert [request.full_url for request in opener.requests] == [
        hdr_http.TOKEN_URL,
        client._shared_url(),
    ]
    assert opener.requests[1].get_header("Authorization") == "Badger synthetic-token"
    assert opener.requests[1].get_header("Prefer") == "autoredeem"


@pytest.mark.parametrize("status", [401, 403])
def test_metadata_refreshes_expired_anonymous_token(status: int) -> None:
    url = f"{hdr_http.API}/drives/example/items/folder/children"
    opener = FakeOpener(
        b'{"token":"expired-token"}',
        b'{"value":[]}',
        HTTPError(url, status, "expired", {}, None),
        b'{"token":"fresh-token"}',
        b'{"value":[]}',
        b'{"value":[]}',
    )
    client = hdr_http.OneDrive(opener=opener, retries=2)

    assert client._json(url) == {"value": []}
    assert [request.full_url for request in opener.requests] == [
        hdr_http.TOKEN_URL,
        client._shared_url(),
        url,
        hdr_http.TOKEN_URL,
        client._shared_url(),
        url,
    ]
    for index in (1, 2):
        assert opener.requests[index].get_header("Authorization") == "Badger expired-token"
    for index in (4, 5):
        assert opener.requests[index].get_header("Authorization") == "Badger fresh-token"
    assert opener.requests[4].get_header("Prefer") == "autoredeem"


def test_file_request_refreshes_signed_url_and_sends_range_without_metadata_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = FakeOpener(PAYLOAD)
    client = hdr_http.OneDrive(opener=opener, retries=1)
    client._token = "synthetic-token"
    refreshes: list[dict[str, Any]] = []

    def refresh(item: dict[str, Any]) -> dict[str, Any]:
        refreshes.append(dict(item))
        return {**item, "url": PUBLIC_URL + "&fresh=1"}

    monkeypatch.setattr(client, "refresh", refresh)

    with client.open_file(_item(), 19) as response:
        assert response.read() == PAYLOAD

    assert refreshes == [_item()]
    request = opener.requests[0]
    assert request.full_url.endswith("&fresh=1")
    assert request.get_header("Range") == "bytes=19-"
    assert request.get_header("Authorization") is None


@pytest.mark.parametrize("changed_key", ["name", "id", "drive_id", "size", "sha256"])
def test_refresh_rejects_changed_file_revision(
    monkeypatch: pytest.MonkeyPatch, changed_key: str
) -> None:
    client = hdr_http.OneDrive(retries=1)
    current = _metadata()
    if changed_key == "drive_id":
        current["parentReference"] = {"driveId": "changed-drive"}
    elif changed_key == "sha256":
        current["file"] = {"hashes": {"sha256Hash": "0" * 64}}
    else:
        current[changed_key] = len(PAYLOAD) + 1 if changed_key == "size" else "changed"
    monkeypatch.setattr(client, "_json", lambda *_args, **_kwargs: current)

    with pytest.raises(hdr_http.DownloadError):
        client.refresh(_item())


@pytest.mark.parametrize("raw", [b"not-json", b"[]", b'{"error":"synthetic-secret"}'])
def test_metadata_invalid_response_fails_without_raw_details(raw: bytes) -> None:
    client = hdr_http.OneDrive(opener=FakeOpener(raw), retries=1)

    with pytest.raises(hdr_http.DownloadError) as error:
        client._json(hdr_http.TOKEN_URL, authenticated=False)

    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize("token", [None, "", "value\nsynthetic-token", "value\rsynthetic-token", 1])
def test_metadata_rejects_missing_or_unsafe_anonymous_token(token: Any) -> None:
    client = hdr_http.OneDrive(opener=FakeOpener(json.dumps({"token": token}).encode()), retries=1)

    with pytest.raises(hdr_http.DownloadError) as error:
        client._json(f"{hdr_http.API}/drives/example/items/folder/children")

    assert "synthetic-token" not in str(error.value)


@pytest.mark.parametrize("field", ["file", "parentReference", "hashes", "name"])
@pytest.mark.parametrize("invalid", [None, [], {"private-detail": "synthetic-secret"}])
def test_inventory_rejects_malformed_metadata_fields_safely(
    monkeypatch: pytest.MonkeyPatch, field: str, invalid: Any
) -> None:
    updates = {field: invalid} if field != "hashes" else {"file": {"hashes": invalid}}
    client = _inventory_client(monkeypatch, [_metadata(**updates)])

    with pytest.raises(hdr_http.DownloadError) as error:
        client.inventory({"train": ("1.h5",)})

    assert "synthetic-secret" not in str(error.value)


@pytest.mark.parametrize("value", [None, [], "synthetic-secret"])
def test_file_metadata_must_be_an_object(value: Any) -> None:
    with pytest.raises(hdr_http.DownloadError) as error:
        hdr_http.OneDrive._file(value, "train")

    assert "synthetic-secret" not in str(error.value)


class DatasetClient(FakeClient):
    def inventory(self, expected: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
        return [_item()]


@pytest.mark.parametrize("extra_name", ["extra.hdf5", "extra.H5", "nested/1.h5"])
def test_dataset_rejects_all_unexpected_hdf5_extensions_before_transfer(
    tmp_path: Path, extra_name: str
) -> None:
    destination = tmp_path / "EventHDR"
    extra = destination / "train" / extra_name
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"existing content")
    client = DatasetClient()

    with pytest.raises(hdr_http.DownloadError):
        hdr_http.download_dataset(destination, {"train": ("1.h5",)}, client=client)

    assert extra.read_bytes() == b"existing content"
    assert not (destination / "train" / "1.h5").exists()
    assert client.calls == []


@pytest.mark.parametrize("failure_type", [get_hdr.ImportError, OSError])
def test_download_cli_postcheck_failure_suppresses_private_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_type: type[Exception],
) -> None:
    private_path = tmp_path / "private-data-mount" / "damaged.h5"
    monkeypatch.setattr(hdr_http, "download_dataset", lambda *_args: {"downloaded": 70, "kept": 0})

    def fail_postcheck(*_args: Any) -> None:
        raise failure_type(f"invalid file {private_path} on private-compute-node")

    monkeypatch.setattr(get_hdr, "check_destination", fail_postcheck)

    assert get_hdr.main(["--download", "--destination", str(tmp_path / "EventHDR")]) == 1

    output = capsys.readouterr()
    assert "ERROR:" in output.err
    assert "passed" not in output.out
    assert str(private_path) not in output.out + output.err
    assert "private-compute-node" not in output.out + output.err


def test_download_cli_rejects_lexical_symlink_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    referent = tmp_path / "preserve-original"
    referent.mkdir()
    original = referent / "keep.txt"
    original.write_bytes(b"existing content")
    destination = tmp_path / "EventHDR"
    try:
        destination.symlink_to(referent, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    client = DatasetClient()
    monkeypatch.setattr(hdr_http, "OneDrive", lambda: client)

    assert get_hdr.main(["--download", "--destination", str(destination)]) == 1

    output = capsys.readouterr()
    assert "symlink" in output.err.lower()
    assert "passed" not in output.out
    assert destination.is_symlink()
    assert list(referent.iterdir()) == [original]
    assert original.read_bytes() == b"existing content"
    assert client.calls == []
