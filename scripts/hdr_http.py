"""Download the public EventHDR release directly, without a browser or user login.

OneDrive's anonymous Badger endpoint is an undocumented compatibility interface,
not a guaranteed Microsoft Graph contract. Tokens and signed URLs stay in memory.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

SHARE_URL = "https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N"
API = "https://my.microsoftpersonalcontent.com/_api/v2.0"
TOKEN_URL = "https://api-badgerp.svc.ms/v1.0/token"
# Public anonymous-client identifier, not a user credential or client secret.
APP_ID = "5cbed6ac-a083-4e14-b191-b4ba07653de2"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_BYTES = 100_000_000_000
MAX_JSON_BYTES = 4 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
RETRY_CODES = {401, 403, 408, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """A safe-to-publish error that excludes credentials and host paths."""


def _check_url(value: str, *, api: bool = False) -> str:
    if not isinstance(value, str):
        raise DownloadError("Missing HTTPS download address in official metadata")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise DownloadError("Invalid download address") from None
    host = (parsed.hostname or "").lower()
    approved = host == "api-badgerp.svc.ms" or any(
        host == domain or host.endswith("." + domain)
        for domain in ("microsoftpersonalcontent.com", "1drv.com", "sharepoint.com")
    )
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not approved
        or not value.isascii()
        or any(ord(character) <= 32 for character in value)
    ):
        raise DownloadError("Refusing a non-Microsoft or non-HTTPS download address")
    if api and (
        host != "my.microsoftpersonalcontent.com" or not parsed.path.startswith("/_api/v2.0/")
    ):
        raise DownloadError("Refusing a metadata link outside the official API")
    return value


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        if req.has_header("Authorization") and (
            urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc
        ):
            raise DownloadError("Refusing to forward authorization to a different host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRY_CODES
    return isinstance(error, (OSError, urllib.error.URLError, http.client.HTTPException))


def _delay(attempt: int) -> None:
    time.sleep(min(2**attempt, 16))


class OneDrive:
    def __init__(self, opener=None, *, timeout: int = 30, retries: int = 5) -> None:
        self.opener = opener or urllib.request.build_opener(_SafeRedirect())
        self.timeout = timeout
        self.retries = retries
        self._token: str | None = None

    def _json(self, url: str, *, authenticated: bool = True, data: bytes | None = None) -> dict:
        _check_url(url, api=authenticated)
        for attempt in range(self.retries):
            try:
                headers = {"Accept": "application/json", "User-Agent": "asgcn-unet/0.3"}
                if authenticated:
                    if self._token is None:
                        token = self._json(
                            TOKEN_URL,
                            authenticated=False,
                            data=json.dumps({"appId": APP_ID}).encode("utf-8"),
                        ).get("token")
                        if (
                            not isinstance(token, str)
                            or not token
                            or "\n" in token
                            or "\r" in token
                        ):
                            raise DownloadError("Anonymous OneDrive authorization was unavailable")
                        self._token = token
                        # Share access must be redeemed for every fresh anonymous
                        # token, including a replacement issued during a long run.
                        if url != self._shared_url():
                            self._json(self._shared_url())
                    headers.update(
                        {"Authorization": "Badger " + self._token, "Prefer": "autoredeem"}
                    )
                if data is not None:
                    headers["Content-Type"] = "application/json"
                request = urllib.request.Request(url, data=data, headers=headers)
                with self.opener.open(request, timeout=self.timeout) as response:
                    raw = response.read(MAX_JSON_BYTES + 1)
                if len(raw) > MAX_JSON_BYTES:
                    raise DownloadError("Official metadata response exceeded the safety limit")
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise DownloadError("Official metadata response was not valid JSON") from None
                if not isinstance(value, dict) or "error" in value:
                    raise DownloadError("Official metadata service returned an invalid response")
                return value
            except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403):
                    self._token = None
                if not _retryable(error) or attempt + 1 == self.retries:
                    raise DownloadError(
                        "Official OneDrive metadata request failed "
                        + (
                            f"(HTTP {error.code})"
                            if isinstance(error, urllib.error.HTTPError)
                            else f"({type(error).__name__})"
                        )
                        + "; retry later or check server "
                        "HTTPS access. No dataset file was substituted."
                    ) from None
                _delay(attempt)
        raise DownloadError("No metadata request attempts were configured")

    def _children(self, url: str) -> list[dict]:
        children: list[dict] = []
        seen: set[str] = set()
        while url:
            _check_url(url, api=True)
            if url in seen or len(seen) >= 100:
                raise DownloadError("Invalid or repeated metadata pagination link")
            seen.add(url)
            page = self._json(url)
            values = page.get("value")
            if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
                raise DownloadError("Official folder metadata has an invalid file list")
            children.extend(values)
            if len(children) > 10_000:
                raise DownloadError("Official folder metadata exceeded the file count limit")
            url = page.get("@odata.nextLink", "")
            if not isinstance(url, str):
                raise DownloadError("Invalid metadata pagination link")
        return children

    @staticmethod
    def _file(value: dict, split: str) -> dict:
        if not isinstance(value, dict):
            raise DownloadError("Official HDF5 metadata is not an object")
        file_info = value.get("file")
        parent = value.get("parentReference")
        if not isinstance(file_info, dict) or not isinstance(parent, dict):
            raise DownloadError("Official HDF5 metadata has invalid file or parent fields")
        hashes = file_info.get("hashes")
        if not isinstance(hashes, dict) or not isinstance(value.get("name"), str):
            raise DownloadError("Official HDF5 metadata has invalid hash or name fields")
        size = value.get("size")
        sha256 = hashes.get("sha256Hash", "")
        if type(size) is not int or not 8 <= size < MAX_BYTES:
            raise DownloadError("Invalid official HDF5 file size")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise DownloadError("Official HDF5 metadata is missing a valid SHA-256 checksum")
        identity = value.get("id")
        drive_id = parent.get("driveId")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(drive_id, str)
            or not drive_id
        ):
            raise DownloadError("Official HDF5 metadata is missing its file identity")
        return {
            "split": split,
            "name": value.get("name"),
            "id": identity,
            "drive_id": drive_id,
            "size": size,
            "sha256": sha256.lower(),
            "url": _check_url(value.get("@content.downloadUrl")),
        }

    @staticmethod
    def _shared_url() -> str:
        share = "u!" + base64.urlsafe_b64encode(SHARE_URL.encode()).decode().rstrip("=")
        return f"{API}/shares/{share}/driveItem/children"

    def inventory(self, expected: dict[str, tuple[str, ...]]) -> list[dict]:
        folders = self._children(self._shared_url())
        files: list[dict] = []
        for split, names in expected.items():
            matches = [
                item
                for item in folders
                if item.get("name") == split and isinstance(item.get("folder"), dict)
            ]
            if len(matches) != 1:
                raise DownloadError(f"Official shared folder must contain one {split} folder")
            folder = matches[0]
            identity = folder.get("id")
            parent = folder.get("parentReference")
            drive_id = parent.get("driveId") if isinstance(parent, dict) else None
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(drive_id, str)
                or not drive_id
            ):
                raise DownloadError("Official folder metadata is missing its identity")
            url = self._item_url(drive_id, identity) + "/children"
            entries = self._children(url)
            found = [entry.get("name") for entry in entries]
            if (
                not all(isinstance(name, str) for name in found)
                or len(found) != len(names)
                or set(found) != set(names)
            ):
                raise DownloadError(
                    f"Official {split} metadata must contain exactly {len(names)} H5 files"
                )
            by_name = {entry["name"]: entry for entry in entries}
            files.extend(self._file(by_name[name], split) for name in names)
        if sum(item["size"] for item in files) >= MAX_BYTES:
            raise DownloadError("EventHDR release must be smaller than 100 GB")
        return files

    @staticmethod
    def _item_url(drive_id: str, identity: str) -> str:
        return f"{API}/drives/{urllib.parse.quote(drive_id, safe='')}/items/{urllib.parse.quote(identity, safe='')}"

    def refresh(self, item: dict) -> dict:
        current = self._file(
            self._json(self._item_url(item["drive_id"], item["id"])), item["split"]
        )
        if any(current[key] != item[key] for key in ("id", "drive_id", "name", "size", "sha256")):
            raise DownloadError("Official file changed; refusing to mix different revisions")
        return current

    def open_file(self, item: dict, offset: int):
        # Obtain a fresh signed URL. Never forward the metadata token to file storage.
        current = self.refresh(item)
        request = urllib.request.Request(
            current["url"],
            headers={"Range": f"bytes={offset}-", "User-Agent": "asgcn-unet/0.3"},
        )
        return self.opener.open(request, timeout=self.timeout)


def _verify(path: Path, item: dict) -> None:
    if path.stat().st_size != item["size"]:
        raise DownloadError(f"{item['name']}: file size does not match official metadata")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if stream.read(8) != HDF5_MAGIC:
            raise DownloadError(f"{item['name']}: downloaded content is not HDF5")
        stream.seek(0)
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise DownloadError(f"{item['name']}: SHA-256 mismatch; file preserved, not accepted")


def _check_response(response, offset: int, size: int) -> None:
    status = response.status
    length = response.headers.get("Content-Length")
    if status == 206:
        value = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
        if match is None:
            raise DownloadError("Invalid Content-Range in HDF5 response")
        start, end, total = map(int, match.groups())
        if start != offset or total != size or not start <= end < size:
            raise DownloadError("HDF5 Content-Range does not match the requested offset and size")
        expected_length = end - start + 1
    elif status == 200 and offset == 0:
        expected_length = size
    else:
        raise DownloadError(
            "Server did not honor the resume range; refusing to append its response"
        )
    if length is None or not length.isdigit() or int(length) != expected_length:
        raise DownloadError("HDF5 Content-Length does not match the expected response size")


def download_file(client: OneDrive, item: dict, target: Path, retries: int = 5) -> str:
    partial = target.with_name(target.name + ".part")
    state_path = target.with_name(target.name + ".part.json")
    identity = {"size": item["size"], "sha256": item["sha256"]}
    if any(path.is_symlink() for path in (target, partial, state_path)):
        raise DownloadError("Refusing symlinked HDF5 download files")
    if target.exists():
        _verify(target, item)
        return "kept"
    if partial.exists() or state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            raise DownloadError(
                "Unverified partial file; preserve it and inspect its resume metadata"
            ) from None
        if saved != identity:
            raise DownloadError(
                "Partial file identity differs from the official checksum; refusing overwrite"
            )
        if partial.exists() and partial.stat().st_size > item["size"]:
            raise DownloadError("Partial file is larger than the official file; refusing overwrite")
    else:
        with state_path.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["size"]:
            break
        try:
            with client.open_file(item, offset) as response:
                _check_response(response, offset, item["size"])
                progress_at = time.monotonic()
                with partial.open("ab") as stream:
                    while chunk := response.read(CHUNK_BYTES):
                        if stream.tell() + len(chunk) > item["size"]:
                            raise DownloadError("HDF5 response exceeded the declared file size")
                        stream.write(chunk)
                        if time.monotonic() - progress_at >= 5:
                            print(
                                f"  {stream.tell() / 1e6:.1f}/{item['size'] / 1e6:.1f} MB received",
                                flush=True,
                            )
                            progress_at = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != item["size"]:
                raise OSError("Incomplete response")
            break
        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
            if not _retryable(error) or attempt + 1 == retries:
                raise DownloadError(
                    f"{item['name']}: download interrupted; partial data retained. "
                    "Run the same --download command to resume."
                ) from None
            print(f"  retry {attempt + 1}/{retries - 1}; keeping received bytes", flush=True)
            _delay(attempt)
    print("  checking SHA-256", flush=True)
    _verify(partial, item)
    # The dataset-level lock prevents cooperating downloaders from replacing each other.
    if target.exists():
        raise DownloadError("Destination appeared during transfer; refusing overwrite")
    partial.replace(target)
    state_path.unlink()
    return "downloaded"


@contextmanager
def _download_lock(destination: Path):
    lock_path = destination / ".download.lock"
    if lock_path.is_symlink():
        raise DownloadError("Refusing a symlinked download lock")
    with lock_path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise DownloadError("Another EventHDR download is active in this destination") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def download_dataset(
    destination: Path, expected: dict[str, tuple[str, ...]], client=None
) -> dict[str, int]:
    client = client or OneDrive()
    files = client.inventory(expected)
    total = sum(item["size"] for item in files)
    print(
        f"Official EventHDR: {len(files)} HDF5 files, {total / 1e9:.2f} GB; verifying SHA-256",
        flush=True,
    )
    if destination.is_symlink():
        raise DownloadError("Refusing a symlinked download root")
    destination.mkdir(parents=True, exist_ok=True)
    with _download_lock(destination):
        for split, names in expected.items():
            directory = destination / split
            if directory.is_symlink():
                raise DownloadError("Refusing a symlinked download split")
            directory.mkdir(exist_ok=True)
            existing = {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.suffix.lower() in {".h5", ".hdf5"}
            }
            if existing - set(names):
                raise DownloadError(
                    f"Unexpected HDF5 files already exist in {split}; refusing to mix data"
                )
        needed = 0
        for item in files:
            target = destination / item["split"] / item["name"]
            partial = target.with_name(target.name + ".part")
            present = target if target.exists() else partial
            if present.is_symlink():
                raise DownloadError("Refusing symlinked HDF5 download files")
            received = present.stat().st_size if present.is_file() else 0
            needed += max(0, item["size"] - received)
        if shutil.disk_usage(destination).free < needed + CHUNK_BYTES:
            raise DownloadError("Not enough free disk space for the remaining EventHDR files")
        counts = {"downloaded": 0, "kept": 0}
        for index, item in enumerate(files, 1):
            print(
                f"[{index}/{len(files)}] {item['split']}/{item['name']} ({item['size'] / 1e6:.1f} MB)",
                flush=True,
            )
            outcome = download_file(client, item, destination / item["split"] / item["name"])
            counts[outcome] += 1
        return counts
