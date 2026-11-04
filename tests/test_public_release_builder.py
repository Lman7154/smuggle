from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_public_release.py"
SCRIPT_NAMESPACE = runpy.run_path(str(SCRIPT))
WINDOWS_USER_PATH = SCRIPT_NAMESPACE["WINDOWS_USER_PATH"]
VALIDATE_DERIVED_HASHES = SCRIPT_NAMESPACE["_validate_derived_public_hashes"]


def _user_path(*, separator: str, repeated: bool = False) -> str:
    sep = separator * (2 if repeated else 1)
    return f"C:{sep}Users{sep}researcher{sep}project{sep}result.txt"


def test_windows_user_path_matches_native_backslashes() -> None:
    assert WINDOWS_USER_PATH.search(_user_path(separator=chr(92)))


def test_windows_user_path_matches_forward_slashes() -> None:
    assert WINDOWS_USER_PATH.search(_user_path(separator="/"))


def test_windows_user_path_matches_json_escaped_backslashes() -> None:
    assert WINDOWS_USER_PATH.search(_user_path(separator=chr(92), repeated=True))


def test_windows_user_path_ignores_unrelated_text() -> None:
    assert not WINDOWS_USER_PATH.search("C:/Program Files/tool")
    assert not WINDOWS_USER_PATH.search("https://example.test/Users/researcher")
    assert not WINDOWS_USER_PATH.search("Users/researcher/result.txt")


def _derived_tree(tmp_path: Path, *, stale: bool = False) -> Path:
    derived = tmp_path / "data" / "derived"
    derived.mkdir(parents=True)
    payloads = {
        "exp1_metrics.csv": b"column\nvalue\n",
        "exp3a_metrics.csv": b"column\nother\n",
    }
    for name, payload in payloads.items():
        (derived / name).write_bytes(payload)
    hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }
    if stale:
        hashes["exp1_metrics.csv"] = "0" * 64
    (derived / "manifest.json").write_text(
        json.dumps({"public_hashes": hashes}),
        encoding="utf-8",
        newline="\n",
    )
    return tmp_path


def test_nested_public_hashes_match_canonical_release_files(tmp_path) -> None:
    VALIDATE_DERIVED_HASHES(_derived_tree(tmp_path))


def test_nested_public_hashes_reject_stale_value(tmp_path) -> None:
    with pytest.raises(ValueError, match="derived public hash mismatch.*exp1_metrics"):
        VALIDATE_DERIVED_HASHES(_derived_tree(tmp_path, stale=True))
