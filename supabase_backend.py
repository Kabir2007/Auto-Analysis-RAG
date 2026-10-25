"""Supabase Auth, Storage, and PostgREST helpers using the signed-in user's JWT."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

import requests

REQUEST_TIMEOUT = (10, 90)
DATASET_TABLE = "user_datasets"
DATASET_BUCKET = "user-datasets"
_BASE_URL = ""
_API_KEY = ""


class SupabaseError(RuntimeError):
    """An authenticated Supabase request failed."""


def get_config(secret_values: Optional[dict[str, Any]] = None) -> tuple[str, str]:
    """Read only the public project URL and publishable/anon key."""
    values = secret_values or {}
    url = str(values.get("SUPABASE_URL") or os.getenv("SUPABASE_URL") or _BASE_URL or "").strip().rstrip("/")
    key = str(
        values.get("SUPABASE_PUBLISHABLE_KEY")
        or values.get("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or _API_KEY
        or ""
    ).strip()
    return url, key


def configure(base_url: str, api_key: str) -> None:
    """Configure the shared project endpoint and public publishable key."""
    global _BASE_URL, _API_KEY
    _BASE_URL = base_url.strip().rstrip("/")
    _API_KEY = api_key.strip()


def is_configured(secret_values: Optional[dict[str, Any]] = None) -> bool:
    url, key = get_config(secret_values)
    return bool(url and key)


def _base_headers(api_key: str, access_token: Optional[str] = None) -> dict[str, str]:
    headers = {"apikey": api_key, "Accept": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _json_request(method: str, url: str, *, headers: dict[str, str], **kwargs: Any) -> Any:
    try:
        response = requests.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise SupabaseError(f"Could not reach Supabase: {exc}") from exc
    if not response.ok:
        try:
            detail = response.json()
            message = detail.get("msg") or detail.get("message") or detail.get("error_description") or detail.get("error")
        except ValueError:
            message = response.text[:500]
        raise SupabaseError(f"Supabase returned HTTP {response.status_code}: {message or 'request failed'}")
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.content


def _session_payload(data: dict[str, Any]) -> dict[str, Any]:
    session = data.get("session") or data
    user = data.get("user") or session.get("user")
    if not session.get("access_token") or not user or not user.get("id"):
        return {"user": user, "needs_email_confirmation": True}
    return {
        "access_token": session["access_token"],
        "refresh_token": session.get("refresh_token"),
        "expires_at": session.get("expires_at"),
        "expires_in": session.get("expires_in", 3600),
        "user": {"id": user["id"], "email": user.get("email")},
    }


def sign_up(base_url: str, api_key: str, email: str, password: str) -> dict[str, Any]:
    data = _json_request(
        "POST",
        f"{base_url}/auth/v1/signup",
        headers=_base_headers(api_key),
        json={"email": email, "password": password},
    )
    return _session_payload(data or {})


def sign_in(base_url: str, api_key: str, email: str, password: str) -> dict[str, Any]:
    data = _json_request(
        "POST",
        f"{base_url}/auth/v1/token?grant_type=password",
        headers=_base_headers(api_key),
        json={"email": email, "password": password},
    )
    return _session_payload(data or {})


def refresh_session(base_url: str, api_key: str, refresh_token: str) -> dict[str, Any]:
    data = _json_request(
        "POST",
        f"{base_url}/auth/v1/token?grant_type=refresh_token",
        headers=_base_headers(api_key),
        json={"refresh_token": refresh_token},
    )
    return _session_payload(data or {})


def sign_out(base_url: str, api_key: str, access_token: str) -> None:
    _json_request(
        "POST",
        f"{base_url}/auth/v1/logout",
        headers=_base_headers(api_key, access_token),
    )


def _user_headers(api_key: str, access_token: str, *, content_type: Optional[str] = None) -> dict[str, str]:
    headers = _base_headers(api_key, access_token)
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def insert_dataset(base_url: str, api_key: str, access_token: str, record: dict[str, Any]) -> None:
    _json_request(
        "POST",
        f"{base_url}/rest/v1/{DATASET_TABLE}",
        headers={**_user_headers(api_key, access_token, content_type="application/json"), "Prefer": "return=minimal"},
        json=record,
    )


def list_datasets(base_url: str, api_key: str, access_token: str, owner_id: str) -> list[dict[str, Any]]:
    rows = _json_request(
        "GET",
        f"{base_url}/rest/v1/{DATASET_TABLE}",
        headers=_user_headers(api_key, access_token),
        params={"select": "*", "owner_id": f"eq.{owner_id}", "order": "uploaded_at.desc"},
    )
    return rows or []


def get_dataset(base_url: str, api_key: str, access_token: str, dataset_id: str) -> dict[str, Any]:
    rows = _json_request(
        "GET",
        f"{base_url}/rest/v1/{DATASET_TABLE}",
        headers=_user_headers(api_key, access_token),
        params={"select": "*", "id": f"eq.{dataset_id}", "limit": "1"},
    )
    if not rows:
        raise FileNotFoundError("Dataset not found for this account.")
    return rows[0]


def update_dataset(base_url: str, api_key: str, access_token: str, dataset_id: str, changes: dict[str, Any]) -> None:
    _json_request(
        "PATCH",
        f"{base_url}/rest/v1/{DATASET_TABLE}",
        headers={**_user_headers(api_key, access_token, content_type="application/json"), "Prefer": "return=minimal"},
        params={"id": f"eq.{dataset_id}"},
        json=changes,
    )


def delete_dataset_record(base_url: str, api_key: str, access_token: str, dataset_id: str) -> None:
    _json_request(
        "DELETE",
        f"{base_url}/rest/v1/{DATASET_TABLE}",
        headers=_user_headers(api_key, access_token),
        params={"id": f"eq.{dataset_id}"},
    )


def upload_object(base_url: str, api_key: str, access_token: str, path: str, content: bytes, content_type: str) -> None:
    url_path = quote(path, safe="/")
    _json_request(
        "POST",
        f"{base_url}/storage/v1/object/{DATASET_BUCKET}/{url_path}",
        headers=_user_headers(api_key, access_token, content_type=content_type),
        data=content,
    )


def download_object(base_url: str, api_key: str, access_token: str, path: str) -> bytes:
    url_path = quote(path, safe="/")
    result = _json_request(
        "GET",
        f"{base_url}/storage/v1/object/authenticated/{DATASET_BUCKET}/{url_path}",
        headers=_user_headers(api_key, access_token),
    )
    if not isinstance(result, bytes):
        raise SupabaseError("Supabase Storage returned an unexpected file response.")
    return result


def delete_object(base_url: str, api_key: str, access_token: str, path: str) -> None:
    _json_request(
        "DELETE",
        f"{base_url}/storage/v1/object/{DATASET_BUCKET}",
        headers=_user_headers(api_key, access_token, content_type="application/json"),
        json={"prefixes": [path]},
    )
