"""Data ingestion, profiling, cleaning, encoding, EDA, and visualization utilities."""

from __future__ import annotations

import hashlib
import io
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

import supabase_backend

WORKSPACE = Path("workspace")
MAX_UPLOAD_MB = 50
MAX_ROWS = 200_000


def _slugify(text: str) -> str:
    """Convert a file stem into a valid slug for Chroma collections."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned if len(cleaned) >= 3 else f"ds_{cleaned}"


def to_jsonable(obj: Any) -> Any:
    """Convert numpy, pandas, and other custom objects into JSON-safe structures."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump())
    if isinstance(obj, pd.DataFrame):
        return [
            {k: to_jsonable(v) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj


def run_descriptive_statistics(
    df: pd.DataFrame,
    *,
    operation: str = "mean",
    metric_columns: Optional[list[str]] = None,
    group_columns: Optional[list[str]] = None,
    entity_column: Optional[str] = None,
    item_column: Optional[str] = None,
    count_distinct_items: bool = True,
) -> dict[str, Any]:
    """Compute overall and optional grouped descriptive summaries.

    When ``entity_column`` is provided, first count items per entity, then
    summarize those counts overall and within each requested group. This covers
    questions such as average projects handled per employee when each row is a
    project record.
    """
    supported = {"mean", "median", "sum", "count", "min", "max", "std", "nunique"}
    if operation not in supported:
        return {"error": f"Unsupported descriptive operation '{operation}'."}

    group_columns = group_columns or []
    requested = list(dict.fromkeys([*group_columns, *([entity_column] if entity_column else []), *([item_column] if item_column else []), *(metric_columns or [])]))
    missing = [column for column in requested if column not in df.columns]
    if missing:
        return {"error": f"Column(s) not found in dataset: {', '.join(missing)}."}

    if entity_column:
        entity_df = df.dropna(subset=[entity_column])
        if item_column:
            if count_distinct_items:
                per_entity = entity_df.groupby([*group_columns, entity_column], dropna=False)[item_column].nunique(dropna=True)
            else:
                per_entity = entity_df.groupby([*group_columns, entity_column], dropna=False)[item_column].count()
        else:
            per_entity = entity_df.groupby([*group_columns, entity_column], dropna=False).size()

        if per_entity.empty:
            return {"error": "No valid entity records remain for the requested summary."}

        records: list[dict[str, Any]] = [{
            "scope": "Overall",
            "mean_items_per_entity": float(per_entity.mean()),
            "entities": int(per_entity.count()),
        }]
        if group_columns:
            per_entity_frame = per_entity.rename("item_count").reset_index()
            grouped = per_entity_frame.groupby(group_columns, dropna=False)["item_count"]
            summary = grouped.agg(mean_items_per_entity="mean", entities="count").reset_index()
            for row in summary.to_dict(orient="records"):
                records.append({"scope": "Grouped", **row})
        return {
            "operation": "mean_items_per_entity",
            "entity_column": entity_column,
            "item_column": item_column,
            "group_columns": group_columns,
            "summary": to_jsonable(records),
        }

    if not metric_columns:
        return {"error": "Select at least one metric column for a descriptive summary."}
    if operation in {"mean", "median", "sum", "std", "min", "max"}:
        non_numeric = [column for column in metric_columns if not pd.api.types.is_numeric_dtype(df[column])]
        if non_numeric:
            return {"error": f"Operation '{operation}' requires numeric columns: {', '.join(non_numeric)}."}

    records = []
    for column in metric_columns:
        series = df[column].dropna()
        value = series.nunique() if operation == "nunique" else series.count() if operation == "count" else getattr(series, operation)()
        records.append({"scope": "Overall", "column": column, "operation": operation, "value": value, "rows": int(series.count())})
        if group_columns:
            grouped = df.groupby(group_columns, dropna=False)[column]
            group_values = grouped.nunique() if operation == "nunique" else grouped.count() if operation == "count" else getattr(grouped, operation)()
            group_counts = grouped.count()
            group_frame = pd.concat([group_values.rename("value"), group_counts.rename("rows")], axis=1).reset_index()
            for row in group_frame.to_dict(orient="records"):
                records.append({"scope": "Grouped", "column": column, "operation": operation, **row})

    return {
        "operation": operation,
        "metric_columns": metric_columns,
        "group_columns": group_columns,
        "summary": to_jsonable(records),
    }


def _parse_uploaded_df(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Validate and parse a supported upload without persisting it locally."""
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File size exceeds the {MAX_UPLOAD_MB} MB limit.")

    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    if suffix not in [".csv", ".xlsx"]:
        raise ValueError(f"Unsupported file format '{suffix}'. Only .csv and .xlsx are supported.")

    if suffix == ".csv":
        df = None
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=enc,
                    sep=None,
                    engine="python",
                )
                break
            except Exception:
                continue
        if df is None:
            raise ValueError("Failed to parse CSV file with utf-8 or latin-1 encodings.")
    else:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0)
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}")

    if len(df) > MAX_ROWS:
        raise ValueError(f"File exceeds maximum allowed row count of {MAX_ROWS:,} (found {len(df):,}).")

    return df


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to an in-memory parquet payload."""
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
    except Exception:
        cast_df = df.copy()
        for col in cast_df.select_dtypes(include="object").columns:
            cast_df[col] = cast_df[col].astype(str)
        buffer = io.BytesIO()
        cast_df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def save_upload_for_user(
    file_bytes: bytes,
    filename: str,
    *,
    owner_id: str,
    access_token: str,
    sheet_name: Optional[str] = None,
) -> str:
    """Parse an upload and persist it to the signed-in user's private Supabase paths."""
    from uuid import uuid4

    max_bytes = 50 * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError("Uploads must be 50 MB or smaller.")
    df = _parse_uploaded_df(file_bytes, filename, sheet_name)
    base_url, api_key = supabase_backend.get_config()
    dataset_id = str(uuid4())
    prefix = f"{owner_id}/{dataset_id}"
    suffix = Path(filename).suffix.lower()
    original_path = f"{prefix}/original{suffix}"
    raw_path = f"{prefix}/raw.parquet"
    uploaded_paths: list[str] = []

    try:
        supabase_backend.upload_object(base_url, api_key, access_token, original_path, file_bytes, "application/octet-stream")
        uploaded_paths.append(original_path)
        raw_bytes = _parquet_bytes(df)
        if len(raw_bytes) > max_bytes:
            raise ValueError("The stored dataset exceeds Supabase's 50 MB per-file limit after conversion.")
        supabase_backend.upload_object(base_url, api_key, access_token, raw_path, raw_bytes, "application/octet-stream")
        uploaded_paths.append(raw_path)
        supabase_backend.insert_dataset(base_url, api_key, access_token, {
            "id": dataset_id,
            "owner_id": owner_id,
            "filename": Path(filename).name,
            "original_path": original_path,
            "raw_path": raw_path,
            "sheet_name": sheet_name,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
        })
    except Exception:
        for path in uploaded_paths:
            try:
                supabase_backend.delete_object(base_url, api_key, access_token, path)
            except Exception:
                pass
        raise

    return dataset_id


def save_upload(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> str:
    """Legacy local persistence helper; the authenticated app uses save_upload_for_user."""
    df = _parse_uploaded_df(file_bytes, filename, sheet_name)

    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    file_hash = hashlib.sha1(file_bytes).hexdigest()[:8]
    slug = _slugify(stem)[:40]
    dataset_id = f"{slug}_{file_hash}"

    ds_dir = WORKSPACE / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    original_path = ds_dir / f"original{suffix}"
    original_path.write_bytes(file_bytes)

    raw_parquet_path = ds_dir / "raw.parquet"
    try:
        df.to_parquet(raw_parquet_path, index=False)
    except Exception:
        # On ArrowInvalid or mixed types, cast object columns to string and retry
        cast_df = df.copy()
        for col in cast_df.select_dtypes(include="object").columns:
            cast_df[col] = cast_df[col].astype(str)
        cast_df.to_parquet(raw_parquet_path, index=False)

    meta = {
        "dataset_id": dataset_id,
        "filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "has_clean": False,
    }
    pd.Series(meta).to_json(ds_dir / "meta.json")

    return dataset_id


def list_datasets() -> list[dict]:
    """Return metadata for all persisted datasets in the workspace."""
    if not WORKSPACE.exists():
        return []

    datasets = []
    for path in WORKSPACE.iterdir():
        if path.is_dir():
            meta_file = path / "meta.json"
            if meta_file.exists():
                try:
                    meta = pd.read_json(meta_file, typ="series").to_dict()
                    datasets.append(meta)
                except Exception:
                    continue

    datasets.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
    return datasets


def load_df(dataset_id: str, version: str = "active") -> pd.DataFrame:
    """Load the raw or cleaned DataFrame for a given dataset_id without caching."""
    ds_dir = WORKSPACE / dataset_id
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found in workspace.")

    clean_file = ds_dir / "clean.parquet"
    raw_file = ds_dir / "raw.parquet"

    if version == "clean":
        if not clean_file.exists():
            raise FileNotFoundError(f"Clean version does not exist for dataset '{dataset_id}'.")
        return pd.read_parquet(clean_file)
    elif version == "raw":
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw parquet file not found for dataset '{dataset_id}'.")
        return pd.read_parquet(raw_file)
    else:  # "active"
        if clean_file.exists():
            return pd.read_parquet(clean_file)
        return pd.read_parquet(raw_file)


def _user_dataset_metadata(dataset_id: str, owner_id: str, access_token: str) -> dict[str, Any]:
    base_url, api_key = supabase_backend.get_config()
    record = supabase_backend.get_dataset(base_url, api_key, access_token, dataset_id)
    if record.get("owner_id") != owner_id:
        raise FileNotFoundError("Dataset not found for this account.")
    return {
        **record,
        "dataset_id": record["id"],
        "has_clean": bool(record.get("clean_path")),
        "cleaned_at": record.get("updated_at") if record.get("clean_path") else None,
    }


def list_user_datasets(owner_id: str, access_token: str) -> list[dict[str, Any]]:
    """List only datasets visible to the authenticated user through Supabase RLS."""
    base_url, api_key = supabase_backend.get_config()
    records = supabase_backend.list_datasets(base_url, api_key, access_token, owner_id)
    return [
        {
            **record,
            "dataset_id": record["id"],
            "has_clean": bool(record.get("clean_path")),
            "cleaned_at": record.get("updated_at") if record.get("clean_path") else None,
        }
        for record in records
    ]


def load_user_df(dataset_id: str, *, owner_id: str, access_token: str, version: str = "active") -> pd.DataFrame:
    """Download a user's raw or clean parquet only after RLS-backed ownership lookup."""
    from io import BytesIO

    metadata = _user_dataset_metadata(dataset_id, owner_id, access_token)
    if version == "raw":
        object_path = metadata["raw_path"]
    elif version == "clean":
        object_path = metadata.get("clean_path")
        if not object_path:
            raise FileNotFoundError(f"Clean version does not exist for dataset '{dataset_id}'.")
    else:
        object_path = metadata.get("clean_path") or metadata["raw_path"]
    base_url, api_key = supabase_backend.get_config()
    payload = supabase_backend.download_object(base_url, api_key, access_token, object_path)
    return pd.read_parquet(BytesIO(payload))


def get_user_dataset_metadata(dataset_id: str, *, owner_id: str, access_token: str) -> dict[str, Any]:
    return _user_dataset_metadata(dataset_id, owner_id, access_token)


def get_user_clean_log(dataset_id: str, *, owner_id: str, access_token: str) -> list[dict[str, Any]]:
    metadata = _user_dataset_metadata(dataset_id, owner_id, access_token)
    return metadata.get("clean_log") or []


def save_user_clean(
    dataset_id: str,
    df: pd.DataFrame,
    log: list[dict],
    *,
    owner_id: str,
    access_token: str,
) -> None:
    """Persist a versioned clean file and audit metadata in the user's private storage."""
    from uuid import uuid4

    metadata = _user_dataset_metadata(dataset_id, owner_id, access_token)
    clean_path = f"{owner_id}/{dataset_id}/clean-{uuid4().hex}.parquet"
    base_url, api_key = supabase_backend.get_config()
    clean_bytes = _parquet_bytes(df)
    if len(clean_bytes) > 50 * 1024 * 1024:
        raise ValueError("The cleaned dataset exceeds Supabase's 50 MB per-file limit.")
    supabase_backend.upload_object(base_url, api_key, access_token, clean_path, clean_bytes, "application/octet-stream")
    try:
        supabase_backend.update_dataset(base_url, api_key, access_token, dataset_id, {
            "clean_path": clean_path,
            "clean_log": to_jsonable(log),
            "clean_rows": int(len(df)),
            "clean_cols": int(len(df.columns)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        try:
            supabase_backend.delete_object(base_url, api_key, access_token, clean_path)
        except Exception:
            pass
        raise
    old_clean_path = metadata.get("clean_path")
    if old_clean_path:
        try:
            supabase_backend.delete_object(base_url, api_key, access_token, old_clean_path)
        except Exception:
            # The new version is active; a failed cleanup does not invalidate it.
            pass


def delete_user_dataset(dataset_id: str, *, owner_id: str, access_token: str) -> None:
    """Delete a user's dataset objects, then its metadata row, under RLS ownership."""
    metadata = _user_dataset_metadata(dataset_id, owner_id, access_token)
    base_url, api_key = supabase_backend.get_config()
    for key in ("original_path", "raw_path", "clean_path"):
        path = metadata.get(key)
        if path:
            supabase_backend.delete_object(base_url, api_key, access_token, path)
    supabase_backend.delete_dataset_record(base_url, api_key, access_token, dataset_id)


def _detect_semantic_type(s: pd.Series) -> str:
    """Detect whether a series is boolean, datetime, id_like, text, numeric, or categorical."""
    n = len(s)
    if n == 0:
        return "categorical"

    valid = s.dropna()
    if len(valid) == 0:
        return "categorical"

    # Boolean check
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    unique_vals = set(valid.unique())
    if unique_vals.issubset({0, 1, True, False, "0", "1", "True", "False", "true", "false", "yes", "no"}):
        return "boolean"

    # Datetime check
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        sample = valid.head(50).astype(str)
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.9:
                return "datetime"
        except Exception:
            pass

    # id_like check: unique ratio > 0.95 and not float
    if not pd.api.types.is_float_dtype(s):
        unique_ratio = valid.nunique() / n
        if unique_ratio > 0.95 and valid.nunique() > 10:
            return "id_like"

    # text check: string/object with mean length > 50
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        mean_len = valid.astype(str).str.len().mean()
        if mean_len > 50:
            return "text"

    # numeric check
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    return "categorical"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate structural summary, per-column statistics, and data quality flags."""
    n_rows, n_cols = df.shape
    dup_rows = int(df.duplicated().sum())
    memory_bytes = int(df.memory_usage(deep=True).sum())

    columns_info: dict[str, dict] = {}
    flags: list[dict] = []

    for col in df.columns:
        s = df[col]
        sem_type = _detect_semantic_type(s)
        missing_count = int(s.isna().sum())
        missing_pct = float(missing_count / n_rows) if n_rows > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))

        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "semantic_type": sem_type,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "n_unique": n_unique,
        }

        if sem_type == "numeric":
            valid = s.dropna()
            if len(valid) > 0:
                info["min"] = float(valid.min())
                info["max"] = float(valid.max())
                info["mean"] = float(valid.mean())
                info["std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

        elif sem_type == "categorical":
            top_vals = s.value_counts(dropna=True).head(5).to_dict()
            info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        elif sem_type == "datetime":
            dt_s = pd.to_datetime(s, errors="coerce").dropna()
            if len(dt_s) > 0:
                info["min_date"] = dt_s.min().isoformat()
                info["max_date"] = dt_s.max().isoformat()

        columns_info[col] = info

        # Quality flags
        if n_unique <= 1 and n_rows > 1:
            flags.append({"column": col, "flag": "constant", "detail": "Column has only 1 unique value"})
        if sem_type == "categorical" and n_unique > 50:
            flags.append({"column": col, "flag": "high_cardinality", "detail": f"Categorical with {n_unique} unique levels"})
        if missing_pct > 0.40:
            flags.append({"column": col, "flag": "high_missing", "detail": f"{missing_pct*100:.1f}% missing values"})

    return {
        "shape": [n_rows, n_cols],
        "duplicate_rows": dup_rows,
        "memory_bytes": memory_bytes,
        "columns": columns_info,
        "flags": flags,
    }


def _snake_case(name: str) -> str:
    """Normalize string to snake_case."""
    s = re.sub(r"[\s\-]+", "_", str(name).strip())
    s = re.sub(r"[^\w]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()


def clean_dataset(
    df: pd.DataFrame,
    drop_col_missing_threshold: float = 0.6,
    cap_outliers: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Clean dataset via snake_casing, deduplication, threshold dropping, imputation, and outlier handling."""
    cleaned = df.copy()
    logs: list[dict] = []

    # 1. snake_case column names
    new_cols = {col: _snake_case(col) for col in cleaned.columns}
    renamed_count = sum(1 for o, n in new_cols.items() if o != n)
    if renamed_count > 0:
        cleaned.rename(columns=new_cols, inplace=True)
        logs.append({
            "action": "rename_columns",
            "column": "ALL",
            "detail": f"Standardized {renamed_count} column names to snake_case",
            "rows_affected": 0,
        })

    # 2. Drop exact duplicate rows
    n_dups = int(cleaned.duplicated().sum())
    if n_dups > 0:
        cleaned.drop_duplicates(inplace=True)
        cleaned.reset_index(drop=True, inplace=True)
        logs.append({
            "action": "drop_duplicates",
            "column": "ALL",
            "detail": f"Removed {n_dups} exact duplicate rows",
            "rows_affected": n_dups,
        })

    # 3. Drop columns with missing share > threshold
    cols_to_drop = [
        col for col in cleaned.columns
        if cleaned[col].isna().mean() > drop_col_missing_threshold
    ]
    for col in cols_to_drop:
        missing_pct = cleaned[col].isna().mean()
        cleaned.drop(columns=[col], inplace=True)
        logs.append({
            "action": "drop_column",
            "column": col,
            "detail": f"Dropped column with {missing_pct*100:.1f}% missing values (> {drop_col_missing_threshold*100:.0f}%)",
            "rows_affected": len(cleaned),
        })

    # 4. Coerce object/string columns that are >=90% numeric-parseable or >=90% date-parseable
    for col in list(cleaned.columns):
        if cleaned[col].dtype == "object" or pd.api.types.is_string_dtype(cleaned[col]):
            valid = cleaned[col].dropna()
            if len(valid) == 0:
                continue

            # Try numeric
            num_try = pd.to_numeric(valid, errors="coerce")
            if num_try.notna().mean() >= 0.9:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                logs.append({
                    "action": "coerce_numeric",
                    "column": col,
                    "detail": "Coerced column to numeric (>=90% parseable)",
                    "rows_affected": int(num_try.notna().sum()),
                })
                continue

            # Try datetime
            dt_try = pd.to_datetime(valid, errors="coerce", format="mixed")
            if dt_try.notna().mean() >= 0.9:
                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")
                logs.append({
                    "action": "coerce_datetime",
                    "column": col,
                    "detail": "Coerced column to datetime (>=90% parseable)",
                    "rows_affected": int(dt_try.notna().sum()),
                })

    # 5. Impute
    for col in cleaned.columns:
        s = cleaned[col]
        missing_count = int(s.isna().sum())
        if missing_count == 0:
            continue

        missing_pct = missing_count / len(cleaned)
        sem_type = _detect_semantic_type(s)

        if sem_type == "numeric":
            median_val = s.median()
            cleaned[col] = s.fillna(median_val)
            logs.append({
                "action": "impute_numeric",
                "column": col,
                "detail": f"Imputed {missing_count} missing values with median ({median_val:.3g})",
                "rows_affected": missing_count,
            })
        elif sem_type in ("categorical", "boolean", "text"):
            if missing_pct > 0.20:
                cleaned[col] = s.fillna("Missing")
                logs.append({
                    "action": "impute_category_missing",
                    "column": col,
                    "detail": f"Imputed {missing_count} missing values ({missing_pct*100:.1f}%) with literal 'Missing'",
                    "rows_affected": missing_count,
                })
            else:
                mode_vals = s.mode()
                mode_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Missing"
                cleaned[col] = s.fillna(mode_val)
                logs.append({
                    "action": "impute_category_mode",
                    "column": col,
                    "detail": f"Imputed {missing_count} missing values with mode '{mode_val}'",
                    "rows_affected": missing_count,
                })
        elif sem_type == "datetime":
            logs.append({
                "action": "preserve_nat",
                "column": col,
                "detail": f"Preserved {missing_count} missing datetime values as NaT",
                "rows_affected": missing_count,
            })

    # 6. Outliers: flag via 1.5*IQR. Winsorize to 1st/99th percentile only if cap_outliers=True
    numeric_cols = [c for c in cleaned.columns if _detect_semantic_type(cleaned[c]) == "numeric"]
    for col in numeric_cols:
        s = cleaned[col].dropna()
        if len(s) < 10:
            continue
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        if iqr == 0:
            continue

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_mask = (cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)
        n_outliers = int(outlier_mask.sum())

        if n_outliers > 0:
            if cap_outliers:
                p01 = float(s.quantile(0.01))
                p99 = float(s.quantile(0.99))
                cleaned[col] = cleaned[col].clip(lower=p01, upper=p99)
                logs.append({
                    "action": "winsorize_outliers",
                    "column": col,
                    "detail": f"Winsorized {n_outliers} outliers outside [{lower_bound:.2f}, {upper_bound:.2f}] to 1st/99th percentiles [{p01:.2f}, {p99:.2f}]",
                    "rows_affected": n_outliers,
                })
            else:
                logs.append({
                    "action": "flag_outliers",
                    "column": col,
                    "detail": f"Flagged {n_outliers} outliers outside 1.5*IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    "rows_affected": n_outliers,
                })

    return cleaned, logs


def save_clean(dataset_id: str, df: pd.DataFrame, log: list[dict]) -> None:
    """Save cleaned parquet, cleaning log, and update dataset meta.json."""
    ds_dir = WORKSPACE / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    clean_parquet_path = ds_dir / "clean.parquet"
    df.to_parquet(clean_parquet_path, index=False)

    log_path = ds_dir / "clean_log.json"
    pd.Series(log).to_json(log_path, orient="values")

    meta_file = ds_dir / "meta.json"
    if meta_file.exists():
        try:
            meta = pd.read_json(meta_file, typ="series").to_dict()
        except Exception:
            meta = {}
    else:
        meta = {}

    meta["has_clean"] = True
    meta["cleaned_at"] = datetime.now(timezone.utc).isoformat()
    meta["clean_rows"] = int(len(df))
    meta["clean_cols"] = int(len(df.columns))
    pd.Series(meta).to_json(meta_file)


def encode_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Encode dataframe for user export (one-hot <=10 levels, frequency >10 levels, datetime expanded, id_like dropped)."""
    encoded = df.copy()

    # Identify semantic types
    types = {col: _detect_semantic_type(encoded[col]) for col in encoded.columns}

    # Drop id_like columns
    id_cols = [col for col, t in types.items() if t == "id_like"]
    if id_cols:
        encoded.drop(columns=id_cols, inplace=True)

    # Decompose datetime columns
    dt_cols = [col for col, t in types.items() if t == "datetime" and col in encoded.columns]
    for col in dt_cols:
        parsed = pd.to_datetime(encoded[col], errors="coerce")
        encoded[f"{col}_year"] = parsed.dt.year
        encoded[f"{col}_month"] = parsed.dt.month
        encoded[f"{col}_dayofweek"] = parsed.dt.dayofweek
        encoded.drop(columns=[col], inplace=True)

    # Encode categoricals
    cat_cols = [col for col in encoded.columns if _detect_semantic_type(encoded[col]) in ("categorical", "text")]
    for col in cat_cols:
        n_unique = encoded[col].nunique(dropna=True)
        if n_unique <= 10:
            dummies = pd.get_dummies(encoded[col], prefix=col, drop_first=False, dtype=int)
            encoded = pd.concat([encoded.drop(columns=[col]), dummies], axis=1)
        else:
            freq = encoded[col].value_counts(normalize=True).to_dict()
            encoded[f"{col}_freq"] = encoded[col].map(freq).fillna(0.0)
            encoded.drop(columns=[col], inplace=True)

    return encoded


def run_eda(df: pd.DataFrame) -> dict:
    """Compute descriptive tables, exploratory plotly figures, and rule-based insights."""
    tables: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    insights: list[str] = []

    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if _detect_semantic_type(df[c]) == "numeric"]
    cat_cols = [c for c in df.columns if _detect_semantic_type(df[c]) in ("categorical", "boolean")]
    dt_cols = [c for c in df.columns if _detect_semantic_type(df[c]) == "datetime"]

    # Table 1: Numeric describe + skew/kurtosis
    if numeric_cols:
        num_desc = df[numeric_cols].describe().T
        num_desc["skew"] = df[numeric_cols].skew(numeric_only=True)
        num_desc["kurtosis"] = df[numeric_cols].kurt(numeric_only=True)
        tables["numeric_summary"] = num_desc.reset_index().rename(columns={"index": "column"})

        for col in numeric_cols:
            skew_val = df[col].skew()
            if not pd.isna(skew_val) and abs(skew_val) > 1.5:
                direction = "right-skewed" if skew_val > 0 else "left-skewed"
                insights.append(f"{col} is highly {direction} (skew={skew_val:.2f})")

    # Table 2: Missing values summary
    missing_counts = df.isna().sum()
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_pct": (missing_counts.values / n_rows if n_rows > 0 else 0.0),
    })
    tables["missing_summary"] = missing_df

    for _, row in missing_df.iterrows():
        if row["missing_pct"] > 0.20:
            insights.append(f"Column '{row['column']}' has substantial missingness ({row['missing_pct']*100:.1f}%)")

    # Table 3: Categorical top frequencies
    if cat_cols:
        cat_rows = []
        for col in cat_cols[:8]:
            top_counts = df[col].value_counts(dropna=False).head(3)
            for val, cnt in top_counts.items():
                cat_rows.append({
                    "column": col,
                    "value": str(val),
                    "count": int(cnt),
                    "share": round(cnt / n_rows, 4) if n_rows > 0 else 0.0,
                })
        tables["categorical_summary"] = pd.DataFrame(cat_rows)

    # Figure 1: Missing values bar
    if missing_df["missing_count"].sum() > 0:
        fig_miss = px.bar(
            missing_df[missing_df["missing_count"] > 0],
            x="column",
            y="missing_pct",
            title="Missing Values Ratio by Column",
            labels={"missing_pct": "Missing Share", "column": "Column"},
        )
        fig_miss.update_layout(yaxis_tickformat=".0%")
        figures["missing_values"] = fig_miss

    # Figure 2: Histograms for up to 6 numeric columns
    for col in numeric_cols[:6]:
        fig_hist = px.histogram(
            df,
            x=col,
            marginal="box",
            title=f"Distribution of {col}",
            template="plotly_white",
        )
        figures[f"hist_{col}"] = fig_hist

    # Figure 3: Correlation heatmap (Pearson)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(method="pearson").round(3)
        tables["correlation_matrix"] = corr_matrix.reset_index().rename(columns={"index": "feature"})
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap",
        )
        figures["correlation_heatmap"] = fig_corr

        # Correlation insights (|r| > 0.5)
        seen_pairs = set()
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:
                    r_val = corr_matrix.loc[col1, col2]
                    if not pd.isna(r_val) and abs(r_val) >= 0.5:
                        seen_pairs.add((col1, col2))
                        insights.append(f"{col1} and {col2} are strongly correlated (r={r_val:.2f})")

    # Figure 4: Top category bars
    for col in cat_cols[:4]:
        val_counts = df[col].astype(str).value_counts().head(10).reset_index()
        val_counts.columns = [col, "count"]
        fig_bar = px.bar(
            val_counts,
            x=col,
            y="count",
            title=f"Top Categories for {col}",
            template="plotly_white",
        )
        figures[f"cat_{col}"] = fig_bar

    # Figure 5: Trend line if a datetime column exists
    if dt_cols and numeric_cols:
        time_col = dt_cols[0]
        metric_col = numeric_cols[0]
        try:
            ts_df = df[[time_col, metric_col]].dropna().copy()
            ts_df[time_col] = pd.to_datetime(ts_df[time_col])
            ts_grouped = ts_df.set_index(time_col).resample("W")[metric_col].mean().reset_index()
            fig_trend = px.line(
                ts_grouped,
                x=time_col,
                y=metric_col,
                title=f"Weekly Trend of {metric_col} over {time_col}",
                template="plotly_white",
            )
            figures["time_trend"] = fig_trend
        except Exception:
            pass

    return {
        "tables": tables,
        "figures": figures,
        "insights": insights,
    }


class ChartSpec(BaseModel):
    """Specification schema for generating deterministic Plotly charts."""

    chart_type: Literal["bar", "line", "scatter", "histogram", "box"] = Field(
        ..., description="Type of chart to build"
    )
    x: str = Field(..., description="X-axis column name")
    y: Optional[str] = Field(None, description="Y-axis column name")
    color: Optional[str] = Field(None, description="Optional categorical column for color grouping")
    agg: Optional[Literal["mean", "sum", "count", "median"]] = Field(
        "sum", description="Aggregation function for bar/line charts"
    )
    top_n: int = Field(15, description="Maximum categorical buckets to display")
    title: str = Field("", description="Chart title")


def build_chart(df: pd.DataFrame, spec: ChartSpec | dict) -> go.Figure:
    """Build an aggregated, sanitized Plotly figure according to a ChartSpec."""
    if isinstance(spec, dict):
        spec = ChartSpec(**spec)

    plot_df = df.copy()

    # Validate columns
    if spec.x not in plot_df.columns:
        raise ValueError(f"Column '{spec.x}' not found in dataset.")
    if spec.y and spec.y not in plot_df.columns:
        raise ValueError(f"Column '{spec.y}' not found in dataset.")
    if spec.color and spec.color not in plot_df.columns:
        spec.color = None

    # Handle datetime resampling if >100 distinct dates
    if _detect_semantic_type(plot_df[spec.x]) == "datetime":
        try:
            plot_df[spec.x] = pd.to_datetime(plot_df[spec.x], errors="coerce")
            if plot_df[spec.x].nunique() > 100:
                plot_df[spec.x] = plot_df[spec.x].dt.to_period("M").dt.to_timestamp()
        except Exception:
            pass

    # Aggregations for bar and line charts
    if spec.chart_type in ("bar", "line") and spec.y:
        group_cols = [spec.x]
        if spec.color and spec.color != spec.x:
            group_cols.append(spec.color)

        agg_func = spec.agg or "sum"
        agg_map = {"mean": "mean", "sum": "sum", "count": "count", "median": "median"}
        func = agg_map.get(agg_func, "sum")

        plot_df = (
            plot_df.groupby(group_cols, dropna=False)[spec.y]
            .agg(func)
            .reset_index()
        )

        # Cap top_n for categorical x
        if _detect_semantic_type(df[spec.x]) in ("categorical", "text"):
            top_x = (
                plot_df.groupby(spec.x)[spec.y]
                .sum()
                .sort_values(ascending=False)
                .head(spec.top_n)
                .index
            )
            plot_df = plot_df[plot_df[spec.x].isin(top_x)]

    title_text = spec.title or f"{spec.chart_type.title()} of {spec.y or spec.x} by {spec.x}"

    if spec.chart_type == "bar":
        fig = px.bar(
            plot_df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title_text,
            template="plotly_white",
        )
    elif spec.chart_type == "line":
        fig = px.line(
            plot_df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title_text,
            template="plotly_white",
        )
    elif spec.chart_type == "scatter":
        fig = px.scatter(
            plot_df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title_text,
            template="plotly_white",
        )
    elif spec.chart_type == "histogram":
        fig = px.histogram(
            plot_df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title_text,
            template="plotly_white",
        )
    elif spec.chart_type == "box":
        fig = px.box(
            plot_df,
            x=spec.x,
            y=spec.y,
            color=spec.color,
            title=title_text,
            template="plotly_white",
        )
    else:
        fig = px.scatter(plot_df, x=spec.x, y=spec.y, title=title_text, template="plotly_white")

    return fig
