# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""``exhibit.toml`` ingest-manifest loader (ADR-013 / D-013.1).

The manifest is the single human-authored pre-run input. This loader:

  * parses the TOML (stdlib ``tomllib``),
  * resolves ``env:NAME`` secret indirections from the environment — a missing
    referenced env-var is a hard, named failure (never a silent empty string),
    and an inline (non-``env:``) credential is rejected,
  * builds a typed :class:`Manifest`, and
  * materialises a runtime :class:`PipelineConfig` (the JSON run-record artifact)
    by overlaying the manifest onto the SOTA defaults — an additive front door,
    not a dataclass rewrite.

Secrets are resolved in-memory only. ``PipelineConfig.save()`` already redacts
secret-named fields before persisting, so run snapshots never carry tokens.

CLI::

    python -m pipeline.manifest <exhibit.toml>             # validate + summary
    python -m pipeline.manifest <exhibit.toml> --json      # redacted config -> stdout
    python -m pipeline.manifest <exhibit.toml> -o cfg.json # write redacted config
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.config import PipelineConfig, _redact_secrets

SCHEMA_VERSION = "1.0"

# Keys under [secrets] that MUST use env: indirection (an inline value is a
# misconfiguration). gcloud_project is a plain identifier, not a secret.
_SECRET_KEYS = {"hf_token", "gcloud_credentials"}
_VALID_PRIORITIES = {"key", "standard"}


class ManifestError(ValueError):
    """Raised on a malformed manifest or an unresolved/inline secret."""


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------

@dataclass
class ExhibitMeta:
    id: str = ""
    name: str = ""
    venue: str = ""
    date: str = ""
    curator: str = ""
    description: str = ""


@dataclass
class DriveSource:
    url: str = ""
    rclone_remote: str = "gdrive"
    recursive: bool = True


@dataclass
class ObjectSpec:
    id: str = ""
    name: str = ""
    sam3_concept: str = ""
    description: str = ""
    priority: str = "standard"          # key | standard
    expected_count: int = 1

    @property
    def is_key(self) -> bool:
        return self.priority == "key"


@dataclass
class Secrets:
    """Resolved secret VALUES (in-memory only — never persisted)."""
    hf_token: str = ""
    gcloud_credentials: str = ""
    gcloud_project: str = ""


@dataclass
class PipelineOverrides:
    mesh_backend: Optional[str] = None      # -> training.mesh_method
    matcher: Optional[str] = None           # -> reconstruct.matcher
    comfyui_url: Optional[str] = None        # -> endpoints.comfyui_url (legacy single-host)
    comfyui_api_url: Optional[str] = None    # -> endpoints.comfyui_api_url
    agent_vlm_url: Optional[str] = None      # -> endpoints.agent_vlm_url


@dataclass
class Oversight:
    backend: str = "claude_code"        # claude_code (default) | gemma_local
    artifact_vlm: str = "gemma_local"   # gemma_local | claude_code


@dataclass
class Manifest:
    schema_version: str = SCHEMA_VERSION
    exhibit: ExhibitMeta = field(default_factory=ExhibitMeta)
    drive: DriveSource = field(default_factory=DriveSource)
    objects: list[ObjectSpec] = field(default_factory=list)
    secrets: Secrets = field(default_factory=Secrets)
    pipeline: PipelineOverrides = field(default_factory=PipelineOverrides)
    oversight: Oversight = field(default_factory=Oversight)

    @property
    def key_objects(self) -> list[ObjectSpec]:
        """Objects that trigger the ADR-010 per-object hull/recovery path."""
        return [o for o in self.objects if o.is_key]

    # -- config materialisation ------------------------------------------

    def to_pipeline_config(self, base: Optional[PipelineConfig] = None) -> PipelineConfig:
        """Overlay this manifest onto a PipelineConfig (SOTA defaults by default).

        Mappings (additive — only set fields the manifest actually declares):
          * [[objects]].sam3_concept   -> decompose.sam3_concepts
          * [[objects]].description    -> decompose.descriptions
          * [pipeline].mesh_backend    -> training.mesh_method
          * [pipeline].matcher         -> reconstruct.matcher
          * [pipeline].comfyui_url     -> endpoints.comfyui_url      (optional override)
          * [pipeline].comfyui_api_url -> endpoints.comfyui_api_url  (optional override)
          * [pipeline].agent_vlm_url   -> endpoints.agent_vlm_url    (optional override)
          * [oversight].backend        -> oversight.backend
          * [oversight].artifact_vlm   -> oversight.artifact_vlm
          * [secrets].hf_token         -> inpaint.hf_token (resolved value)
        """
        cfg = base if base is not None else PipelineConfig()

        concepts = [o.sam3_concept for o in self.objects if o.sam3_concept]
        descriptions = [o.description for o in self.objects if o.description]
        if concepts:
            cfg.decompose.sam3_concepts = concepts
        if descriptions:
            cfg.decompose.descriptions = descriptions

        if self.pipeline.mesh_backend:
            cfg.training.mesh_method = self.pipeline.mesh_backend
        if self.pipeline.matcher:
            cfg.reconstruct.matcher = self.pipeline.matcher

        # Optional [pipeline] endpoint overrides (legacy single-host / D-013.3).
        if self.pipeline.comfyui_url:
            cfg.endpoints.comfyui_url = self.pipeline.comfyui_url
        if self.pipeline.comfyui_api_url:
            cfg.endpoints.comfyui_api_url = self.pipeline.comfyui_api_url
        if self.pipeline.agent_vlm_url:
            cfg.endpoints.agent_vlm_url = self.pipeline.agent_vlm_url

        # [oversight] block -> oversight config (D-013.6).
        if self.oversight.backend:
            cfg.oversight.backend = self.oversight.backend
        if self.oversight.artifact_vlm:
            cfg.oversight.artifact_vlm = self.oversight.artifact_vlm

        if self.secrets.hf_token:
            cfg.inpaint.hf_token = self.secrets.hf_token

        return cfg


# ---------------------------------------------------------------------------
#  Secret resolution
# ---------------------------------------------------------------------------

def _resolve_secret(key: str, raw: Any) -> str:
    """Resolve one [secrets] value.

    ``env:NAME`` -> os.environ[NAME] (hard-fail if unset). Secret-named keys
    MUST use env: indirection; an inline value is rejected. gcloud_project and
    other non-secret keys pass through verbatim.
    """
    if raw is None or raw == "":
        return ""
    if not isinstance(raw, str):
        raise ManifestError(f"[secrets].{key} must be a string, got {type(raw).__name__}")

    if raw.startswith("env:"):
        var = raw[4:].strip()
        if not var:
            raise ManifestError(f"[secrets].{key} = 'env:' with no variable name")
        val = os.environ.get(var)
        if val is None or val == "":
            raise ManifestError(
                f"[secrets].{key} references env var ${var}, but it is not set"
            )
        return val

    if key in _SECRET_KEYS:
        raise ManifestError(
            f"[secrets].{key} must use env: indirection (e.g. 'env:HF_TOKEN'); "
            f"inline credentials are forbidden"
        )
    return raw


# ---------------------------------------------------------------------------
#  Parsing
# ---------------------------------------------------------------------------

def _as_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"[{where}] must be a table, got {type(value).__name__}")
    return value


def _parse(raw: dict[str, Any]) -> Manifest:
    version = str(raw.get("schema_version", "")).strip()
    if not version:
        raise ManifestError("manifest missing 'schema_version'")
    if version.split(".", 1)[0] != SCHEMA_VERSION.split(".", 1)[0]:
        raise ManifestError(
            f"unsupported schema_version '{version}' (loader supports {SCHEMA_VERSION})"
        )

    ex_raw = _as_dict(raw.get("exhibit", {}), "exhibit")
    exhibit = ExhibitMeta(
        id=str(ex_raw.get("id", "")),
        name=str(ex_raw.get("name", "")),
        venue=str(ex_raw.get("venue", "")),
        date=str(ex_raw.get("date", "")),
        curator=str(ex_raw.get("curator", "")),
        description=str(ex_raw.get("description", "")),
    )
    if not exhibit.id:
        raise ManifestError("[exhibit].id is required (used as the run-record key)")

    dr_raw = _as_dict(raw.get("drive", {}), "drive")
    drive = DriveSource(
        url=str(dr_raw.get("url", "")),
        rclone_remote=str(dr_raw.get("rclone_remote", "gdrive")),
        recursive=bool(dr_raw.get("recursive", True)),
    )

    objects: list[ObjectSpec] = []
    raw_objects = raw.get("objects", [])
    if raw_objects and not isinstance(raw_objects, list):
        raise ManifestError("[[objects]] must be an array of tables")
    seen_ids: set[str] = set()
    for i, obj in enumerate(raw_objects):
        ob = _as_dict(obj, f"objects[{i}]")
        priority = str(ob.get("priority", "standard"))
        if priority not in _VALID_PRIORITIES:
            raise ManifestError(
                f"objects[{i}].priority must be one of {sorted(_VALID_PRIORITIES)}, "
                f"got '{priority}'"
            )
        oid = str(ob.get("id", ""))
        if oid and oid in seen_ids:
            raise ManifestError(f"duplicate object id '{oid}'")
        if oid:
            seen_ids.add(oid)
        objects.append(ObjectSpec(
            id=oid,
            name=str(ob.get("name", "")),
            sam3_concept=str(ob.get("sam3_concept", "")),
            description=str(ob.get("description", "")),
            priority=priority,
            expected_count=int(ob.get("expected_count", 1)),
        ))

    sec_raw = _as_dict(raw.get("secrets", {}), "secrets")
    secrets = Secrets(
        hf_token=_resolve_secret("hf_token", sec_raw.get("hf_token")),
        gcloud_credentials=_resolve_secret(
            "gcloud_credentials", sec_raw.get("gcloud_credentials")
        ),
        gcloud_project=_resolve_secret("gcloud_project", sec_raw.get("gcloud_project")),
    )

    pl_raw = _as_dict(raw.get("pipeline", {}), "pipeline")
    pipeline = PipelineOverrides(
        mesh_backend=(str(pl_raw["mesh_backend"]) if pl_raw.get("mesh_backend") else None),
        matcher=(str(pl_raw["matcher"]) if pl_raw.get("matcher") else None),
        comfyui_url=(str(pl_raw["comfyui_url"]) if pl_raw.get("comfyui_url") else None),
        comfyui_api_url=(str(pl_raw["comfyui_api_url"]) if pl_raw.get("comfyui_api_url") else None),
        agent_vlm_url=(str(pl_raw["agent_vlm_url"]) if pl_raw.get("agent_vlm_url") else None),
    )

    ov_raw = _as_dict(raw.get("oversight", {}), "oversight")
    oversight = Oversight(
        backend=str(ov_raw.get("backend", "claude_code")),
        artifact_vlm=str(ov_raw.get("artifact_vlm", "gemma_local")),
    )

    return Manifest(
        schema_version=version,
        exhibit=exhibit,
        drive=drive,
        objects=objects,
        secrets=secrets,
        pipeline=pipeline,
        oversight=oversight,
    )


def load_manifest(path: str | Path) -> Manifest:
    """Parse and validate an exhibit.toml, resolving secrets from the env.

    Raises ManifestError on any schema or secret-resolution failure.
    """
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ManifestError(f"invalid TOML in {p}: {e}") from e
    return _parse(raw)


def load_pipeline_config(
    path: str | Path, base: Optional[PipelineConfig] = None
) -> tuple[Manifest, PipelineConfig]:
    """Load a manifest and materialise its PipelineConfig in one call."""
    manifest = load_manifest(path)
    return manifest, manifest.to_pipeline_config(base)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _summary(m: Manifest, cfg: PipelineConfig) -> str:
    lines = [
        "exhibit.toml manifest",
        f"  schema   : {m.schema_version}",
        f"  exhibit  : {m.exhibit.id}  '{m.exhibit.name}'",
        f"  venue    : {m.exhibit.venue or '-'}",
        f"  drive    : {m.drive.url or '-'}  (remote={m.drive.rclone_remote}, "
        f"recursive={m.drive.recursive})",
        f"  objects  : {len(m.objects)} ({len(m.key_objects)} key)",
    ]
    for o in m.objects:
        tag = "*" if o.is_key else " "
        lines.append(f"    [{tag}] {o.id or '-':10} {o.name or '-':24} "
                     f"x{o.expected_count}  <- {o.sam3_concept or '-'}")
    lines += [
        f"  oversight: backend={m.oversight.backend}, "
        f"artifact_vlm={m.oversight.artifact_vlm}",
        f"  secrets  : hf_token={'set' if m.secrets.hf_token else 'unset'}, "
        f"gcloud_credentials={'set' if m.secrets.gcloud_credentials else 'unset'}, "
        f"gcloud_project={m.secrets.gcloud_project or 'unset'}",
        "  -> PipelineConfig overlay:",
        f"       training.mesh_method   = {cfg.training.mesh_method}",
        f"       reconstruct.matcher    = {cfg.reconstruct.matcher}",
        f"       decompose.sam3_concepts= {cfg.decompose.sam3_concepts}",
    ]
    errors = cfg.validate()
    if errors:
        lines.append("  ! config validation errors:")
        lines += [f"      - {e}" for e in errors]
    else:
        lines.append("  config validation: OK")
    return "\n".join(lines)


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="exhibit.toml manifest loader (ADR-013)")
    ap.add_argument("manifest", help="path to exhibit.toml")
    ap.add_argument("--json", action="store_true",
                    help="emit the materialised (redacted) PipelineConfig as JSON")
    ap.add_argument("-o", "--output", help="write the redacted PipelineConfig to PATH")
    args = ap.parse_args(argv)

    try:
        manifest, cfg = load_pipeline_config(args.manifest)
    except ManifestError as e:
        print(f"MANIFEST ERROR: {e}")
        return 2

    if args.output:
        cfg.save(args.output)  # redacts secrets on write
        print(f"wrote redacted PipelineConfig -> {args.output}")
    if args.json:
        print(json.dumps(_redact_secrets(cfg.to_dict()), indent=2, default=str))
    if not args.json and not args.output:
        print(_summary(manifest, cfg))

    return 1 if cfg.validate() else 0


if __name__ == "__main__":
    raise SystemExit(_main())
