#!/usr/bin/env bash
# resolve_pins.sh — resolve the floating refs in pins.lock.toml to real,
# reproducible identifiers. RUN THIS ON THE GPU HOST (.48), where the repos are
# actually checked out and the packages are actually installed.
#
# For each component:
#   kind="git"  + a present host_path  -> READ-ONLY `git -C <path> rev-parse HEAD`
#                                          writes the SHA into resolved_commit.
#   kind="pip"                          -> `pip show <name>` writes the installed
#                                          version into `version`.
#   kind="checkpoint"                   -> (none today) left untouched.
#
# READ-ONLY GUARANTEE: this script NEVER checkout/fetch/pull/clone/modify any
# repo. It only calls `rev-parse` (a read). Missing paths warn and continue.
# It NEVER fabricates a hash — an unresolvable component stays empty.
#
# Idempotent: re-running overwrites the output with freshly-read values.
#
# Usage:
#   scripts/resolve_pins.sh [--input pins.lock.toml] [--output pins.resolved.toml]
#                           [--pip CMD] [-h|--help]
#
#   --input   PATH   source lock (default: pins.lock.toml next to this script's repo root)
#   --output  PATH   destination (default: pins.resolved.toml beside the input)
#   --pip     CMD    pip command to query (default: pip3, falls back to pip)
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate repo root (parent of this script's dir) ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

INPUT="${REPO_ROOT}/pins.lock.toml"
OUTPUT=""
PIP_CMD=""

usage() {
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# --- arg parse -------------------------------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)  INPUT="${2:?--input needs a path}"; shift 2 ;;
        --output) OUTPUT="${2:?--output needs a path}"; shift 2 ;;
        --pip)    PIP_CMD="${2:?--pip needs a command}"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [ -z "${OUTPUT}" ]; then
    OUTPUT="$(dirname "${INPUT}")/pins.resolved.toml"
fi

# --- pip command discovery -------------------------------------------------
if [ -z "${PIP_CMD}" ]; then
    if command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3"
    elif command -v pip >/dev/null 2>&1; then
        PIP_CMD="pip"
    else
        PIP_CMD=""   # no pip: pip components stay unresolved (warned below)
    fi
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required (for tomllib parsing)." >&2
    exit 1
fi

if [ ! -f "${INPUT}" ]; then
    echo "ERROR: input lock not found: ${INPUT}" >&2
    exit 1
fi

echo "resolve_pins: input  = ${INPUT}"
echo "resolve_pins: output = ${OUTPUT}"
echo "resolve_pins: pip    = ${PIP_CMD:-<none>}"
echo

# ---------------------------------------------------------------------------
# Safe tilde-expansion: only a leading "~/" or bare "~" maps to $HOME. Anything
# else (incl. embedded ~) is returned verbatim. Empty stays empty.
# ---------------------------------------------------------------------------
expand_path() {
    local p="$1"
    case "${p}" in
        "~")       printf '%s' "${HOME}" ;;
        "~/"*)     printf '%s' "${HOME}/${p#\~/}" ;;
        *)         printf '%s' "${p}" ;;
    esac
}

# ---------------------------------------------------------------------------
# READ the lock with python3/tomllib and emit one TSV row per component:
#   index \t name \t kind \t host_path
# (tabs are safe; names/paths here contain none.)
# ---------------------------------------------------------------------------
read_components() {
    python3 - "${INPUT}" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    data = tomllib.load(fh)
for i, c in enumerate(data.get("component", [])):
    print("\t".join((
        str(i),
        str(c.get("name", "")),
        str(c.get("kind", "")),
        str(c.get("host_path", "")),
    )))
PY
}

# ---------------------------------------------------------------------------
# Resolve every component into two parallel temp files keyed by index:
#   <tmp>/commit.<i>   -> resolved git SHA  (empty if unresolved)
#   <tmp>/version.<i>  -> resolved pip ver  (empty if unresolved)
# Plus a summary stream for the printed table.
# ---------------------------------------------------------------------------
TMPDIR_RES="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_RES}"' EXIT

SUMMARY="${TMPDIR_RES}/summary.tsv"
: > "${SUMMARY}"

while IFS=$'\t' read -r idx name kind host_path; do
    [ -n "${idx}" ] || continue
    commit=""
    version=""
    status=""

    case "${kind}" in
        git)
            if [ -z "${host_path}" ]; then
                status="SKIP(no host_path)"
            else
                expanded="$(expand_path "${host_path}")"
                if [ -d "${expanded}/.git" ] || git -C "${expanded}" rev-parse --git-dir >/dev/null 2>&1; then
                    # READ-ONLY: rev-parse never mutates the repo.
                    if commit="$(git -C "${expanded}" rev-parse HEAD 2>/dev/null)"; then
                        status="OK"
                    else
                        commit=""
                        status="WARN(rev-parse failed)"
                        echo "WARN: ${name}: git rev-parse failed at ${expanded}" >&2
                    fi
                else
                    status="WARN(missing repo)"
                    echo "WARN: ${name}: no git repo at ${expanded} — leaving unresolved" >&2
                fi
            fi
            ;;
        pip)
            if [ -z "${PIP_CMD}" ]; then
                status="WARN(no pip)"
                echo "WARN: ${name}: no pip available — leaving unresolved" >&2
            else
                # `pip show` is read-only.
                if version="$("${PIP_CMD}" show "${name}" 2>/dev/null \
                        | awk -F': ' '/^Version:/{print $2; exit}')" && [ -n "${version}" ]; then
                    status="OK"
                else
                    version=""
                    status="WARN(not installed)"
                    echo "WARN: ${name}: '${PIP_CMD} show ${name}' found no version — leaving unresolved" >&2
                fi
            fi
            ;;
        checkpoint)
            status="SKIP(checkpoint)"
            ;;
        *)
            status="SKIP(kind=${kind:-?})"
            ;;
    esac

    printf '%s' "${commit}"  > "${TMPDIR_RES}/commit.${idx}"
    printf '%s' "${version}" > "${TMPDIR_RES}/version.${idx}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${idx}" "${name}" "${kind}" "${status}" "${commit:-${version}}" >> "${SUMMARY}"
done < <(read_components)

# ---------------------------------------------------------------------------
# WRITE the resolved TOML. python3 re-reads the input for full fidelity of all
# original keys/values, then overlays resolved_commit / version from the temp
# files and emits a clean, valid TOML document (string values quoted/escaped).
# ---------------------------------------------------------------------------
python3 - "${INPUT}" "${OUTPUT}" "${TMPDIR_RES}" <<'PY'
import sys, os, tomllib, datetime

inp, outp, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(inp, "rb") as fh:
    data = tomllib.load(fh)

def read_tmp(prefix, idx):
    path = os.path.join(tmp, f"{prefix}.{idx}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

def toml_str(s):
    # basic TOML string with escaping
    s = (s.replace("\\", "\\\\")
          .replace('"', '\\"')
          .replace("\n", "\\n")
          .replace("\t", "\\t"))
    return f'"{s}"'

components = data.get("component", [])
for i, c in enumerate(components):
    kind = c.get("kind", "")
    if kind == "git":
        commit = read_tmp("commit", i)
        if commit:
            c["resolved_commit"] = commit
    elif kind == "pip":
        version = read_tmp("version", i)
        if version:
            c["version"] = version
        # keep resolved_commit as-is (n/a for pip)

# Preferred key order per component for readable output.
KEY_ORDER = ["name", "repo", "kind", "ref", "version",
             "resolved_commit", "host_path", "clone_site", "note"]

lines = []
lines.append("# pins.resolved.toml — generated by scripts/resolve_pins.sh")
lines.append("# Resolved on the GPU host. resolved_commit / version reflect the")
lines.append("# real checked-out repos / installed packages at resolution time.")
lines.append(f"# resolved_at = {datetime.datetime.now().astimezone().isoformat()}")
lines.append("")
sv = data.get("schema_version", "1.0")
lines.append(f"schema_version = {toml_str(str(sv))}")
lines.append("")

for c in components:
    lines.append("[[component]]")
    seen = set()
    for key in KEY_ORDER:
        if key in c:
            lines.append(f"{key} = {toml_str(str(c[key]))}")
            seen.add(key)
    for key, val in c.items():
        if key not in seen:
            lines.append(f"{key} = {toml_str(str(val))}")
    lines.append("")

with open(outp, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines).rstrip("\n") + "\n")

print(f"resolve_pins: wrote {outp} ({len(components)} components)")
PY

# ---------------------------------------------------------------------------
# Print the summary table.
# ---------------------------------------------------------------------------
echo
echo "=== resolution summary ==="
printf '%-22s %-6s %-22s %s\n' "COMPONENT" "KIND" "STATUS" "RESOLVED"
printf '%-22s %-6s %-22s %s\n' "----------------------" "------" "----------------------" "--------"
resolved_ct=0
total_ct=0
while IFS=$'\t' read -r _idx name kind status resolved; do
    [ -n "${name}" ] || continue
    total_ct=$((total_ct + 1))
    case "${status}" in OK) resolved_ct=$((resolved_ct + 1)) ;; esac
    short="${resolved:0:40}"
    printf '%-22s %-6s %-22s %s\n' "${name}" "${kind}" "${status}" "${short}"
done < "${SUMMARY}"
echo "--------------------------"
echo "resolved ${resolved_ct}/${total_ct} components -> ${OUTPUT}"

# Non-zero exit only on hard failure; partial resolution is a normal state
# (e.g. running off-host where some repos are absent).
exit 0
