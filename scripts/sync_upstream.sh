#!/usr/bin/env bash
# =============================================================================
# scripts/sync_upstream.sh
# Upstream sync script for the Gaussian Toolkit fork of LichtFeld-Studio.
#
# PURPOSE
#   Pull changes from MrNeRF/LichtFeld-Studio into this ISOLATED fork.
#   This is a ONE-WAY PULL ONLY. Pushing to upstream is DISABLED at the
#   git-remote level and is FORBIDDEN by project policy (ADR-002).
#
# USAGE
#   # Dry run — shows conflict zone, does NOT merge
#   ./scripts/sync_upstream.sh
#
#   # Actually execute the merge
#   DO_MERGE=1 ./scripts/sync_upstream.sh
#
#   # Override the target ref (use with care — see Vulkan warning below)
#   UPSTREAM_REF=v0.5.1 ./scripts/sync_upstream.sh
#   UPSTREAM_REF=master FORCE=1 ./scripts/sync_upstream.sh
#
# ENVIRONMENT VARIABLES
#   UPSTREAM_REF  Tag or commit to sync from. Default: v0.5.2
#   DO_MERGE      Set to 1 to execute the git merge step. Omit for dry run.
#   FORCE         Set to 1 to bypass the bleeding-edge branch guard.
#
# ISOLATION POLICY (read this before every run)
#   - NEVER push to the 'upstream' remote — the push URL is set to DISABLED.
#   - NEVER open a PR against MrNeRF/LichtFeld-Studio.
#   - Sync is ALWAYS inbound: fetch from upstream, merge into a local branch.
#   - This script will NEVER run 'git push' in any form.
#
# RELATED DOCUMENTS
#   research/decisions/adr-002-upstream-sync-strategy.md
#   research/decisions/adr-008-defer-vulkan-migration.md
#   research/decisions/upstream-sync-runbook.md
#   BOUNDARIES.md
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPSTREAM_URL="https://github.com/MrNeRF/LichtFeld-Studio.git"
UPSTREAM_REMOTE="upstream"

# Default sync target — the stable v0.5.2 tag recommended by ADR-002.
# Override with UPSTREAM_REF env var if needed.
UPSTREAM_REF="${UPSTREAM_REF:-v0.5.2}"

# Set DO_MERGE=1 to actually run the merge step.
# Without it the script is a read-only conflict-zone inspection.
DO_MERGE="${DO_MERGE:-0}"

# Set FORCE=1 to override the bleeding-edge branch guard.
FORCE="${FORCE:-0}"

# ---------------------------------------------------------------------------
# Colour helpers (safe: no-op when not a terminal)
# ---------------------------------------------------------------------------
RED=""
YELLOW=""
GREEN=""
BOLD=""
RESET=""
if [ -t 1 ]; then
    RED="\033[0;31m"
    YELLOW="\033[0;33m"
    GREEN="\033[0;32m"
    BOLD="\033[1m"
    RESET="\033[0m"
fi

info()  { printf "${GREEN}[INFO]${RESET}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${RESET}  %s\n" "$*"; }
error() { printf "${RED}[ERROR]${RESET} %s\n" "$*" >&2; }
die()   { error "$*"; exit 1; }
banner(){ printf "\n${BOLD}=== %s ===${RESET}\n\n" "$*"; }

# ---------------------------------------------------------------------------
# Guard: ensure we are inside the repository root
# ---------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || die "Not inside a git repository. Run from within LichtFeld-Studio/."
info "Repository root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Guard: refuse to run with a dirty working tree
# Merging into a dirty tree risks confusion about what changed.
# ---------------------------------------------------------------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree is dirty (uncommitted changes or staged files). " \
        "Stash or commit before running this script."
fi
info "Working tree is clean."

# ---------------------------------------------------------------------------
# Guard: bleeding-edge branch detection (ADR-008)
#
# If UPSTREAM_REF looks like a live branch (master, main, HEAD, or any name
# without a version-like prefix) we warn loudly about the Vulkan migration
# and refuse to proceed unless FORCE=1.
#
# ADR-008 documents the risks: CUDA/OpenGL renderer removed (#1170, #1234),
# coordinate-system regression (#1066, issue #1104), unreleased v0.5.3,
# Python/MCP API churn. Do NOT sync to master until all ADR-008 trigger
# conditions are satisfied.
# ---------------------------------------------------------------------------
is_version_tag() {
    # Returns 0 (true) if the string looks like a version tag: v1.2, v0.5.2, etc.
    [[ "$1" =~ ^v[0-9]+\.[0-9]+ ]]
}

if ! is_version_tag "${UPSTREAM_REF}"; then
    warn "UPSTREAM_REF='${UPSTREAM_REF}' does not look like a stable version tag."
    warn ""
    warn "*** VULKAN MIGRATION WARNING (ADR-008) ***"
    warn "Upstream master (v0.5.3-dev) has REMOVED the CUDA and OpenGL"
    warn "rendering backends entirely (#1170, #1234). Syncing to master will:"
    warn "  - Break any LichtFeld rendering call in our pipeline."
    warn "  - Risk propagating coordinate-system regression #1066 into"
    warn "    src/pipeline/coordinate_transform.py (known issue #1104)."
    warn "  - Require Vulkan system packages not present in our Docker image."
    warn "  - Require MCP/Python API audit of mcp_client.py."
    warn ""
    warn "ADR-008 trigger conditions must ALL be met before syncing to master."
    warn "See: research/decisions/adr-008-defer-vulkan-migration.md"
    warn ""
    if [ "${FORCE:-0}" != "1" ]; then
        die "Aborting. Set FORCE=1 to override this guard (only if you have" \
            "read ADR-008 and confirmed all trigger conditions are met)."
    fi
    warn "FORCE=1 set — proceeding despite Vulkan migration warning."
fi

# ---------------------------------------------------------------------------
# Step 1: Ensure the 'upstream' remote exists and is pointed at the right URL
# ---------------------------------------------------------------------------
banner "Step 1: Configure upstream remote"

if git remote get-url "${UPSTREAM_REMOTE}" &>/dev/null; then
    EXISTING_URL="$(git remote get-url "${UPSTREAM_REMOTE}")"
    if [ "${EXISTING_URL}" != "${UPSTREAM_URL}" ]; then
        warn "Remote '${UPSTREAM_REMOTE}' exists but points to '${EXISTING_URL}'."
        warn "Setting fetch URL to '${UPSTREAM_URL}'."
        git remote set-url "${UPSTREAM_REMOTE}" "${UPSTREAM_URL}"
    else
        info "Remote '${UPSTREAM_REMOTE}' already exists with correct fetch URL."
    fi
else
    info "Adding remote '${UPSTREAM_REMOTE}' -> ${UPSTREAM_URL}"
    git remote add "${UPSTREAM_REMOTE}" "${UPSTREAM_URL}"
fi

# CRITICAL ISOLATION GUARD: Disable the push URL for the upstream remote.
# This means 'git push upstream <anything>' will fail immediately with a
# "fatal: 'DISABLED' does not appear to be a git repository" error.
# Accidental pushes to MrNeRF/LichtFeld-Studio are FORBIDDEN by ADR-002.
info "Disabling push URL for '${UPSTREAM_REMOTE}' remote (isolation policy)."
git remote set-url --push "${UPSTREAM_REMOTE}" "DISABLED"

# Verify the configuration
FETCH_URL="$(git remote get-url "${UPSTREAM_REMOTE}")"
PUSH_URL="$(git remote get-url --push "${UPSTREAM_REMOTE}")"
info "  fetch URL : ${FETCH_URL}"
info "  push  URL : ${PUSH_URL}  <-- pushes will fail loudly (by design)"

# ---------------------------------------------------------------------------
# Step 2: Fetch upstream tags and refs
# ---------------------------------------------------------------------------
banner "Step 2: Fetch upstream"

info "Fetching from '${UPSTREAM_REMOTE}' (tags included) ..."
git fetch "${UPSTREAM_REMOTE}" --tags

# Verify the target ref is now locally known
if ! git rev-parse "${UPSTREAM_REF}" &>/dev/null; then
    # The ref might be a remote branch rather than a tag
    if git rev-parse "${UPSTREAM_REMOTE}/${UPSTREAM_REF}" &>/dev/null; then
        info "Ref '${UPSTREAM_REF}' resolved via remote tracking: " \
             "${UPSTREAM_REMOTE}/${UPSTREAM_REF}"
        MERGE_TARGET="${UPSTREAM_REMOTE}/${UPSTREAM_REF}"
    else
        die "Ref '${UPSTREAM_REF}' not found locally or in '${UPSTREAM_REMOTE}'." \
            "Check that the tag or branch name is correct."
    fi
else
    MERGE_TARGET="${UPSTREAM_REF}"
    info "Ref '${UPSTREAM_REF}' is available locally."
fi

TARGET_SHA="$(git rev-parse "${MERGE_TARGET}")"
info "Sync target: ${MERGE_TARGET} (${TARGET_SHA})"

# ---------------------------------------------------------------------------
# Step 3: Determine current branch and compute merge base
# ---------------------------------------------------------------------------
banner "Step 3: Branch context"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
info "Current branch: ${CURRENT_BRANCH}"

# The sync branch we will create
SYNC_BRANCH="sync/upstream-${UPSTREAM_REF}"
info "Sync branch will be: ${SYNC_BRANCH}"

# Warn if the sync branch already exists (operator may be re-running after
# a previous dry run or partial merge attempt)
if git rev-parse --verify "${SYNC_BRANCH}" &>/dev/null; then
    warn "Branch '${SYNC_BRANCH}' already exists."
    warn "If you are re-running after a failed merge, check its state first."
    warn "To start fresh: git branch -D '${SYNC_BRANCH}' then re-run this script."
    die "Aborting to prevent clobbering an existing sync branch." \
        "Delete it manually if you want to restart."
fi

# Find merge base so we know what territory the merge covers
MERGE_BASE="$(git merge-base HEAD "${MERGE_TARGET}" 2>/dev/null || echo 'none')"
if [ "${MERGE_BASE}" = "none" ]; then
    warn "No merge base found between HEAD and '${MERGE_TARGET}'."
    warn "These histories appear completely unrelated. Proceed with extreme caution."
else
    info "Merge base: ${MERGE_BASE}"
    OUR_AHEAD="$(git log --oneline "${MERGE_BASE}..HEAD" | wc -l)"
    THEIR_AHEAD="$(git log --oneline "${MERGE_BASE}..${MERGE_TARGET}" | wc -l)"
    info "Our commits since merge base  : ${OUR_AHEAD}"
    info "Upstream commits since merge base: ${THEIR_AHEAD}"
fi

# ---------------------------------------------------------------------------
# Step 4: Conflict-zone report (always runs — this is the dry-run output)
# ---------------------------------------------------------------------------
banner "Step 4: Conflict-zone report (read-only)"

info "Files changed in upstream since merge base:"
echo "---"
git diff --name-only "${MERGE_BASE}..${MERGE_TARGET}" | sort
echo "---"

info "Files changed in OUR fork since merge base:"
echo "---"
git diff --name-only "${MERGE_BASE}..HEAD" | sort
echo "---"

info "Potential conflict zones (changed in BOTH):"
echo "---"
# Compute the intersection: files modified in both histories
OUR_FILES="$(git diff --name-only "${MERGE_BASE}..HEAD" | sort)"
THEIR_FILES="$(git diff --name-only "${MERGE_BASE}..${MERGE_TARGET}" | sort)"
CONFLICTS="$(comm -12 \
    <(echo "${OUR_FILES}") \
    <(echo "${THEIR_FILES}"))"

if [ -z "${CONFLICTS}" ]; then
    info "No overlapping file changes detected. Merge should be clean."
else
    printf "%s\n" "${CONFLICTS}"
    echo "---"
    warn "The files above were modified in BOTH histories."
    warn "Resolve conflicts per BOUNDARIES.md rules:"
    warn "  UPSTREAM dirs (src/core/, src/app/, src/mcp/, src/rendering/,"
    warn "  src/training/, src/geometry/, src/io/, src/sequencer/,"
    warn "  src/visualizer/, src/python/, cmake/, external/, eval/,"
    warn "  tools/, tests/, CMakeLists.txt, vcpkg.json) -> accept UPSTREAM"
    warn "  OUR dirs (src/pipeline/, src/web/, docker/, scripts/,"
    warn "  research/, docs/, Dockerfile.consolidated,"
    warn "  docker-compose.consolidated.yml) -> keep OURS"
    warn "  .gitignore -> merge both sections"
    warn "  README.md -> keep OURS"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Gate — require DO_MERGE=1 to proceed past the report
# ---------------------------------------------------------------------------
if [ "${DO_MERGE}" != "1" ]; then
    info "Dry-run complete. The conflict zone report is above."
    info ""
    info "No changes have been made to the repository."
    info "To proceed with the actual merge, review the report and then run:"
    info ""
    info "  DO_MERGE=1 ./scripts/sync_upstream.sh"
    info ""
    info "Or, if syncing a different ref:"
    info "  UPSTREAM_REF=${UPSTREAM_REF} DO_MERGE=1 ./scripts/sync_upstream.sh"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 6: Create the sync branch and execute the merge
# ---------------------------------------------------------------------------
banner "Step 6: Create sync branch and merge"

info "Creating branch '${SYNC_BRANCH}' from '${CURRENT_BRANCH}' ..."
git checkout -b "${SYNC_BRANCH}"

info "Merging '${MERGE_TARGET}' into '${SYNC_BRANCH}' (--no-ff) ..."
info ""
info "If conflicts arise, this script will print the BOUNDARIES.md resolution"
info "rules and exit non-zero. You must resolve conflicts manually."
info "DO NOT run 'git merge --abort' unless you want to discard the sync."
info ""

# Run the merge. On conflict, git exits non-zero, which triggers set -e.
# We catch that, print guidance, and re-exit non-zero.
if ! git merge --no-ff "${MERGE_TARGET}" \
        -m "Sync upstream LichtFeld-Studio ${UPSTREAM_REF}

One-way pull from https://github.com/MrNeRF/LichtFeld-Studio.git
Target: ${UPSTREAM_REF} (${TARGET_SHA})
Merge base: ${MERGE_BASE}

This fork is isolated. No code was pushed to upstream.
See: research/decisions/adr-002-upstream-sync-strategy.md"; then

    echo ""
    error "Merge produced conflicts. Resolve them manually, then run:"
    error "  git add <resolved files>"
    error "  git merge --continue"
    error ""
    error "BOUNDARIES.md conflict resolution rules:"
    error "  UPSTREAM directories -> accept upstream version (theirs):"
    error "    src/core/  src/app/  src/mcp/  src/rendering/"
    error "    src/training/  src/geometry/  src/io/  src/sequencer/"
    error "    src/visualizer/  src/python/  cmake/  external/"
    error "    eval/  tools/  tests/  CMakeLists.txt  vcpkg.json"
    error "    CONTRIBUTING.md  LICENSE  THIRD_PARTY_LICENSES.md"
    error "  OUR directories -> keep our version (ours):"
    error "    src/pipeline/  src/web/  docker/  scripts/"
    error "    research/  docs/  Dockerfile.consolidated"
    error "    docker-compose.consolidated.yml  BOUNDARIES.md"
    error "    GAUSSIAN_TOOLKIT_README.md  AGENTS.md  CLAUDE_CONTAINER.md"
    error "  MERGE BOTH:"
    error "    .gitignore  (upstream block first, then ours)"
    error "  KEEP OURS:"
    error "    README.md"
    error ""
    error "Quick git commands for conflict resolution:"
    error "  # Accept upstream for a file in an upstream directory:"
    error "  git checkout --theirs <file> && git add <file>"
    error "  # Keep ours for a file in our directories:"
    error "  git checkout --ours <file> && git add <file>"
    error ""
    error "See research/decisions/upstream-sync-runbook.md for the full procedure."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 7: Post-merge summary
# ---------------------------------------------------------------------------
banner "Step 7: Merge complete"

info "Branch '${SYNC_BRANCH}' now contains the merged result."
info ""
info "NEXT STEPS (from the runbook):"
info "  1. Review: git log --oneline main..${SYNC_BRANCH}"
info "  2. Rebuild the C++ core inside Docker:"
info "       docker compose -f docker-compose.consolidated.yml build"
info "  3. Verify MCP server: docker exec <container> ./LichtFeld_Studio"
info "     and confirm it responds on port 45677."
info "  4. Run pipeline end-to-end: python -m src.pipeline <test-video>"
info "  5. Test native USD export."
info "  6. Test mcp_client.py against updated MCP server (#984 changes)."
info "  7. When all checks pass, open an internal PR:"
info "       ${SYNC_BRANCH} -> main"
info ""
info "ISOLATION REMINDER:"
info "  This script has NOT pushed anything. Do NOT push to 'upstream'."
info "  Only 'git push origin ${SYNC_BRANCH}' to YOUR fork is permitted."
info ""

# ---------------------------------------------------------------------------
# Final guard: assert we never called git push
# If the script reaches here, we definitely did not push anywhere.
# This comment is intentional documentation, not dead code.
# ---------------------------------------------------------------------------
# git push is NEVER called in this script. The push guard is structural:
# the 'upstream' push URL is DISABLED, and the script contains no 'git push'
# invocation. Grep for 'git push' to verify.
