#!/usr/bin/env bash
#
# Build (or rebuild) the HACA3 Singularity/Apptainer image from the current
# repository checkout. The resulting image includes the new theta-only encoder
# CLI `haca3-encode` alongside the existing `haca3-test` / `haca3-train`.
#
# Run this on a Linux machine that has Singularity (or Apptainer) and sudo.
# It builds from the LOCAL source you have checked out, so make sure you have
# pulled the branch you want before building.
#
# Usage:
#   sudo bash singularity/build_singularity.sh [OUTPUT_SIF]
#
#   OUTPUT_SIF   Optional path for the image. Default: ./haca3.sif (repo root).
#
# Examples:
#   sudo bash singularity/build_singularity.sh
#   sudo bash singularity/build_singularity.sh /data/images/haca3_encode.sif

set -euo pipefail

# --- Resolve repo root (the directory that contains setup.py) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEF_FILE="$SCRIPT_DIR/haca3.def"
OUTPUT_SIF="${1:-$REPO_ROOT/haca3.sif}"

if [ ! -f "$REPO_ROOT/setup.py" ]; then
  echo "ERROR: could not find setup.py at repo root ($REPO_ROOT)." >&2
  exit 1
fi

# --- Pick the container builder (apptainer is the maintained fork) ----------
if command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
elif command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
else
  echo "ERROR: neither 'singularity' nor 'apptainer' was found on PATH." >&2
  echo "       Install one of them on the build machine and retry." >&2
  exit 1
fi

echo "==> Builder:    $BUILDER ($($BUILDER --version 2>/dev/null || echo unknown))"
echo "==> Repo root:  $REPO_ROOT"
echo "==> Def file:   $DEF_FILE"
echo "==> Output sif: $OUTPUT_SIF"

# The def file's %files section copies '.', so build from the repo root.
cd "$REPO_ROOT"

# A normal build needs root. Re-exec under sudo if we are not already root.
if [ "$(id -u)" -ne 0 ]; then
  echo "==> Not running as root; re-running the build under sudo..."
  exec sudo "$BUILDER" build "$OUTPUT_SIF" "$DEF_FILE"
fi

"$BUILDER" build "$OUTPUT_SIF" "$DEF_FILE"

echo
echo "==> Build complete: $OUTPUT_SIF"
echo "==> Verify the theta encoder is available inside the image:"
echo "      $BUILDER exec $OUTPUT_SIF haca3-encode --help"
