#!/usr/bin/env bash
#
# Build (or rebuild) the HACA3 Singularity/Apptainer image from the current
# repository checkout. The resulting image includes the new theta-only encoder
# CLI `haca3-encode` alongside the existing `haca3-test` / `haca3-train`.
#
# It builds from the LOCAL source you have checked out, so make sure you have
# pulled the branch you want before building.
#
# Build mode (auto-detected, override with HACA3_BUILD_MODE):
#   fakeroot  Unprivileged build that runs entirely as YOU (no sudo). This is
#             the default and the only mode that works when the source lives on
#             root-squashed network storage (e.g. cluster/NFS like /valiant02).
#   sudo      Privileged build via `sudo`. Only works when the source tree is on
#             a filesystem the root user can read (local disk such as /tmp),
#             NOT on root-squashed NFS.
#
# Usage:
#   bash singularity/build_singularity.sh [OUTPUT_SIF]
#   HACA3_BUILD_MODE=sudo bash singularity/build_singularity.sh [OUTPUT_SIF]
#
#   OUTPUT_SIF   Optional path for the image. Default: ./haca3.sif (repo root).
#
# Examples:
#   bash singularity/build_singularity.sh
#   bash singularity/build_singularity.sh /data/images/haca3_encode.sif
#   HACA3_BUILD_MODE=sudo bash singularity/build_singularity.sh

set -euo pipefail

# --- Resolve repo root WITHOUT relying on `cd` ------------------------------
# `cd` into the repo can fail with "Permission denied" when this runs as a
# root-squashed user on NFS, so resolve paths with dirname instead.
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" 2>/dev/null && pwd || dirname "$SCRIPT_SOURCE")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
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

# --- Choose build mode ------------------------------------------------------
BUILD_MODE="${HACA3_BUILD_MODE:-fakeroot}"

echo "==> Builder:    $BUILDER ($($BUILDER --version 2>/dev/null || echo unknown))"
echo "==> Repo root:  $REPO_ROOT"
echo "==> Def file:   $DEF_FILE"
echo "==> Output sif: $OUTPUT_SIF"
echo "==> Build mode: $BUILD_MODE"

# The def file's %files section copies '.', so build from the repo root.
cd "$REPO_ROOT"

case "$BUILD_MODE" in
  fakeroot)
    # Rootless build: runs as the current user, so it works on root-squashed
    # network storage. Requires unprivileged user namespaces / fakeroot to be
    # enabled (common on HPC). If this fails, see the note printed below.
    if ! "$BUILDER" build --fakeroot "$OUTPUT_SIF" "$DEF_FILE"; then
      echo >&2
      echo "ERROR: fakeroot build failed." >&2
      echo "       If your site does not allow --fakeroot, copy the repo to" >&2
      echo "       local disk and build there as root instead, e.g.:" >&2
      echo "         cp -a \"$REPO_ROOT\" /tmp/haca3-build" >&2
      echo "         cd /tmp/haca3-build" >&2
      echo "         HACA3_BUILD_MODE=sudo bash singularity/build_singularity.sh /tmp/haca3.sif" >&2
      echo "         mv /tmp/haca3.sif \"$OUTPUT_SIF\"" >&2
      exit 1
    fi
    ;;
  sudo)
    # Privileged build. NOTE: this reads the source as root, so it only works
    # when REPO_ROOT is on a filesystem root can read (local disk, NOT
    # root-squashed NFS). Re-exec under sudo if we are not already root.
    if [ "$(id -u)" -ne 0 ]; then
      echo "==> Not running as root; re-running the build under sudo..."
      exec sudo "$BUILDER" build "$OUTPUT_SIF" "$DEF_FILE"
    fi
    "$BUILDER" build "$OUTPUT_SIF" "$DEF_FILE"
    ;;
  *)
    echo "ERROR: unknown HACA3_BUILD_MODE='$BUILD_MODE' (expected 'fakeroot' or 'sudo')." >&2
    exit 1
    ;;
esac

echo
echo "==> Build complete: $OUTPUT_SIF"
echo "==> Verify the theta encoder is available inside the image:"
echo "      $BUILDER exec $OUTPUT_SIF haca3-encode --help"
