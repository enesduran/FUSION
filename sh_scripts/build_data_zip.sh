#!/bin/bash
# Build data.zip from the exact paths that unzip_data.sh verifies.
#
# The path list is parsed straight out of unzip_data.sh's REQUIRED_PATHS
# array, so this archive and that check can never drift apart. The resulting
# data.zip stores paths relative to the repo root (data/...), so the matching
# `unzip -o data.zip -d .` in unzip_data.sh reproduces the data/ tree.
#
# Usage:
#   sh_scripts/build_data_zip.sh [-o OUTPUT] [--list] [--allow-missing]
#     --list           dry run: show what would be bundled (+ sizes), zip nothing
#     --allow-missing  warn instead of abort when a required path is absent
#     -o, --output P   write archive to P (default: <repo>/data.zip)
set -euo pipefail

# This script lives in <repo>/sh_scripts/ ; operate on the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_SH="$SCRIPT_DIR/unzip_data.sh"
OUT_ZIP="$REPO_ROOT/data.zip"

LIST_ONLY=0
ALLOW_MISSING=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --list|--dry-run) LIST_ONLY=1 ;;
        --allow-missing)  ALLOW_MISSING=1 ;;
        -o|--output)      shift; OUT_ZIP="$1" ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$DATA_SH" ] || { echo "Error: cannot find $DATA_SH to read REQUIRED_PATHS." >&2; exit 1; }
if [ "$LIST_ONLY" -eq 0 ]; then
    command -v zip >/dev/null 2>&1 || { echo "Error: 'zip' is required but not installed." >&2; exit 1; }
fi

# Single source of truth: lift REQUIRED_PATHS out of setup_environment.sh.
REQUIRED_PATHS=()
while IFS= read -r p; do
    [ -n "$p" ] && REQUIRED_PATHS+=("$p")
done < <(sed -n '/REQUIRED_PATHS=(/,/)/p' "$DATA_SH" | grep -oE '"[^"]+"' | tr -d '"')

if [ "${#REQUIRED_PATHS[@]}" -eq 0 ]; then
    echo "Error: could not parse any paths from REQUIRED_PATHS in $DATA_SH." >&2
    exit 1
fi

cd "$REPO_ROOT"

# Check presence.
present=(); missing=()
echo "Paths declared in REQUIRED_PATHS (${#REQUIRED_PATHS[@]}):"
for p in "${REQUIRED_PATHS[@]}"; do
    if [ -e "$p" ]; then
        size="$(du -shL "$p" 2>/dev/null | cut -f1)"
        printf "  [ok]   %-40s %s\n" "$p" "$size"
        present+=("$p")
    else
        printf "  [MISS] %s\n" "$p"
        missing+=("$p")
    fi
done
echo

if [ "${#missing[@]}" -gt 0 ] && [ "$ALLOW_MISSING" -eq 0 ]; then
    echo "Error: ${#missing[@]} required path(s) missing. Aborting (pass --allow-missing to override)." >&2
    exit 1
fi
[ "${#missing[@]}" -gt 0 ] && echo "Warning: bundling WITHOUT ${#missing[@]} missing path(s) (--allow-missing)."

if [ "$LIST_ONLY" -eq 1 ]; then
    total="$(du -shLc "${present[@]}" 2>/dev/null | tail -1 | cut -f1)"
    echo "Dry run (--list): would bundle ${#present[@]} path(s), ~${total} uncompressed -> $OUT_ZIP"
    exit 0
fi

# Build a clean archive (remove any stale one first so no orphan entries linger).
rm -f "$OUT_ZIP"
echo "Creating $OUT_ZIP ..."
# -r recurse; symlinks are followed (contents stored) by default — what we want
# for a distributable archive. Drop editor/OS/python cruft.
zip -r "$OUT_ZIP" "${present[@]}" -x '*.DS_Store' '*__pycache__*' '*.pyc'

echo
echo "Done: $OUT_ZIP ($(du -h "$OUT_ZIP" | cut -f1))"
echo "Inspect with : unzip -l \"$OUT_ZIP\""
echo "Extracts via : sh_scripts/unzip_data.sh   (or unzip -o data.zip -d .)"
