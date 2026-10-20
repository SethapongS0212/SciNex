#!/usr/bin/env bash
# fetch_corpus.sh — download the open-access papers of the handpicked 20-paper corpus.
#
# Reads papers/manifest.csv and downloads every row marked `oa` into papers/<paper_id>.pdf.
# The PDF stem becomes the paper id used by the whole pipeline (output/<paper_id>/...).
#
# Idempotent: an existing valid PDF is left alone. Anything that comes back as HTML
# (publisher bot-block, login wall) is deleted rather than left as a fake PDF, because
# main.py would otherwise "parse" it into garbage.
#
# Rows marked `oa_blocked` are open access but served behind a bot-blocker
# (IEEE Xplore, ScienceDirect) — they must be downloaded by hand from the listed URL.
# Rows marked `paywalled` need institutional access.
#
# Usage:  bash fetch_corpus.sh

set -u
cd "$(dirname "$0")"
MANIFEST="papers/manifest.csv"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

[ -f "$MANIFEST" ] || { echo "Missing $MANIFEST"; exit 1; }
mkdir -p papers

ok=0; skip=0; fail=0
manual=""

while IFS=, read -r paper_id year venue doi access pdf_url rest; do
    [ "$paper_id" = "paper_id" ] && continue
    [ -z "$paper_id" ] && continue

    if [ "$access" != "oa" ]; then
        manual="${manual}  ${paper_id}  [${access}]  ${pdf_url}\n"
        continue
    fi

    dest="papers/${paper_id}.pdf"
    if [ -s "$dest" ] && [ "$(head -c 4 "$dest")" = "%PDF" ]; then
        echo "SKIP  $paper_id (already downloaded)"
        skip=$((skip + 1))
        continue
    fi

    curl -sSL --max-time 120 -A "$UA" -o "$dest" "$pdf_url" 2>/dev/null
    if [ -s "$dest" ] && [ "$(head -c 4 "$dest")" = "%PDF" ]; then
        echo "OK    $paper_id  $(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest") bytes"
        ok=$((ok + 1))
    else
        echo "FAIL  $paper_id  ($pdf_url)"
        rm -f "$dest"
        fail=$((fail + 1))
    fi
done < "$MANIFEST"

echo
echo "Downloaded $ok, already present $skip, failed $fail"
if [ -n "$manual" ]; then
    echo
    echo "Download these by hand into papers/<paper_id>.pdf (name matters — it becomes the paper id):"
    printf "$manual"
fi
