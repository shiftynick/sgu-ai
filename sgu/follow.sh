#!/bin/bash
# Segment and extract (two Haiku runs) each episode as soon as its transcript exists, until all are done.
cd "$(dirname "$0")/.."
todo=$(cat "$1")
while :; do
  left=0
  for n in $todo; do
    [ -f data/tx/$n.json ] || { left=1; continue; }
    [ -f data/seg/$n.json ] || [ -f data/seg/$n.none ] || { python3 -I sgu/segment.py $n | grep -q "NO SEGMENT" && touch data/seg/$n.none; }
    [ -f data/seg/$n.json ] || continue
    [ -f data/extract/haiku/$n.json ] || .venv/bin/python -I sgu/extract.py haiku $n &
    [ -f data/extract/haiku_b/$n.json ] || .venv/bin/python -I sgu/extract.py haiku_b $n &
    wait
  done
  [ $left = 0 ] && break
  sleep 30
done
echo all done
