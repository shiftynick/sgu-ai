# SGU "Okay, <rogue>" pipeline

## Running the whole archive

- `python3 -I sgu/run.py start [--limit N] [--batch 25]` starts or resumes the run in the background.
- `python3 -I sgu/run.py stop` stops it after the current episode; `stop --now` kills it, and unfinished episodes are redone on the next start.
- `python3 -I sgu/run.py status` shows progress and dataset size.

Finished and skipped episodes are recorded in `data/run/done/`. The logs are `data/run/run.log` and `data/run/steps.log`, and the dataset is rebuilt into `data/dataset.csv` after every batch. The runner calls the steps below.

## Steps

Run these from the project folder (the parent of `sgu/`). Each step skips episodes it has already done. The episode lists are `data/all_episodes.txt`, `data/batch2.txt` and so on.

1. `python3 -I sgu/answers.py <episodes>` scrapes the answer key from theskepticsguide.org into `data/episodes.json`. Early episode pages have no item text; for those the fiction comes from the host's reveal instead.
2. `python3 -I sgu/download.py <episodes>` downloads the MP3s in parallel.
3. `sgu/run_whisper.sh sgu/transcribe.py [--full] <episodes>` transcribes the second half of each episode into `data/tx/`. Use `--full` for early episodes, where Science or Fiction comes near the start.
4. `python3 -I sgu/segment.py <episodes>` finds the Science or Fiction segment and writes it to `data/seg/`.
5. `.venv/bin/python -I sgu/extract.py haiku <episodes>` extracts each rogue's pick; run it again with `haiku_b` for a second, independent run.
   - Both run Haiku 5.5 through `claude -p`, so they bill to the Claude subscription.
   - `sgu/follow.sh <list>` runs steps 4 and 5 as transcripts appear.
6. `python3 -I sgu/consensus.py haiku haiku_b <episodes>` keeps the episodes where both runs agree. A third run in `data/extract/haiku_tiebreak/` breaks ties.
7. `.venv/bin/python -I sgu/recap.py <episodes>` is an independent Haiku pass over the host's recap. It writes `data/recap_check.csv` and flags picks the recap contradicts.
8. `python3 -I sgu/handoffs.py consensus <episodes>` finds the cue and name handoffs.
   - A handoff only counts if it sits between the two rogues' stated picks and the previous rogue's own turn was confirmed by a handoff into it.
9. `.venv/bin/python -I sgu/cut.py consensus` re-aligns the cues, runs the host-voice check and cuts the clips.
10. `sgu/run_whisper.sh sgu/check_clips.py > data/check_clips.log` re-transcribes the clips. Then `python3 -I sgu/clip_text.py consensus` sets the final usable flag, and `python3 -I sgu/dataset.py` builds `data/dataset.csv`.

Other scripts:
- `compare.py` scores extraction backends against the hand labels in `data/reference.json`.
- `extract.py qwen` uses the local Qwen3.5-9B server (stop it before running Whisper).
- `extract.py decisions` needs `OPENAI_API_KEY` and has not been run yet.
