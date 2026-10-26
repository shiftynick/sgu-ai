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

## Classifiers

Steps 11-14 work on the "Okay" clips; 15-16 on the item readings.

11. `.venv/bin/python -I sgu/features.py` measures each clip against the host's own reading voice in that episode, cached in `data/feat/`. It records pitch, loudness and timing, how much the host spoke during the rogue's turn, WavLM and emotion-model embeddings of the cue and of "cue + name", and the emotion model's arousal/dominance/valence.
12. `.venv/bin/python -I sgu/classify.py [--perms 200]` cross-validates classifiers by episode and writes `data/classify_report.txt`.
    - `meta` is a no-audio baseline (position, cue word, year, rogue), because later rogues are right more often and "All right" is more common after a correct guess.
    - Each audio or text feature set is tested on whether it improves on `meta`. The test shuffles that feature set between clips with the same position, cue word and era.
    - A second table uses only episodes with both a right and a wrong guess before a cue, comparing the two within the episode.
13. `.venv/bin/python -I sgu/eras.py` repeats the main analysis per era (`data/eras_report.txt`).
14. `.venv/bin/python -I sgu/finetune.py --backbone wavlm|emotion [--clip cue_name|cue]` fine-tunes a speech model on the same folds (`data/finetune_report.txt`).
15. `.venv/bin/python -I sgu/readings.py` finds the three item readings in each episode by matching the website's item text and measures them (`data/readings.csv`, `data/readings_emb.npz`).
16. `.venv/bin/python -I sgu/readings_classify.py [--perms 200]` tests how often each feature set picks the fiction, with and without a text-only model in the baseline, and per era (`data/readings_report.txt`).

`python3 -I sgu/listen.py` is a listening test for people. It plays random "Okay" clips at http://localhost:8765 and saves answers to `data/listen.csv`.

Other scripts:
- `compare.py` scores extraction backends against the hand labels in `data/reference.json`.
- `extract.py qwen` uses the local Qwen3.5-9B server (stop it before running Whisper).
- `extract.py decisions` needs `OPENAI_API_KEY` and has not been run yet.
