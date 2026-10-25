# sgu-ai

Can you hear the answer in the host's voice?

In the Science or Fiction segment of [The Skeptics' Guide to the Universe](https://www.theskepticsguide.org/), Steve Novella reads three items, one of them fake, and the rogues take turns guessing which is the fiction. After each guess he hands off to the next person: "Okay, Jay." He knows the answer. This project asks whether the way he says that "Okay" gives away whether the guess he just heard was right.

## What it does

`sgu/` is a pipeline that builds a labeled dataset from the podcast archive, then (next) trains a classifier on it.

For each episode it:

1. Downloads the episode and transcribes the second half with Whisper.
2. Finds the Science or Fiction segment.
3. Has Claude Haiku extract each rogue's pick, twice independently. It then checks the picks against Steve's recap during the reveal and against the answer key on theskepticsguide.org.
4. Finds Steve's "Okay/All right, <name>" handoffs and labels each one by whether the guess before it was correct.
5. Aligns each cue precisely, checks it's the host's voice, and cuts a clip of just the cue word.

`sgu/README.md` has the step-by-step commands. `python3 -I sgu/run.py start|stop|status` runs the whole archive in resumable background batches.

## Status

- **Extraction accuracy:** on 10 hand-labeled episodes, Haiku 5.5 got 39 of 39 picks right. A local Qwen3.5-9B got 33 of 39.
- **Full-archive run:** in progress. Before it started, a 109-episode sample gave 195 usable clips.
- **Classifier:** not built yet.

## Setup notes

- Transcription uses faster-whisper `large-v3-turbo`, through a separate environment (`sgu/run_whisper.sh` points at it; adjust the path for your machine).
- Everything else runs in `.venv` (see `requirements.txt`).
- Haiku runs through the Claude Code CLI (`claude -p`), so no API key is needed if you're signed in.

## Data

No audio, transcripts or clips are in this repo. They are the show's copyrighted content. The pipeline downloads episodes from the public RSS feed into `data/`, which is git-ignored.
