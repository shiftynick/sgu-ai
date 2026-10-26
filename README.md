# sgu-ai

Can you hear the answer in the host's voice?

In the Science or Fiction segment of [The Skeptics' Guide to the Universe](https://www.theskepticsguide.org/), Steve Novella reads three items, one of them fake, and the rogues take turns guessing which is the fiction. After each guess he hands off to the next person: "Okay, Jay." He knows the answer. This project asks whether his voice gives it away: whether the way he says that "Okay" reveals if the guess he just heard was right, and whether the way he reads the three items reveals which one is fake.

## What it does

`sgu/` is a pipeline that builds labeled datasets from the podcast archive and tests classifiers on them.

For each episode it:

1. Downloads the episode and transcribes the second half with Whisper.
2. Finds the Science or Fiction segment.
3. Has Claude Haiku extract each rogue's pick, twice independently. It then checks the picks against Steve's recap during the reveal and against the answer key on theskepticsguide.org.
4. Finds Steve's "Okay/All right, <name>" handoffs and labels each one by whether the guess before it was correct.
5. Aligns each cue precisely, checks it's the host's voice, and cuts a clip of just the cue word.
6. Separately, finds each of the three item readings by matching the item text from the show's website.

The classifiers use pitch, loudness and pacing measured against the host's own voice in the same episode, plus embeddings from two pretrained speech models (WavLM, and audeering's emotion model trained on podcasts). Everything is cross-validated with whole episodes held out, and compared against a baseline that knows nothing about the audio.

`sgu/README.md` has the step-by-step commands. `python3 -I sgu/run.py start|stop|status` runs the whole archive in resumable background batches.

## Results

From the full archive: 1,109 episodes, of which 795 gave usable "Okay" clips and 886 gave usable item readings.

**The fiction is in how he reads it.** From his voice reading the three items, a model picks the fiction 49% of the time, against 33% for guessing. The item's wording alone gets about 40%. The voice still adds the same amount with the wording model included, and the two predictions are almost unrelated, so it isn't picking up the text. He reads the fiction slightly faster and more smoothly, with flatter pitch, steadier loudness and a little less energy. The tell was strongest in 2005-2009 (61%) and is still there in every later era (41-51%).

**The "Okay" tell is real but faint.** After a correct guess, "Okay, <name>" sounds slightly more energetic and positive. Adding the audio raises the AUC from 0.60 (position, cue word, year and rogue alone) to about 0.63 (p = 0.005). That holds across positions and eras but is far too weak to use. Fine-tuning the speech models didn't improve on it, and pitch, timing and his words found nothing.

Extraction accuracy: on 10 hand-labeled episodes, Haiku 5.5 got 39 of 39 picks right. A local Qwen3.5-9B got 33 of 39.

## Setup notes

- Transcription uses faster-whisper `large-v3-turbo`, through a separate environment (`sgu/run_whisper.sh` points at it; adjust the path for your machine).
- Everything else runs in `.venv` (see `requirements.txt`).
- Haiku runs through the Claude Code CLI (`claude -p`), so no API key is needed if you're signed in.

## Data

No audio, transcripts or clips are in this repo. They are the show's copyrighted content. The pipeline downloads episodes from the public RSS feed into `data/`, which is git-ignored.
