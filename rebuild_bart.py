"""Rebuild the BART transcripts from Frey's raw data, using the wording provided by Psych-201.

Inputs
  data__main__bart__bart_pumps.csv     csv of raw data from the actual study, one row per balloon (Frey et al., 2017)
  psych201_frey.parquet                instructions and sentence format from Psych-201, basically it is the description of experiments in natural language together with participant responses.
  psych101_bart_ids.csv                link between Psych-201 participant numbers (0, 1, 2, ...) and Frey participant IDs (64000401, ...)

Output
  transcripts_rebuilt/bart_transcripts.csv     columns: partid, text

Expected outcome
  Frey CSV: 1507 people | Psych-201: 1407 people (because Psych-201 holds-out a test set, different set of people for each of the 5 tasks.)
  different key pairs in Psych-201: 580
  identical to Psych-201 (keys replaced by H/W): 1407 of 1407
  saved 1507 people -> .../transcripts_rebuilt/bart_transcripts.csv

Every participant inflates 30 balloons, one pump at a time. In the raw data, "trial" is the balloon,
"pumps" the number of pumps, and "exploded" is 1 if the balloon burst.
Balloons are numbered 1, 2, 3, ... in the transcript, in the order of "trial".
Some balloons are missing from the raw data (no row, or no pumps): 1427 participants have 30 balloons,
70 have 29, 9 have 28 and 1 has 27. These participants are kept with the balloons they have.

Psych-201 uses different response keys for different participants (580 different pairs, e.g. Q/A or V/U).
We use the same keys for everyone: <<H>> for a pump and <<W>> for stopping. To compare with Psych-201,
we replace each participant's Psych-201 keys with H and W; apart from the keys, the transcripts are identical.

  If any transcript differs from Psych-201, the first difference is printed and nothing is saved.
"""
import os
import re
import pandas as pd

B = os.path.expanduser("~/bbrs")
RDIR_T = os.path.expanduser("~/hcp_explore/data/transcripts_rebuilt")
os.makedirs(RDIR_T, exist_ok=True)

# ---- Psych-201, linked to Frey IDs ----
P = pd.read_parquet("/scratch/sdemirka/psych201_frey.parquet")
bart = P[P.experiment == "frey2017risk/exp1.csv"]
L = pd.read_csv(f"{B}/psych101_bart_ids.csv")
to_frey = dict(zip(L.participant.astype(int), L.partid.astype(int)))          # Psych-201 number -> Frey ID
PSY = pd.Series({to_frey[n]: t for n, t in zip(bart.participant.astype(int), bart.text) if n in to_frey})

# ---- raw data ----
BR = pd.read_csv(f"{B}/data__main__bart__bart_pumps.csv").dropna(subset=["pumps"]).sort_values(["partid", "trial"])
GB = dict(tuple(BR.groupby("partid")))
print(f"Frey CSV: {len(GB)} people | Psych-201: {len(PSY)} people")

# ---- keys: Psych-201 uses different letters per person ----
PUMP = re.compile(r"pump up the balloon by pressing (\w)")
STOP = re.compile(r"stop pumping up the balloon by pressing (\w)")
print("different key pairs in Psych-201:", PSY.map(lambda t: (PUMP.search(t).group(1), STOP.search(t).group(1))).nunique())


def to_HW(text):
    """replace this person's Psych-201 keys with H (pump) and W (stop), in one pass"""
    pump, stop = PUMP.search(text).group(1), STOP.search(text).group(1)
    text = PUMP.sub("pump up the balloon by pressing H", text)
    text = STOP.sub("stop pumping up the balloon by pressing W", text)
    return re.sub(r"<<(\w+)>>", lambda m: "<<H>>" if m.group(1) == pump else "<<W>>" if m.group(1) == stop else m.group(0), text)


# ---- 1. wording from Psych-201 (with H/W) ----
x = to_HW(PSY.iloc[0])
INSTR = x[:x.index("Balloon 1:")]                                  # instructions + blank lines
END = x[len(x.rstrip("\n")):]                                      # line breaks at the very end, if any


# ---- 2. content from the Frey CSV ----
def build_bart(pid):
    balloons = []
    for k, r in enumerate(GB[pid].itertuples(), 1):                 # balloons numbered 1, 2, 3, ...
        n = int(r.pumps)
        if r.exploded == 1:
            balloons.append(f"Balloon {k}:\nYou press " + " ".join(["<<H>>"] * n)
                            + ". The balloon was inflated too much and explodes.")
        else:
            balloons.append(f"Balloon {k}:\nYou press " + " ".join(["<<H>>"] * n + ["<<W>>"])
                            + f". You stop inflating the balloon and get {n} points.")
    return INSTR + "\n\n".join(balloons) + END


# ---- 3. check: identical to Psych-201 once its keys are replaced by H/W ----
def first_diff(a, b):
    return next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))


common = [p for p in PSY.index if p in GB]
diff = [p for p in common if build_bart(p) != to_HW(PSY[p])]
print(f"\nidentical to Psych-201 (keys replaced by H/W): {len(common) - len(diff)} of {len(common)}")
if diff:
    pid = diff[0]; a, b = build_bart(pid), to_HW(PSY[pid]); i = first_diff(a, b)
    print(f"first difference for {pid} at character {i}:")
    print("REBUILT:  ", repr(a[max(0, i - 150):i + 150]))
    print("PSYCH-201:", repr(b[max(0, i - 150):i + 150]))

# ---- 4. save everyone, only if all match ----
if not diff:
    NEW = pd.DataFrame([dict(partid=int(p), text=build_bart(p)) for p in sorted(GB)])
    NEW.to_csv(f"{RDIR_T}/bart_transcripts.csv", index=False)
    print(f"saved {len(NEW)} people -> {RDIR_T}/bart_transcripts.csv")
