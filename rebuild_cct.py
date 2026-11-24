"""Rebuild the CCT transcripts from Frey's raw data, using the wording provided by Psych-201.

Inputs
  data__main__cct__cct.csv     csv of raw data from the actual study, one row per round (Frey et al., 2017)
  psych201_frey.parquet        instructions and sentence format from Psych-201, basically it is the description of experiments in natural language together with participant responses.
  psych101_cct_ids.csv         link between Psych-201 participant numbers (0, 1, 2, ...) and Frey participant IDs (64000401, ...)

Output
  transcripts_rebuilt/cct_transcripts.csv     columns: partid, text

Expected outcome
  Frey CSV: 1506 people | Psych-201: 1406 people (because Psych-201 holds-out a test set, different set of people for each of the 5 tasks.)
  different key pairs in Psych-201: 581
  identical to Psych-201 (keys replaced by E/C): 1406 of 1406
  saved 1506 people -> .../transcripts_rebuilt/cct_transcripts.csv

Every participant plays 84 rounds in 4 blocks of 21. In each round there are 32 face-down cards, some of them loss cards.
In the raw data, "r_trialnum" is the round, "r_block" the block, "r_winvalue" the points of a gain card,
"r_lossvalue" the points lost on a loss card, "r_lossnum" the number of loss cards, "r_cardschosen" the number of
cards turned over, and "r_censored" the position of the loss card if the participant hit one (0 if not).
If the participant hit a loss card, it is the last card turned over and the round ends there.
Otherwise the round ends with the participant stopping, also when all gain cards were turned over.
Psych-201 puts an extra blank line before the first round of a new block, and so do we.

Psych-201 uses different response keys for different participants (581 different pairs, e.g. A/G or O/C).
We use the same keys for everyone: <<E>> to turn a card over and <<C>> to stop. To compare with Psych-201,
we replace each participant's Psych-201 keys with E and C; apart from the keys, the transcripts are identical.

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
cct = P[P.experiment == "frey2017cct/exp1.csv"]
L = pd.read_csv(f"{B}/psych101_cct_ids.csv")
to_frey = dict(zip(L.participant.astype(int), L.partid.astype(int)))          # Psych-201 number -> Frey ID
PSY = pd.Series({to_frey[n]: t for n, t in zip(cct.participant.astype(int), cct.text) if n in to_frey})

# ---- raw data ----
C = pd.read_csv(f"{B}/data__main__cct__cct.csv").dropna(subset=["r_cardschosen", "r_winvalue", "r_lossvalue", "r_lossnum"])
C = C.sort_values(["partid", "r_trialnum"])
GC = dict(tuple(C.groupby("partid")))
print(f"Frey CSV: {len(GC)} people | Psych-201: {len(PSY)} people")

# ---- keys: Psych-201 uses different letters per person ----
KEYS = re.compile(r"Press (\w) to turn a card over, or (\w) to stop")
print("different key pairs in Psych-201:", PSY.map(lambda t: KEYS.search(t).groups()).nunique())


def to_EC(text):
    """replace this person's Psych-201 keys with E (turn) and C (stop), in one pass"""
    turn, stop = KEYS.search(text).groups()
    text = text.replace(f"Press {turn} to turn a card over, or {stop} to stop", "Press E to turn a card over, or C to stop")
    return re.sub(r"<<(\w+)>>", lambda m: "<<E>>" if m.group(1) == turn else "<<C>>" if m.group(1) == stop else m.group(0), text)


# ---- 1. wording from Psych-201 (with E/C) ----
x = to_EC(PSY.iloc[0])
INSTR = x[:x.index("Round 1:")]                                  # instructions + blank lines
END = x[len(x.rstrip("\n")):]                                      # line breaks at the very end, if any


# ---- 2. content from the Frey CSV ----
def build_cct(pid):
    out, prev_block = INSTR, None
    for k, r in enumerate(GC[pid].itertuples(), 1):
        gn, ls, nl, cards = int(r.r_winvalue), int(r.r_lossvalue), int(r.r_lossnum), int(r.r_cardschosen)
        hit = r.r_censored > 0                                     # they turned over a loss card
        t = (f"Round {k}:\nYou will be awarded {gn} points for turning over a gain card.\n"
             f"You will lose {ls} points for turning over a loss card.\nThere are {nl} loss cards in this round.\n")
        score, n_gain = 0, cards - 1 if hit else cards
        for _ in range(n_gain):
            score += gn; t += f"You press <<E>> and turn over a gain card. Your current score is {score}.\n"
        if hit:
            score -= ls
            t += (f"You press <<E>> and turn over a loss card. Your current score is {score}. "
                  f"The round has now ended because you encountered a loss card.\n")
        else:                                                      # no loss card: the round ends with a stop
            t += "You press <<C>> and claim your payout.\n"
        t += f"Your final score for this round is {score}."
        if prev_block is not None:
            out += "\n\n\n" if r.r_block != prev_block else "\n\n"  # extra blank line when a new block starts
        out += t
        prev_block = r.r_block
    return out + END


# ---- 3. check: identical to Psych-201 once its keys are replaced by E/C ----
def first_diff(a, b):
    return next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))


common = [p for p in PSY.index if p in GC]
diff = [p for p in common if build_cct(p) != to_EC(PSY[p])]
print(f"\nidentical to Psych-201 (keys replaced by E/C): {len(common) - len(diff)} of {len(common)}")
if diff:
    pid = diff[0]; a, b = build_cct(pid), to_EC(PSY[pid]); i = first_diff(a, b)
    print(f"first difference for {pid} at character {i}:")
    print("REBUILT:  ", repr(a[max(0, i - 150):i + 150]))
    print("PSYCH-201:", repr(b[max(0, i - 150):i + 150]))

# ---- 4. save everyone, only if all match ----
if not diff:
    NEW = pd.DataFrame([dict(partid=int(p), text=build_cct(p)) for p in sorted(GC)])
    NEW.to_csv(f"{RDIR_T}/cct_transcripts.csv", index=False)
    print(f"saved {len(NEW)} people -> {RDIR_T}/cct_transcripts.csv")
