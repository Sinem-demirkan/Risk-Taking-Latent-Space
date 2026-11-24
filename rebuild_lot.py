"""Rebuild the LOT transcripts from Frey's raw data, using the wording provided by Psych-201.

Inputs
  data__main__lotteries__lotteries.csv     csv of raw data from the actual study, one row per choice (Frey et al., 2017)
  psych201_frey.parquet         instructions and sentence format from Psych-201, basically it is the description of experiments in natural language together with participant responses.

Output
  transcripts_rebuilt/lot_transcripts.csv     columns: partid, text

Expected outcome
  Frey CSV: 1507 people | Psych-201: 1407 people (because Psych-201 holds-out a test set, different set of people for each of the 5 tasks.)
  excluded: 0 [] | kept: 1507
  identical to Psych-201: 1407 of 1407
  saved 1507 people -> .../transcripts_rebuilt/lot_transcripts.csv

Every participant sees 25 lottery pairs. The task is adaptive: each pair is shown several times
(84 choices in total, according to Frey et al.), and after every choice Lottery A (X) is changed
a bit depending on what the person picked, while Lottery B (Z) stays the same.
The data only keeps the two times each pair was shown, so there are 2 rows per pair and
50 choices per person. The "Stage" column tells you which step it was: the lower number is the
earlier one, the higher number is the changed version.
  
  A participant is excluded if a value is missing, a pair does not have exactly 2 rows, or there are not 25 pairs.
  If any transcript differs from Psych-201, the first difference is printed and nothing is saved.
"""
import os
import pandas as pd

B = os.path.expanduser("~/bbrs")
RDIR_T = os.path.expanduser("~/hcp_explore/data/transcripts_rebuilt")
os.makedirs(RDIR_T, exist_ok=True)

# ---- the two inputs ----
P = pd.read_parquet("/scratch/sdemirka/psych201_frey.parquet")
PSY = P[P.experiment == "frey2017lotteries/"].assign(partid=lambda d: d.participant.astype(int)).set_index("partid").text
LR = pd.read_csv(f"{B}/data__main__lotteries__lotteries.csv").sort_values(["partid", "Dec_ID", "Stage"])
print(f"Frey CSV: {LR.partid.nunique()} people | Psych-201: {len(PSY)} people")

# ---- exclude participants with incomplete data ----
cols = ["X1", "X2", "PX1", "Z1", "Z2", "PZ1", "Decision_X"]
per_pair = LR.groupby(["partid", "Dec_ID"]).size()
bad = set(LR.loc[LR[cols].isna().any(axis=1), "partid"])
bad |= set(per_pair[per_pair != 2].reset_index().partid)
bad |= set(LR.groupby("partid").Dec_ID.nunique().loc[lambda s: s != 25].index)
LR = LR[~LR.partid.isin(bad)]
GL = dict(tuple(LR.groupby("partid")))
print(f"excluded: {len(bad)} {sorted(bad)} | kept: {len(GL)}")

# ---- 1. wording from Psych-201 ----
x = PSY.iloc[0]
INSTR = x[:x.index("Problem 1:")]                                  # instructions + blank lines
p1 = x[x.index("Problem 1:"):x.index("Problem 2:")]
last_choice = [l for l in p1.split("\n") if "<<" in l][-1]
SEP = p1[p1.rindex(last_choice) + len(last_choice):]               # blank lines between pairs


def gamble(o1, o2, p1):            # "gain 50 points with a 70% chance or gain 20 points with a 30% chance."
    part = lambda o, p: f"{'gain' if o >= 0 else 'lose'} {abs(o):g} points with a {p:g}% chance"
    return f"{part(o1, p1)} or {part(o2, 100 - p1)}."


# ---- 2. content from the Frey CSV ----
def build_lot(pid):
    blocks = []
    for k, g in GL[pid].groupby("Dec_ID"):                         # 25 pairs
        versions = [f"Lottery A: {gamble(r.X1, r.X2, r.PX1)}\n"    # X = Lottery A
                    f"Lottery B: {gamble(r.Z1, r.Z2, r.PZ1)}\n"    # Z = Lottery B
                    f"You chose <<{'A' if r.Decision_X == 1 else 'B'}>>."
                    for r in g.itertuples()]                       # 2 steps per pair
        blocks.append(f"Problem {int(k)}:\n" + "\n\n".join(versions))
    return INSTR + SEP.join(blocks) + "\n"                         # Psych-201 ends with a line break


# ---- 3. check against Psych-201 ----
def first_diff(a, b):
    return next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))


common = [p for p in PSY.index if p in GL]
diff = [p for p in common if build_lot(p) != PSY[p]]
print(f"\nidentical to Psych-201: {len(common) - len(diff)} of {len(common)}")
if diff:
    pid = diff[0]; a, b = build_lot(pid), PSY[pid]; i = first_diff(a, b)
    print(f"first difference for {pid} at character {i}:")
    print("REBUILT:", repr(a[max(0, i - 120):i + 120]))
    print("PSYCH-201:", repr(b[max(0, i - 120):i + 120]))

# ---- 4. save everyone, only if all match ----
if not diff:
    NEW = pd.DataFrame([dict(partid=int(p), text=build_lot(p)) for p in sorted(GL)])
    NEW.to_csv(f"{RDIR_T}/lot_transcripts.csv", index=False)
    print(f"saved {len(NEW)} people -> {RDIR_T}/lot_transcripts.csv")
