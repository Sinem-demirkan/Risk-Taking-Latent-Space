"""Rebuild the MPL transcripts from Frey's raw data, using the wording provided by Psych-201.

Inputs
  data__main__mpl__mplProblems.csv     csv of the task design: the 66 rows of the 7 lists, with the outcomes and probabilities of lottery A and B
  data__main__mpl__mpl.csv             csv of raw data from the actual study, one row per choice (Frey et al., 2017)
  psych201_frey_mpl_fixed.parquet      Psych-201 MPL transcripts after our correction, used for the instructions, the sentence format and the check

Output
  transcripts_rebuilt/mpl_transcripts.csv     columns: partid, text

Expected outcome
  Frey CSV: 1507 people | corrected Psych-201: 1405 people (because Psych-201 holds-out a test set, different set of people for each of the 5 tasks.)
  excluded (missing rows): [N] [IDs] | kept: [N]
  identical to corrected Psych-201: 1405 of 1405
  saved [N] people -> .../transcripts_rebuilt/mpl_transcripts.csv

Every participant sees 7 lists, with 10, 7, 5, 16, 11, 11 and 6 rows (66 choices). In each row they choose
between lottery A and lottery B. Within a list, one lottery gets more attractive from row to row, and people
switch to it at some point. In the raw data, "dp" is the list, "decision" is the row, and "choice" is 1 for A, 0 for B.

Psych-201's own MPL transcripts have two errors: in list 1, gains are written as losses ("lose 1.6", "lose 0.1")
and 3.85 is written as 3.9, and in all lists the letters A and B of the chosen lottery are swapped.
The raw data has neither error, so we compare the rebuilt transcripts with the corrected Psych-201 file.

  A participant is excluded if one of their 66 choices is missing.
  If any transcript differs from the corrected Psych-201, the first difference is printed and nothing is saved.
"""
import os
import pandas as pd

B = os.path.expanduser("~/bbrs")
RDIR_T = os.path.expanduser("~/hcp_explore/data/transcripts_rebuilt")
os.makedirs(RDIR_T, exist_ok=True)

# ---- the inputs ----
PROB = pd.read_csv(f"{B}/data__main__mpl__mplProblems.csv").sort_values(["dp", "decision"])
CH = pd.read_csv(f"{B}/data__main__mpl__mpl.csv").dropna(subset=["choice"])
FIX = pd.read_parquet("/scratch/sdemirka/psych201_frey_mpl_fixed.parquet")
PSY = FIX.assign(partid=FIX.participant.astype(int)).set_index("partid").text
print(f"Frey CSV: {CH.partid.nunique()} people | corrected Psych-201: {len(PSY)} people")

# ---- exclude participants with missing choices ----
n_choices = CH.groupby("partid").size()
bad = set(n_choices[n_choices != len(PROB)].index)
CH = CH[~CH.partid.isin(bad)]
C = {pid: g.set_index(["dp", "decision"]).choice for pid, g in CH.groupby("partid")}
print(f"excluded (missing rows): {len(bad)} {sorted(bad)} | kept: {len(C)}")

# ---- 1. wording from Psych-201 ----
x = PSY.iloc[0]
INSTR = x[:x.index("Problem 1:")]                                  # instructions + blank lines
END = x[len(x.rstrip("\n")):]                                      # line breaks at the very end, if any


def side(out, p):                                                  # "10% chance to gain 2 points"
    return f"{p * 100:g}% chance to {'gain' if out >= 0 else 'lose'} {abs(out):g} points"


def lottery(r, L):                                                 # "Lottery A: ... and ..."
    return f"Lottery {L}: {side(r[f'{L}_out1'], r[f'{L}_p1'])} and {side(r[f'{L}_out2'], r[f'{L}_p2'])}."


# ---- 2. content from the Frey CSVs ----
def build_mpl(pid):
    c = C[pid]
    lists = []
    for dp, rows in PROB.groupby("dp"):                            # the 7 lists
        lines = [f"{lottery(r, 'A')}\n{lottery(r, 'B')}\n"
                 f"You chose lottery <<{'A' if c[(dp, r.decision)] == 1 else 'B'}>>."
                 for _, r in rows.iterrows()]                      # one choice per row
        lists.append(f"Problem {dp}:\n" + "\n\n".join(lines))
    return INSTR + "\n\n\n".join(lists) + END


# ---- 3. check against the corrected Psych-201 ----
def first_diff(a, b):
    return next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))


common = [p for p in PSY.index if p in C]
diff = [p for p in common if build_mpl(p) != PSY[p]]
print(f"\nidentical to corrected Psych-201: {len(common) - len(diff)} of {len(common)}")
if diff:
    pid = diff[0]; a, b = build_mpl(pid), PSY[pid]; i = first_diff(a, b)
    print(f"first difference for {pid} at character {i}:")
    print("REBUILT:  ", repr(a[max(0, i - 120):i + 120]))
    print("PSYCH-201:", repr(b[max(0, i - 120):i + 120]))

# ---- 4. save everyone, only if all match ----
if not diff:
    NEW = pd.DataFrame([dict(partid=int(p), text=build_mpl(p)) for p in sorted(C)])
    NEW.to_csv(f"{RDIR_T}/mpl_transcripts.csv", index=False)
    print(f"saved {len(NEW)} people -> {RDIR_T}/mpl_transcripts.csv")
