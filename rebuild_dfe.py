"""Rebuild the DFE transcripts from Frey's raw data, using the wording provided by Psych-201.

Inputs
  data__main__dfe__dfe_samples.csv     csv of raw data from the actual study, one row per sample (Frey et al., 2017)
  psych201_frey.parquet                instructions and sentence format from Psych-201, basically it is the description of experiments in natural language together with participant responses.

Output
  transcripts_rebuilt/dfe_transcripts.csv     columns: partid, text

Expected outcome
  Frey CSV: 1507 people | Psych-201: 1407 people (because Psych-201 holds-out a test set, different set of people for each of the 5 tasks.) | in both: 1407
  problems chosen without sampling: 38 (from 21 participants)
  identical to Psych-201: 1390 of 1407
  different: 17 | all of them people who chose without sampling: True
  saved 1507 people -> .../transcripts_rebuilt/dfe_transcripts.csv

Every participant sees the same 8 problems, in a different order. In each problem they sample from two boxes
as often as they like, then make one final choice. In the raw data, "gamble_ind" is the position of the problem,
"gamble_lab" which problem it is, "sample_ind" the sample number, "sample_opt" the box sampled, "sample_out"
the outcome seen, and "decision" the final choice (repeated on every row of the problem).

Correction: in 38 problems (21 participants), the participant chose a box without sampling first. The raw data
marks these problems with an empty row (sample_ind = 0, no side, no box, outcome 0, reaction time 0).
Psych-201 turns this empty row into a sample ("You decided to sample Box B. You observed an outcome of 0 points.").
We write only the final choice. For the 17 of these participants who are in Psych-201, our transcript therefore
differs from theirs; all other transcripts are identical.

  If a transcript differs from Psych-201 for any other reason, the first difference is printed and nothing is saved.
"""
import os
import re
import pandas as pd

B = os.path.expanduser("~/bbrs")
RDIR_T = os.path.expanduser("~/hcp_explore/data/transcripts_rebuilt")
os.makedirs(RDIR_T, exist_ok=True)

# ---- the two inputs ----
P = pd.read_parquet("/scratch/sdemirka/psych201_frey.parquet")
dfe = P[P.experiment == "frey2017dfe/"]
PSY = dfe.assign(partid=dfe.participant.astype(int)).set_index("partid").text
DS = pd.read_csv(f"{B}/data__main__dfe__dfe_samples.csv").sort_values(["partid", "gamble_ind", "sample_ind"])
G = dict(tuple(DS.groupby("partid")))
print(f"Frey CSV: {len(G)} people | Psych-201: {len(PSY)} people | in both: {len(set(PSY.index) & set(G))}")

# ---- problems chosen without sampling (sample_ind = 0: an empty row, only the final choice is real) ----
no_sampling = DS[DS.sample_ind == 0]
print(f"problems chosen without sampling: {len(no_sampling)} (from {no_sampling.partid.nunique()} participants)")

# ---- 1. wording from Psych-201 ----
x = PSY.iloc[0]
INSTR = x[:x.index("Problem 1:")]                                  # instructions + blank lines
p1 = x[x.index("Problem 1:"):x.index("Problem 2:")]
final_line = [l for l in p1.split("\n") if "<<choose>>" in l][0]
SEP = p1[p1.index(final_line) + len(final_line):]                  # between two problems
FINAL = re.sub(r"<<[AB]>>", "<<{box}>>", final_line)               # final choice, box left open
END = x[len(x.rstrip("\n")):]                                      # line breaks at the very end, if any


# ---- 2. content from the Frey CSV ----
def build_dfe(pid):
    blocks = []
    for gi, s in G[pid].groupby("gamble_ind"):                     # problems in the order the person saw them
        real = s[s.sample_ind > 0]                                 # sample_ind 0 = chose without sampling
        lines = [f"You decided to <<sample>> Box <<{o}>>. You observed an outcome of {float(v):g} points."
                 for o, v in zip(real.sample_opt, real.sample_out)]
        lines.append(FINAL.format(box=s.decision.iloc[0]))
        blocks.append(f"Problem {int(gi)}:\n" + "\n".join(lines))  # keep the original problem number
    return INSTR + SEP.join(blocks) + END


# ---- 3. check against Psych-201 ----
def first_diff(a, b):
    return next((i for i, (c, d) in enumerate(zip(a, b)) if c != d), min(len(a), len(b)))


common = [p for p in PSY.index if p in G]
diff = [p for p in common if build_dfe(p) != PSY[p]]
expected = set(no_sampling.partid)
unexpected = [p for p in diff if p not in expected]
print(f"\nidentical to Psych-201: {len(common) - len(diff)} of {len(common)}")
print(f"different: {len(diff)} | all of them people who chose without sampling: {not unexpected}")
if unexpected:
    pid = unexpected[0]; a, b = build_dfe(pid), PSY[pid]; i = first_diff(a, b)
    print(f"first unexpected difference for {pid} at character {i}:")
    print("REBUILT:  ", repr(a[max(0, i - 120):i + 120]))
    print("PSYCH-201:", repr(b[max(0, i - 120):i + 120]))

# ---- 4. save everyone, only if the only differences are the no-sampling corrections ----
if not unexpected:
    NEW = pd.DataFrame([dict(partid=int(p), text=build_dfe(p)) for p in sorted(G)])
    assert not NEW.text.str.contains("<<nan>>|<<None>>").any()
    NEW.to_csv(f"{RDIR_T}/dfe_transcripts.csv", index=False)
    print(f"saved {len(NEW)} people -> {RDIR_T}/dfe_transcripts.csv")
