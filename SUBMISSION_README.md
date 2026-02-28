# WiDS WorldWide Global Datathon 26 — First Submission

## File to submit on Kaggle

**Submit this file:** `submission.csv`

- **Location:** same folder as this README (`vyc/submission.csv`)
- **Format:** CSV with columns `event_id`, `prob_12h`, `prob_24h`, `prob_48h`, `prob_72h`
- **Rows:** 95 test rows + 1 header (96 lines total)
- **Model:** Best CV approach (e.g. alpha72-hybrid ~97.96% CV)

---

## How to submit (Kaggle)

1. **Open the competition**
   - Go to: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26

2. **Go to “Submit Predictions”**
   - Click the **“Submit Predictions”** (or **“Late Submission”**) button on the right.

3. **Upload your file**
   - Click **“Upload”** or **“Submit from file”**.
   - Choose **`submission.csv`** from the `vyc` folder.
   - Wait for the upload to finish.

4. **Submit**
   - Add a short description (e.g. `First submission - alpha72 hybrid`).
   - Click **“Make Submission”**.

5. **Check the leaderboard**
   - Your score will appear after processing (often 1–2 minutes).
   - Leaderboard: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/leaderboard?search=Bos_

---

## Optional: other submission variants

If you want to try different models on the leaderboard, you can submit these instead of `submission.csv`:

| File | Description |
|------|-------------|
| `submission_alpha72.csv` | Alpha72-hybrid (30% JH + 70% global @ 72h) |
| `submission_blend.csv` | Global optimized blend |
| `submission_tiebreak.csv` | GBC + small GBSA-rank tie-break |
| `submission_custom.csv` | Brier 12–48h, global 72h |
| `submission_mixed.csv` | Brier 12–48h, joint-hybrid 72h |

**For your first submission, use `submission.csv`** — it is the best approach chosen by the script (e.g. alpha72 when that was best).

---

## Checklist before submitting

- [ ] File is **submission.csv** (or one of the variants above).
- [ ] CSV has header: `event_id,prob_12h,prob_24h,prob_48h,prob_72h`
- [ ] Exactly **95 data rows** (same event_ids as in `sample_submission.csv`)
- [ ] Probabilities are between 0 and 1 and non-decreasing: prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h

Your current `submission.csv` already satisfies these.
