# Terminology Review

Use this as an approval list before the next paper-wide terminology pass. The
goal is clear and impactful prose that remains anchored in the statistical
literature.

## Terms to approve

- `harmful shift`: Keep as the manuscript's main term. It matches nearby work on
  harmful distribution shifts while staying plainer than `adverse shift`. Use
  `adverse shift` only when explicitly naming D-SOS lineage.
- `severity score`: Approved plain term for the D-SOS-style score object. First
  use should include the bridge sentence: D-SOS calls these `outlier scores`; in
  monitoring language, we call the user-chosen quantity a severity score when
  larger values mean worse behavior for the application.
  Mention that these can be thought of as harm scores or degradation scores when
  those labels fit the domain.
- `common support`: Keep as the formal statistical term. Precede it with the
  plain gloss `the region both samples actually cover` or `comparable cases` for
  non-expert readers.
- `weak overlap` versus `low overlap`: `Weak overlap` is more standard in causal
  inference and positivity discussions; `low overlap` is plainer and already used
  in figures/results. Decide whether to standardize on `weak overlap` in prose
  and reserve `low overlap` for figure labels.
- `non-sequential`: Keep for the contrast with sequential tests. Avoid
  `fixed-time` in main prose unless explicitly discussing why repeated
  fixed-time testing is not sequentially valid.
- `density-ratio weights`: Keep for the construction. Use `importance weights`
  only when connecting to covariate-shift literature, and avoid `overlap weights`
  for the method because that term has a specific causal-inference meaning.
- `source` and `target`: Keep in the paper for mathematical consistency. In the
  introduction, continue glossing them as any baseline and deployment pair.

## Deep naming review: `severity score` and `outlier score`

Decision as of 2026-05-31: use `severity score` as the manuscript-facing term.
Keep `outlier score` only when naming D-SOS terminology, and translate it with
the bridge sentence. Use `harm score` and `degradation score` as explanatory
glosses rather than primary manuscript terms.

The score object has to do three things at once.

1. Stay plain: a reader should understand the idea before the formal definition.
2. Stay honest: D-SOS only requires a rank-consistent scalar, not a calibrated
   risk, probability, or loss.
3. Stay anchored: the paper should acknowledge that D-SOS and anomaly-detection
   literature call related quantities `outlier scores`, `outlyingness`, `anomaly
   scores`, or `scores of abnormality`.

Research basis:

- D-SOS uses user-defined `outlier scores` and compares contamination rates
  across score thresholds. In our setting, the same statistical role is played
  by a user-chosen score whose ordering matches what the application treats as
  worse.
- Outlier/anomaly-detection literature strongly supports `outlier score`,
  `outlier ranking`, `outlyingness`, `anomaly score`, and `abnormality score` as
  technical names. These terms are about rarity or departure from normality, not
  necessarily harm.
- ML evaluation literature uses `risk`, `loss`, `score`, `utility`, and
  `performance metric` for model quality. These terms are familiar, but they can
  imply stronger structure than D-SOS needs: expected loss, calibration,
  optimization direction, or a named prediction metric.
- Harmful-shift literature often speaks about `risk increase`, `loss`, or
  `performance degradation`, especially in sequential tests. Those are useful
  anchors, but this paper's non-sequential test only needs the score ranking to
  agree with the application-specific notion of worse.

### Alternatives for the manuscript-facing term

| Term | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| `harm score` | Short, plain, directional, broad across risk/error/uncertainty/confidence. Does not imply calibration. Fits `harmful shift`. | `Harm` can sound moralized in low-stakes ML examples. Needs first-use definition because it is not a standard term. | Approved explanatory gloss, not the main manuscript term. |
| `degradation score` | Strong for model monitoring: readers immediately hear performance getting worse. Literature-adjacent through `performance degradation`. | Narrower than harm: less natural for predicted risk, fairness harms, uncertainty, or domain-specific bad outcomes. Four syllables slows prose. | Approved explanatory gloss, especially for model-performance contexts. |
| `severity score` | Plain and polished. Captures a graded notion of worse without saying rare or anomalous. Common in applied domains. | May imply calibrated magnitude, clinical severity, or absolute seriousness rather than rank only. Less tied to `harmful shift`. | Chosen manuscript-facing term. Define once as rank-consistent and not necessarily calibrated. |
| `risk score` | Very familiar in statistics, health, credit, and monitoring. Anchors naturally to risk-increase literature. | Often means calibrated probability or expected loss. Could mislead when examples are prediction error, uncertainty, or confidence drop. | Use only when the score really is risk; do not use as the umbrella term. |
| `loss score` | Mathematically anchored. Larger-is-worse is clear. Connects to expected risk and ML metrics. | Sounds redundant; `loss` already is a score. Too narrow for uncertainty, confidence, or application-specific severity. | Useful in method examples, weak as the paper-wide term. |
| `error score` | Plain for prediction-error examples. Direction is obvious. | Too narrow: D-SOS can use risk, uncertainty, confidence, or other ordered signals. Excludes pre-outcome monitoring. | Avoid as umbrella; use only for prediction-error instantiations. |
| `failure score` | Very plain and high-impact. Makes alarm semantics concrete. | Too binary and too strong. Many examples are graded degradation, not failure. Could sound theatrical. | Too narrow for the main term. |
| `bad-outcome score` | Plain, concrete, and clearly directional. Works for risk, loss, and adverse outcomes. | Clunky. `Bad` is informal for ICML prose; hyphenated phrase is awkward in equations. | Good explanatory gloss, not a term of art. |
| `worse score` | The plainest possible directional name. Makes rank-consistency obvious. | Ungrammatical or childish to many readers; hard to use in formal prose. | Too awkward despite clarity. |
| `worsening score` | Plain and directional; suggests movement toward worse outcomes. | Implies a temporal change within an observation, while the score is per observation. Awkward in formulas. | Avoid. |
| `concern score` | Friendly, intuitive, and less moralized than `harm`. Says what should draw attention. | Informal; not anchored in the statistical literature. May sound subjective or product-oriented. | Good if accessibility beats academic sharpness; otherwise not first choice. |
| `warning score` | Plain monitoring language. Connects naturally to alarms. | A warning is the output of a detector, not the score being compared. Could confuse statistic with alert. | Avoid as score name. |
| `monitoring score` | Broad and plain. Does not overclaim calibration or harm. | Too generic; does not say larger means worse. Weakens the one-sided harmful-shift story. | Useful as a generic umbrella, but not the main term. |
| `diagnostic score` | Polished, broad, and already appears naturally in the related-work contrast. Avoids rarity language. | Direction is not inherent; a diagnostic score could measure anything. Less direct than `severity score`. | Neutral fallback when direction is not central. |
| `adverse score` | Anchors to `adverse shift` and D-SOS lineage. Directional. | More academic and less plain than `severity score`; `adverse score` is not idiomatic. | Avoid except maybe in a one-time lineage phrase. |
| `badness score` | Extremely plain; larger-is-worse is unmistakable. | Too colloquial for the manuscript. Risks sounding unserious. | Do not use in the paper. |

### Alternatives for replacing or translating `outlier score`

| Term | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| `outlier score` | Exact D-SOS and outlier-detection term. Literature-native. Signals ranking across thresholds. | Suggests rarity, tailness, or departure from normality, not harm. Can make our problem sound like anomaly detection. | Keep only when naming D-SOS terminology, then translate. |
| `outlyingness score` | Precise academic variant; makes the outlier literature link explicit. | Even more jargon-heavy than `outlier score`; still about rarity rather than worse outcomes. | Avoid in main prose. |
| `anomaly score` | Widely recognized in ML. Plainer than `outlier score` for some readers. | Anomalies can be benign; still not directional toward harm. Pulls the paper toward anomaly detection. | Mention only as a literature cousin, not as our term. |
| `abnormality score` | Matches scikit-learn-style language such as scores or degrees of abnormality. | Clunky and norm-loaded. Abnormality is not necessarily harmful. | Avoid unless discussing anomaly-detection terminology. |
| `deviation score` | Plain, neutral, and less loaded than outlier/anomaly. | Directionless: deviation can be good, bad, or benign. Does not convey one-sided worsening. | Good for neutral drift, weak for harmful shift. |
| `nonconformity score` | Literature-native in conformal prediction and martingale tests. Means departure from a reference pattern. | Too technical and tied to a different testing lineage. Directionless without extra explanation. | Do not use for this paper's main object. |
| `rank score` | Honest about the statistical requirement: ordering matters more than calibration. | Sounds procedural and opaque. Does not explain what is being ranked or why it matters. | Use in definitions if needed, not as the public term. |
| `diagnostic score` | Broad replacement for `outlier score`; avoids implying rarity. Works for risk, error, uncertainty, and confidence. | Needs an adjective or definition to say which direction is worse. Less direct than `severity score`. | Neutral translation when the prose should not emphasize severity. |
| `monitoring score` | Domain-appropriate and plain. Keeps focus on deployment monitoring rather than anomaly detection. | Too generic; could be any monitored quantity. Needs a one-sided definition. | Acceptable fallback, but weaker than `diagnostic score`. |
| `severity score` | Better than `outlier score` when the score measures how bad a case is. Plain and directional. | Can imply calibrated magnitude and may not fit confidence/drop/error examples equally well. | Chosen translation for this paper, with rank-consistency defined explicitly. |
| `concern score` | Very readable and directly answers why the score matters. | Informal and subjective. No strong literature anchor. | Useful as explanatory prose, not the formal term. |

Decision for the current paper:

1. Use `severity score` as the manuscript-facing term.
2. Preserve `outlier score` only when identifying D-SOS's terminology.
3. Use a bridge sentence on first contact: `D-SOS calls these outlier scores;
  in monitoring language, we call the user-chosen quantity a severity score when
   larger values mean worse behavior for the application.`
4. Mention that severity scores can be thought of as harm scores or degradation
  scores when those labels better match the domain.
5. Use `diagnostic score` only as a neutral fallback in places where direction is
  not central, especially in related-work contrasts.

Shortlist for user choice:

- Chosen: `severity score`.
- Approved explanatory glosses: `harm score` and `degradation score`.
- Strongest model-performance gloss: `degradation score`.
- Strongest neutral replacement for `outlier score`: `diagnostic score`.
- Do not use as umbrella terms: `risk score`, `loss score`, `anomaly score`, or
  `outlier score`, because each imports a narrower literature meaning.

## Terms to strengthen or avoid

- Style preference approved on 2026-05-31 for the abstract and early
  introduction: start with the concrete failure mode, not a principle slogan.
  Prefer `testing for harmful shift` to nominal phrases such as
  `harmful-shift test` or `harmful-shift alarm`.
  Prefer `overlap on comparable cases` over abstract phrasing such as
  `meaningfully comparable` when the sentence is naming the core failure mode.
  Prefer causal failure-mode verbs such as `create that illusion` over
  procedural phrasing such as `raise that result`.
  Delay `severity` and `severity score` until the manuscript has introduced the
  score object; use plain worsening language earlier.
  In empirical payoff sentences, prefer plain operating-characteristic language
  such as `without losing power` over softer abstractions such as
  `preserving sensitivity`.
  In the abstract's mechanism sentence, prefer the leaner `domain classifier`
  over a longer gloss such as `a classifier that predicts source versus target
  membership` when the extra detail is not carrying its weight.
  Carry the abstract's failure-mode voice into the introduction instead of
  reverting to slogan-first openings like `A monitoring alarm should point to
  the right failure.`

- `harmfulness score`: Precise but clunky. Prefer `severity score` after defining
  the D-SOS connection.
- `meaningfully comparable`: Useful in the abstract/introduction, but weak if
  repeated. Prefer `comparable cases`, `common support`, or `regions both samples
  cover` depending on audience.
- `substantively worse`: Literature-adjacent through D-SOS, but vague on first
  read. Pair it with the user-defined severity score or non-negligible threshold
  when precision matters.
- `sharper question`: Good once in the introduction; overuse makes the paper feel
  rhetorical. Prefer naming the actual question: `did the target get worse on
  comparable cases?`
- `contamination`: Strong but can sound moralized. Prefer `weak-overlap regions`,
  `mismatch`, or `incomparable regions` unless discussing a synthetic mechanism.
- `diagnostics`: Acceptable for source-weighted and target-weighted variants, but
  `diagnostic variants` is clearer than `supporting diagnostics`.
- Overheated verbs such as `scream`, `hijacked`, `abandon`, `aggressively`, and
  `sledgehammer`: avoid in the manuscript. The results are strong enough without
  theatrical language.

## Writing Rules (Clarity Lock)

Use this checklist during manuscript edits to keep prose concise, direct,
precise, and readable for both experts and non-experts.

1. One sentence, one main claim. Move mechanisms, caveats, or examples to the
  next sentence.
2. Prefer concrete actions over abstract nouns.
  Write `lambda controls the trade-off by changing weights and effective sample
  size`, not `lambda traces a frontier`.
3. Make cause-and-effect explicit in failure-mode claims.
  Prefer `X causes false alarms when Y` over compressed causal chains.
4. Name the comparison population explicitly when it matters.
  State whether the claim is about full observed samples or common support.
5. Keep technical precision, but stage it.
  Introduce plain language first, then notation or literature qualification.
6. Keep terminology stable in local context.
  Use one primary term per concept in a section, and avoid rapid switching
  between near-synonyms.
7. End paragraphs with concrete takeaways.
  Prefer specific operational implications over abstract summary phrasing.

Quick review pass before commit:

- Any sentence above roughly 30 words should be a split candidate.
- Any sentence containing two or more semicolons should be a split candidate.
- Replace abstract constructions (`trade-off frontier`, `inferential goal`) with
  action-oriented phrasing when possible.
- Verify that each paragraph includes at least one plain-language sentence.