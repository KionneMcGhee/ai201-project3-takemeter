# TakeMeter

**AI201 · Project 3 · Kionne McGhee II**

TakeMeter classifies how answers on Movies & TV Stack Exchange support their central claims. It distinguishes traceable citation, inference from observations, and unsupported assertion. Fans in this community write about everything from the meaning of a scene to the origin of a production detail. These labels let the reader distinguish what can be checked against a source from what needs to be evaluated as an interpretation.

[Watch the narrated demo](demo.mp4) · [Open the offline classifier](TakeMeter.html) · [Read the design plan](planning.md) · [Download the labeled dataset](labeled_posts.csv)

## Taxonomy

The first rule that applies determines the label. These six invented examples illustrate the definitions. The dataset contains real public answers.

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| `sourced` | The answer uses a traceable document, interview, script, transcript, credited statement, or directly located clip or quotation as evidence for its central conclusion. | “The shooting script describes the figure as a projection, which explains why the guards never see her.” | “In the director's BAFTA interview, she says the two endings were shot for different markets.” |
| `reasoned` | The answer makes its case through identifiable observations, matching details, or a causal explanation without a traceable supporting source. | “It is probably the same hotel. The green staircase appears in both scenes, and the door numbers line up.” | “The captain lies to slow the search. Earlier she saw the only working radio being taken off the ship.” |
| `asserted` | The answer supplies a fact, guess, or opinion without giving a source or a distinct reason or observation that supports it. | “That is the 1997 remake, I'm pretty sure.” | “The red jacket was just a continuity error.” |

Use `sourced` when the evidence is traceable and establishes something about the main conclusion. A movie title by itself is the answer to an identification question. Independent scene details that explain the match earn `reasoned`. A directly relevant quotation with a traceable locator can earn `sourced`. Citations for secondary or decorative points do not change the label.

## Annotated dataset

The [Stack Exchange API](https://api.stackexchange.com/docs/answers) supplied public answer bodies from the [Movies & TV community](https://movies.stackexchange.com/). The initial pool was the seven newest 100-answer API pages within 2024. The supplemental short-answer screen covered the remaining 2024 pages and the six newest 2023 pages. The final file contains cleaned text, the reference label, notes where applicable, stable answer and question IDs, the source link, and each public author name and profile. The source material is attributed under [CC BY-SA 4.0](dataset_attribution.md).

After an initial review of 36 answers, ChatGPT applied the agreed decision rules to the individual cleaned candidates. The review queue combined short answers, answers with possible source markers, and randomly ordered remaining candidates. An additional screen enriched the short-answer group. HTML was converted to prose, raw URLs and embeds removed, some quoted material shortened, and long text excerpted at roughly 1,000 characters. Empty, incomplete or unusable excerpts and duplicate questions were screened out. Annotation was based on the resulting visible text. The distribution reflects the sampling design.

| Label | Count | Share | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| `asserted` | 26 | 11.1% | 18 | 4 | 4 |
| `reasoned` | 122 | 51.9% | 86 | 18 | 18 |
| `sourced` | 87 | 37.0% | 61 | 13 | 13 |
| **Total** | **235** | **100%** | **165** | **35** | **35** |

The fixed seed is 201. A question ID appears only once, so answers from the same thread cannot straddle splits. The seed and IDs for all three splits are in [evaluation_results.json](evaluation_results.json).

### Difficult annotation decisions

| Answer | Decision |
| --- | --- |
| [123458 · Identifying The Devil’s Playground](https://movies.stackexchange.com/a/123458) | **Reasoned.** The Catholic school setting and character relationship help identify the movie. Rotten Tomatoes and festival awards support its reception, which is a different claim. |
| [124087 · A photograph and the viewer](https://movies.stackexchange.com/a/124087) | **Reasoned.** Ansel Adams’s quote is an analogy for how the photograph is seen. The explanation of the character’s memory rests on the film’s photograph and painting. |
| [123144 · Dating a scene by Yuma prison](https://movies.stackexchange.com/a/123144) | **Reasoned.** The sheriff’s remark and the prison opening year form a causal time inference. The answer gives no traceable reference for that opening year. |
| [124149 · Identifying Risky Business](https://movies.stackexchange.com/a/124149) | **Sourced.** The answer locates the clock scene at about 46 minutes. That is a directly checkable locator for the specific evidence supporting the identification. |

## Fine-tuning

Training ran locally on an Apple Silicon Mac using Apple’s [pretrained English word embedding](https://developer.apple.com/documentation/naturallanguage/nlembedding) and NumPy. The classifier mean-pools 300-dimensional pretrained word vectors and combines them with signed hashed unigrams and bigrams and surface features such as length, quotation marks, timecodes, and citation words. Supervised training updates the pretrained embedding values and the new three-class softmax head. The selected checkpoint updated all 3,264 vocabulary rows compared with their initial vectors.

The split is approximately 70/15/15. Train-only vocabulary and surface scaling statistics feed the supervised model. Mini-batches contain 16 answers, and inverse-square-root class weighting gives rare asserted examples more influence without fully equalizing class weight. Training compared head learning rates of 0.005 and 0.015 for up to 36 epochs. The embedding layer used 15% of the head rate to limit changes to pretrained values in a small dataset. The 0.005 run at epoch 15 had the best validation macro F1 (0.810), with validation negative log likelihood as the tiebreaker. The kept embedding rate was 0.00075. A single temperature of 0.60 was fit on validation scores after model selection. The held-out test set was used for final scoring.

## Zero-shot baseline

The executed baseline used the untouched Apple embedding and fixed verbal descriptions of the three labels. The [exact nine phrases](evaluation_results.json) are under `zero_shot.prototypes`, including “a short personal guess with no evidence,” “uses observable details to explain why this happened,” and “traceable evidence cited from a book, report or clip.” For each test answer, the baseline averaged its pretrained word vectors and computed cosine similarity to the three prompt vectors for each label. It picked the label with the highest mean similarity. It consumed no labeled examples, learned task weights, or reference labels. The artifact records cosine scores and a prediction for each test ID.

The course specified a Groq Llama 4 Scout instruction baseline. [Groq retired that model from free and developer tiers on July 17, 2026](https://console.groq.com/docs/deprecations). The completed comparison uses zero-shot semantic matching. It measures a weaker form of prompting than an instruction LLM. The [full instruction prompt](baseline_prompt.txt) and [optional Groq runner](groq_baseline.py) support an additional, fixed-test comparison with Groq’s recommended replacement, `openai/gpt-oss-120b`. The reported results come from the local comparison. The optional API run requires a Groq key.

## Evaluation

Both completed systems scored the same 35 test answers. The constant majority-label reference would score **51.4%** accuracy.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Fine-tuned TakeMeter | **65.7% (23/35)** | **0.581** |
| Zero-shot embedding prompt baseline | 11.4% (4/35) | 0.068 |

The zero-shot model predicted `asserted` for all 35 answers. Averaging word vectors largely captures topic and vocabulary overlap. It missed the difference between mentioning a source and using one to establish a claim.

### Per-class metrics

| Label | Test count | TakeMeter precision | TakeMeter recall | TakeMeter F1 | Baseline precision | Baseline recall | Baseline F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asserted` | 4 | 1.000 | 0.250 | 0.400 | 0.114 | 1.000 | 0.205 |
| `reasoned` | 18 | 0.722 | 0.722 | 0.722 | 0.000 | 0.000 | 0.000 |
| `sourced` | 13 | 0.563 | 0.692 | 0.621 | 0.000 | 0.000 | 0.000 |

### Fine-tuned confusion matrix

Rows are reference labels. Columns are predicted labels. [PNG version](confusion_matrix.png)

| Reference ↓ / Predicted → | `asserted` | `reasoned` | `sourced` |
| --- | ---: | ---: | ---: |
| `asserted` | **1** | 1 | 2 |
| `reasoned` | 0 | **13** | 5 |
| `sourced` | 0 | 4 | **9** |

### Three prediction errors

1. [124137 · The Killing Grounds identification](https://movies.stackexchange.com/a/124137) was **asserted**, predicted **sourced** at **68.8%**. “As mentioned in the comments” signals a possible reference, but the cleaned answer offers no identifiable comment or matching detail. The model appears to have treated a vague attribution as citation evidence.
2. [123999 · Estimating the age of the Star Wars galaxy](https://movies.stackexchange.com/a/123999) was **sourced**, predicted **reasoned** at **98.5%**. The post uses a named wiki for its central age premise and then reasons from that premise. The numerical comparison and conclusion take up much of the post, which may obscure the role of the named source. The misspelling “Wookiepedia” may also reduce overlap with familiar citation language.
3. [123100 · Creature behavior in A Quiet Place](https://movies.stackexchange.com/a/123100) was **reasoned**, predicted **sourced** at **69.0%**. The argument relies on paraphrased actions in a flooded tunnel. Film titles, a release year, and language such as “clearly demonstrated” may resemble a citation, although the text gives no exact locator or quoted evidence.

### Sample classifications

The four examples below were re-run through the saved model. References are annotation labels.

| Test answer | Reference | Prediction | Confidence |
| --- | --- | --- | ---: |
| [123483](https://movies.stackexchange.com/a/123483) · IMDb stunt credit identifies Andy Dylan. | `sourced` | `sourced` | 95.0% |
| [124151](https://movies.stackexchange.com/a/124151) · Later hooded appearances support the identification of Robert the Elder. | `reasoned` | `reasoned` | 96.5% |
| [124144](https://movies.stackexchange.com/a/124144) · A standalone identification of the short One, Two, Tree. | `asserted` | `asserted` | 96.8% |
| [124137](https://movies.stackexchange.com/a/124137) · A film title attributed vaguely to a prior comment. | `asserted` | `sourced` | 68.8% |

Example 123483 earned `sourced` because the named credits section contains the relevant role and performer. The text makes the verification path clear.

### Confidence and error patterns

| Confidence range | Predictions | Mean confidence | Observed accuracy |
| --- | ---: | ---: | ---: |
| Below 50% | 2 | 41.0% | 50.0% |
| 50% to below 65% | 6 | 56.1% | 66.7% |
| 65% to below 80% | 11 | 70.1% | 36.4% |
| 80% to below 90% | 7 | 86.6% | 85.7% |
| 90% or more | 9 | 97.8% | 88.9% |

Confidence was informative at the high end, but the middle bucket was materially overconfident. On this test set, temperature scaling increased expected calibration error from **0.103 to 0.154** and Brier score from **0.430 to 0.454**. The scaling value was chosen on validation, so its worse held-out performance reflects weak transfer from a small validation set. The UI shows model confidence as a diagnostic score.

The `sourced`–`reasoned` boundary accounted for **9 of 12** test errors (five in one direction, four in the other). The fine-tuned model picked up recognizable verification cues like script mentions and credit listings. It often confused those cues with the function a source plays in the argument. For instance, it missed a wiki citation used inside a longer inference and overvalued a generic reference to comments. Every wrong asserted answer had a different combination of cues, including a cast name with a year, a personal claim about a painting, or a vague attribution. With four asserted test rows, one prediction changes that class’s recall by 25 points.

The success target was 75% accuracy, 0.70 macro F1, and 60% recall for each class. The reported checkpoint missed all three targets. More matched examples of incidental versus supporting citations and of descriptive claims versus actual inference would test the intended boundary directly. Independent annotation would help measure where the written policy is hard to follow consistently.

### Spec reflection

The spec forced a definition of traceable support before annotation and motivated grouping by question ID before the benchmark. Both choices informed the errors discussed above. The implementation diverged from the recommended Colab/DistilBERT setup because the planned transformer download was blocked in this workspace. It used a locally available pretrained Apple embedding with supervised updates. The provider retirement also made an exact free-tier Llama 4 Scout reproduction unavailable, so the report identifies the substitute baseline explicitly.

### AI usage

- **Taxonomy stress-testing.** ChatGPT was directed to generate borderline examples covering titles with links, episode references, decorative quotes and speculative interpretations. It proposed treating explicit references as support. The rule was tightened to require that a traceable reference establish a relevant part of the main claim. Example 124087 is a real application of that revision.
- **Annotation.** ChatGPT was directed to apply the decision order to cleaned public answers. Each included label is AI-assisted. The annotation pass discarded unusable entries, reviewed ambiguous cases with the finalized rules, and used an additional short-answer screen to increase asserted coverage. This project has no independent human-annotation agreement estimate.
- **Error review.** ChatGPT was directed to look for evidence of a recurring mistake. The comparison of citation markers versus citation function was checked against concrete test rows and the confusion matrix. The report separates observed mistakes from hypotheses about the cues the classifier may have used.

## Reproduce and run

Open [TakeMeter.html](TakeMeter.html) in a modern browser. It embeds the checkpoint and runs locally without a server, network calls, installation, or API key. Select a held-out example or paste a new answer, then click **Classify**.

```sh
python -m pip install -r requirements.txt
python takemeter.py "The screenplay describes the figure as a projection."
python demo.py
```

Retraining uses macOS and Swift to access the same English word embedding. Inference using the bundled checkpoint works wherever NumPy is available. To retrain and rebuild the standalone interface on macOS, run `python train.py` followed by `python build_interface.py`. The optional hosted baseline runs with `python groq_baseline.py` after setting `GROQ_API_KEY` in the environment. The evaluation JSON records the original test scores and individual predictions. The video uses synthetic narration generated with [SAM JS](https://github.com/discordier/sam).
