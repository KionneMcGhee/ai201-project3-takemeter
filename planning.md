# TakeMeter planning

## Community

Movies & TV Stack Exchange is a public question-and-answer community where fans discuss plot details, narrative interpretations, production choices, and the identification of films and series. Some contributors cite scripts, production interviews, or archives. Others build a case using observations and inference, and others offer an answer with little visible support. Distinguishing those approaches can help a reader decide whether to look for a source, examine a line of reasoning, or ask for more support.

The scoping pass inspected 36 recent answers from the 2024 API window. It included film identifications with matching scene details, an Oppenheimer answer invoking a named biography, and a Gromit interpretation inferred from character behavior. The unit of analysis is an answer's cleaned text without its parent question.

## Taxonomy

Apply the labels in order. The examples below are illustrative and excluded from the dataset.

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| `sourced` | The answer uses a traceable document, interview, script, transcript, credited statement, or directly located clip or quotation as evidence for its central conclusion. | “The shooting script describes the figure as a projection, which explains why the guards never see her.” | “In the director's BAFTA interview, she says the two endings were shot for different markets.” |
| `reasoned` | The answer makes its case through identifiable observations, matching details, or a causal explanation without a traceable supporting source. | “It is probably the same hotel. The green staircase appears in both scenes, and the door numbers line up.” | “The captain lies to slow the search. Earlier she saw the only working radio being taken off the ship.” |
| `asserted` | The answer supplies a fact, guess, or opinion without giving a source or a distinct reason or observation that supports it. | “That is the 1997 remake, I'm pretty sure.” | “The red jacket was just a continuity error.” |

A title or year used to identify a work is the proposed answer. A second detail earns `reasoned` when it functions as evidence for the match. Direct dialogue earns `sourced` when placed in a traceable script, episode, scene, or timecode and used as evidence. Plot paraphrase and an author's analysis of visual details earn `reasoned`. A scene timestamp used as a locator for the specific evidence can earn `sourced`. An unrelated link, a title used merely to name the answer, and an unnamed “I read somewhere” carry little evidentiary weight.

## Hardest anticipated boundary

A movie identification can contain a title hyperlink and a synopsis. Ask whether the answer points to a traceable source to establish a relevant detail, compares a concrete scene or character detail to the requested work, or simply offers a title with incidental information. Those three situations map to `sourced`, `reasoned`, and `asserted`, respectively. A vague link marker such as “here” with no meaningful locator in the cleaned text is insufficient by itself. When the answer supplies genuine observable matching details, a hedged “possibly” does not lower it to `asserted`.

## Collection and split

Collect 200 or more public answers from the Stack Exchange API for the Movies & TV site, using a fixed 2024 creation-date window. Retain answer ID, question ID, public attribution, applicable license, and a link to the original. Normalize HTML to visible prose, remove standalone URLs and embedded media markup, and exclude empty and duplicate answers. Treat cleaned text as the annotation evidence. Scope out answers whose main content is a missing image, a correction with no answer, or disconnected chatter. Track screened skips where possible. Aim for at least 20% per label, checking after the first 200 rows. If an infrequent category needs more coverage, continue reviewing the same source and document any deliberate enrichment. Keep posts from a shared question in the same split, or retain just one answer per question before splitting. Use a fixed, stratified approximate 70/15/15 split. Pick the checkpoint with validation macro F1 and use the locked test set for the comparison.

## Evaluation and acceptance

Report accuracy, each label's precision, recall and F1, and macro F1. Accuracy is the overall hit rate. Macro F1 gives short unsupported posts the same weight as the more common reasoned answers, if the sample turns out to be imbalanced. The confusion matrix will show whether the model tends to mistake weakly supported answers for reasoned ones. Run the same test text through a zero-shot instruction model using label definitions and no labeled demonstrations. A prototype is good enough for a low-stakes reading aid if test accuracy is at least **75%**, macro F1 at least **0.70**, and recall for every label at least **60%**. Use confidence buckets and inspect errors before any broader use.

## AI tool plan

- Stress-test the boundary with deliberately ambiguous mini-posts before assigning dataset labels. Try a title with a generic link, an unnamed interview, an episode number used as a citation, a quote used decoratively, a detailed match framed as a guess, and an unsupported production assertion. Check whether each contains evidence for the main claim. Keep these synthetic prompts outside the dataset.
- Use ChatGPT to apply the taxonomy to real answers. Record its role in annotation and flag uncertain cases with decision notes. Treat the labels as AI-assisted judgments with one reviewer in this workflow. Record no human agreement statistic without genuinely independent human labels.
- Use AI to suggest error patterns after the test run. Verify proposed patterns against individual prediction rows and quantify a recurring confusion or a text-feature slice with denominators.
- Stretch work selected up front includes confidence buckets, quantitative error slices, and a local interface that returns the label and confidence for new text. A second annotator is outside the current collection plan.

## Collection adjustment

The initial 2024 sample yielded few standalone unsupported claims. The supplemental pass screened short public answers from the same site in the remainder of the 2024 pool and six 2023 API pages. Include selected short sourced and reasoned answers as well so that being short is insufficient to determine the label. The resulting benchmark is an enriched sample from this community, so class proportions give no estimate of the site's actual prevalence.
