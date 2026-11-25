# Sources: how this folder works

Shared overview of the literature for the thesis, used by every Claude session (claude.ai and Claude Code) and by the author. Last restructured 2026-11-25.

**This folder in the git repo is the master copy.** The claude.ai project "thesis research" holds a mirror of these three files for chat sessions; if they disagree, the repo wins. Edit here, commit, and push.

## Files

| File | What it is | When to edit |
|---|---|---|
| `sources/README.md` | This page: layout, rating rules, editing rules. | Rarely. |
| `sources/ratings.md` | **Master list.** One row per source: bib key, venue, citations, rating, how much was read, where it is used. The single place to check "can I cite this?". | Whenever a source is added, checked or re-rated. |
| `sources/annotated-bibliography.md` | Detailed notes per paper: architecture, where memory lives, caveats, verified quotes. Explains *what* a paper says; the rating lives in `ratings.md`. | When a paper is read in more depth. |

The Claude Sheet "Related work sources: learned underwater navigation" (https://claude.ai/artifact/CdeHNgasdDdj64hdKG8wnc) is an older snapshot (2026-11-24, Sec. 2.4 only) and is **not** kept in sync.

## Rating rules

| Rating | Meaning | Use it for |
|---|---|---|
| **Strong** | Respected venue (e.g. IJRR, RA-L, T-RO, Science Robotics, ICRA, IROS, RSS, CoRL, ISRR, ICLR, NeurIPS, Nature, IEEE JOE, Ocean Engineering, Autonomous Robots, ACM Computing Surveys, Eng. Appl. of AI) **or** ≥100 citations. | Survey/trend claims, framing in the Introduction. |
| **Borderline** | Peer-reviewed, mid-tier venue (e.g. JIRS, Frontiers, OCEANS, workshops, smaller LNCS conferences) or 30–100 citations. | Primary examples; trend claims only together with a Strong source. |
| **Weak** | Low-tier venue and few citations (e.g. many MDPI journals with <30 cit.). | Primary examples only, never alone for a claim about the field. |
| **Preprint** | Not peer-reviewed (arXiv, `@misc` in the bib). | Primary examples, clearly as recent work. Never for trend claims. |
| **Do not cite** | Retracted, or wrong for the claim. | — |

"Basis" column: `S2 2026-11-24` = citation count checked on Semantic Scholar that day; `venue only` = rated from venue, citations not checked.

"Read" column: `full` = full text checked; `abstract` = abstract/metadata only; `notes` = read via a summarising tool, quotes not yet checked against the PDF.

"Bib" column: `Y` = key exists in `references.bib` (checked 2026-11-25); `–` = not yet added.

## Editing rules

1. **Key = bib key.** The first column must match `references.bib`. Add the source to `ratings.md` before citing it in the thesis.
2. **One row per source.** Update the existing row; do not add duplicates in other files.
3. **Date changes.** When a rating or verification status changes, write the date in the Note column (e.g. "re-rated Borderline 2026-12-01: venue checked").
4. **Quotes need page numbers** before they go in the thesis. Quotes in these docs were mostly extracted by a summarising tool.
5. **Do not upgrade a rating from memory.** Check venue and citations first.
6. Keep the claude.ai project mirror in sync when convenient (a chat session can copy these files into the project).

## Thesis context (for a new session)

The thesis studies a mapless local navigation policy for a UUV (BlueROV2 in MarineGym): raw sonar + depth image → frozen encoder latent (DeFM or VAE) → recurrent (GRU) + MLP policy trained with RL, outputting velocity commands to reach a goal. It is the underwater counterpart of Kulkarni & Alexis' Deep Collision Encoder (DCE). Research questions: RQ1 bring compressed-depth policies to UUVs with sonar; RQ2 characterise their limits underwater; RQ3 effect of memory architecture (none / recurrent / spatially-enhanced). LaTeX conventions: `NOTATION_STYLE.md`.
