# StudyChat downstream interaction-profile analysis

Reproducibility artifacts for the dialogue-act classifier and downstream
analysis used in the article *Padrões de Interação em Tutoria por IA: Perfis
Conversacionais Compatíveis com Aprendizagem Autorregulada*.

The repository starts from fixed, pseudonymized dialogue-act sequences and
reproduces the analytical stages reported in the article:

1. first- and second-order Markov transition matrices per student-assignment;
2. z-score standardization, two-dimensional UMAP, and K-means clustering;
3. structural metrics and PM4Py Heuristic Miner models;
4. masked-Hamming longitudinal trajectory clustering;
5. the four result figures used in the manuscript.

## Scope

This release contains the complete downstream package and a portable,
checkpointed implementation of the supervised SwDA classifier. It does not
distribute raw StudyChat messages, tutor responses, Switchboard transcripts,
classified prompts or model checkpoints. Those inputs are acquired locally as
documented in [`classifier/README.md`](classifier/README.md).

The fixed downstream analytical input contains:

| Item | Count |
|---|---:|
| Student-assignment units | 916 |
| Students | 201 |
| Dialogue traces | 1,727 |
| Dialogue-act events | 16,364 |
| Observed dialogue-act states | 29 |
| Observed second-order superstates | 276 |

## Repository structure

```text
data/
  da_artifacts/             fixed sequences, axes, Markov outputs and partitions
  process_mining_outputs/   structural metrics and dominant transitions
  trajectory/               longitudinal matrices, groups and flows
figures/reference/          figures included with the article
scripts/
  reproduce_analysis.py     rebuilds and validates clusters, metrics and trajectories
  regenerate_figures.py     rebuilds the article figures
results/reference/          machine-readable reference summary
classifier/                 supervised SwDA training and StudyChat inference
```

Column-level definitions are available in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
`MANIFEST.sha256` records the checksum of every derived input artifact.

## Run the supervised classifier

The classifier has its own Python 3.11 environment because its pinned deep
learning dependencies differ from the downstream numerical environment. On an
Apple Silicon Mac, start with:

```bash
cd classifier
./setup_macos.sh
```

Then follow [`classifier/README.md`](classifier/README.md) to accept StudyChat
access, prepare the local SwDA inputs and launch the resumable full run.

## Reproduce

Python 3.11 and Graphviz are required. On macOS, Graphviz can be installed with
`brew install graphviz`; on Debian/Ubuntu, use `apt-get install graphviz`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/reproduce_analysis.py --require-exact-umap
.venv/bin/python scripts/regenerate_figures.py --with-umap

# Optional integrity check (macOS)
shasum -a 256 -c MANIFEST.sha256
```

Generated tables are written to `results/reproduced/`; regenerated PDF and PNG
figures are written to `figures/reproduced/`.

## Expected validation

The scripts fail explicitly if a structural or numerical invariant changes. In
the validated macOS arm64 environment, exact UMAP validation recovers:

| Analysis | k | Silhouette | ARI/NMI against stored partition |
|---|---:|---:|---:|
| First-order Markov | 10 | 0.429854 | 1.0 / 1.0 |
| Second-order Markov | 6 | 0.463708 | 1.0 / 1.0 |
| Longitudinal trajectories | 8 | 0.212073 | 1.0 / 1.0 |

It also validates the 916 student-assignment structural-metric rows, the six
cluster summaries, 42 longitudinal nodes, 168 links, and 614 transitions
between consecutive assignments.

UMAP uses approximate nearest-neighbor kernels whose floating-point execution
can vary by CPU architecture even with fixed versions and seed. On a different
platform, omit `--require-exact-umap` and `--with-umap`: the script still
rebuilds both embeddings, reports ARI/NMI against the stored partitions, and
strictly validates all deterministic Markov, process-metric, and trajectory
invariants. The reference UMAP partition and figure remain preserved in the
repository.

## Reproducibility choices

- random seed: `42`;
- UMAP: 2 components, cosine metric, 15 neighbors, minimum distance 0.1;
- K-means: 10 initializations, with `k=10` and `k=6`;
- trajectory distance: masked Hamming;
- trajectory clustering: average-linkage agglomerative clustering over a
  precomputed distance matrix;
- process mining: PM4Py 2.7.18 Heuristic Miner with that version's parameters
  made explicit in the figure script.

Exact downstream dependency versions are pinned in `requirements.txt`, while
classifier versions are pinned separately in `classifier/requirements.txt`.
GitHub Actions runs the portable downstream validation, regenerates Figures
2--4 and checks the classifier source contract on every push; the exact UMAP
check is available explicitly for the validated environment.

## Data and citation

The distributed data are derived analytical artifacts with pseudonymous IDs;
they contain no dialogue text. See [DATA_NOTICE.md](DATA_NOTICE.md) for provenance
and reuse conditions and [CITATION.cff](CITATION.cff) for citation metadata.

The analysis code is released under the MIT license.
