# Data dictionary

All identifiers are pseudonymous strings inherited from the public StudyChat
release. Assignments are represented as `a1` through `a7`.

## `data/da_artifacts/per_user_topic_sequences.json`

One object per student-assignment analytical unit:

- `userId`: pseudonymous student identifier;
- `topic`: assignment identifier;
- `sequences`: list of dialogue traces; each trace is an ordered list of
  dialogue-act names.

## `data/da_artifacts/axes_global.json`

- `states_global`: ordered list of the 29 observed dialogue acts;
- `superstates_global`: ordered list of the 276 observed consecutive-act pairs.

These orders define the rows and columns of every Markov feature vector.

## `data/da_artifacts/user_topic_cluster_labels_umap_order*.csv`

- `userId`, `topic`: analytical-unit key;
- `cluster`: partition assigned after UMAP and K-means.

The first-order file contains 10 clusters; the second-order file contains the
six profiles used in the main analysis.

## `data/da_artifacts/cluster_*_mean_matrix_*.txt`

Tab-separated, human-readable cluster-average transition matrices. Second-order
row labels use `previous-act|current-act`; columns are possible next acts. Files
ending in `_filtered` retain nonzero or analytically relevant rows.

## `data/process_mining_outputs/user_topic_metrics.csv`

One row per analytical unit, containing cluster/profile, event and trace counts,
mean trace length, unique-act count, mean branching factor, and mean row entropy.

## `data/process_mining_outputs/cluster_metrics.csv`

Aggregate event, trace, act, branching, entropy, and trace-length statistics for
the six second-order profiles.

## `data/process_mining_outputs/cluster_top_transitions.csv`

Dominant directed act transitions per profile, with absolute count and
conditional proportion from the source act.

## `data/process_mining_outputs/user_topic_metrics_boxplot_summary.csv`

Quartiles and medians used to summarize the student-assignment boxplots.

## `data/trajectory/user_trajectory_matrix.csv`

One row per student and one column per assignment (`a1`--`a7`). Cells contain
second-order cluster IDs; missing cells indicate an unobserved assignment.

## `data/trajectory/user_trajectory_groups.csv`

Masked-Hamming agglomerative-clustering assignment for each student trajectory.

## `data/trajectory/transition_counts_by_step.csv`

Counts of profile transitions between consecutive assignments. The columns are
`from_cluster`, `to_cluster`, `count`, `from_topic`, and `to_topic`.

