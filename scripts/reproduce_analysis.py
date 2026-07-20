#!/usr/bin/env python3
"""Reproduce and validate the downstream analysis from fixed act sequences."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "results" / "reproduced"
TOPICS = [f"a{i}" for i in range(1, 8)]
PROFILE_LABELS = {
    0: "Exploratorio / investigativo A",
    1: "Localmente sistematico",
    2: "Linear / diretivo",
    3: "Estrategia mista",
    4: "Exploratorio / investigativo B",
    5: "Semi-exploratorio",
}
EXPECTED = {
    "units": 916,
    "users": 201,
    "traces": 1727,
    "events": 16364,
    "states": 29,
    "superstates": 276,
    "first_transitions": 14637,
    "second_transitions": 12910,
    "zero_second_order_units": 90,
}
EXPECTED_CLUSTER_SIZES_ORDER1 = {0: 151, 1: 60, 2: 134, 3: 81, 4: 136, 5: 149, 6: 74, 7: 19, 8: 82, 9: 30}
EXPECTED_CLUSTER_SIZES_ORDER2 = {0: 191, 1: 117, 2: 43, 3: 197, 4: 184, 5: 184}
EXPECTED_TRAJECTORY_SIZES = {0: 10, 1: 15, 2: 53, 3: 37, 4: 44, 5: 20, 6: 12, 7: 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-exact-umap",
        action="store_true",
        help="Fail unless UMAP/K-means exactly recovers the stored partitions.",
    )
    return parser.parse_args()


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(float, copy=False)
    sums = matrix.sum(axis=1, keepdims=True)
    np.divide(matrix, sums, out=matrix, where=sums != 0)
    return matrix


def transition_vectors(
    units: list[dict[str, object]],
    states: list[str],
    superstates: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], int, int]:
    state_index = {state: i for i, state in enumerate(states)}
    superstate_index = {state: i for i, state in enumerate(superstates)}
    first_vectors: list[np.ndarray] = []
    second_vectors: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    total_first = 0
    total_second = 0

    for unit in units:
        first = np.zeros((len(states), len(states)), dtype=float)
        second = np.zeros((len(superstates), len(states)), dtype=float)
        for sequence in unit["sequences"]:
            sequence = [str(value) for value in sequence]
            for left, right in zip(sequence, sequence[1:]):
                first[state_index[left], state_index[right]] += 1
                total_first += 1
            for left, middle, right in zip(sequence, sequence[1:], sequence[2:]):
                second[superstate_index[(left, middle)], state_index[right]] += 1
                total_second += 1
        first_vectors.append(normalized_rows(first).ravel())
        second_vectors.append(normalized_rows(second).ravel())
        keys.append((str(unit["userId"]), str(unit["topic"])))

    return np.vstack(first_vectors), np.vstack(second_vectors), keys, total_first, total_second


def cluster(features: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    scaled = StandardScaler().fit_transform(features)
    embedding = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=seed,
    ).fit_transform(scaled)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embedding)
    return labels, embedding, float(silhouette_score(embedding, labels))


def compare_partition(
    keys: list[tuple[str, str]],
    labels: np.ndarray,
    reference_path: Path,
) -> tuple[float, float, pd.DataFrame]:
    reproduced = pd.DataFrame(keys, columns=["userId", "topic"])
    reproduced["cluster_reproduced"] = labels.astype(int)
    reference = pd.read_csv(reference_path, dtype={"userId": str, "topic": str})
    reference["cluster"] = reference["cluster"].astype(int)
    merged = reproduced.merge(reference, on=["userId", "topic"], validate="one_to_one")
    if len(merged) != len(keys):
        raise ValueError("Reference and reproduced partitions do not contain the same units.")
    ari = float(adjusted_rand_score(merged["cluster"], merged["cluster_reproduced"]))
    nmi = float(normalized_mutual_info_score(merged["cluster"], merged["cluster_reproduced"]))
    return ari, nmi, reproduced


def local_transition_matrix(sequences: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    acts = sorted({str(act) for sequence in sequences for act in sequence})
    index = {act: i for i, act in enumerate(acts)}
    matrix = np.zeros((len(acts), len(acts)), dtype=float)
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            matrix[index[str(left)], index[str(right)]] += 1
    return normalized_rows(matrix), acts


def entropy(matrix: np.ndarray) -> float:
    values = []
    for row in matrix:
        probabilities = row[row > 0]
        if len(probabilities):
            values.append(float(-np.sum(probabilities * np.log2(probabilities))))
    return float(np.mean(values)) if values else 0.0


def user_topic_metrics(units: list[dict[str, object]], labels: pd.DataFrame) -> pd.DataFrame:
    lookup = labels.set_index(["userId", "topic"])["cluster"]
    rows = []
    for unit in units:
        key = (str(unit["userId"]), str(unit["topic"]))
        sequences = [[str(act) for act in sequence] for sequence in unit["sequences"]]
        lengths = [len(sequence) for sequence in sequences]
        matrix, acts = local_transition_matrix(sequences)
        cluster_id = int(lookup.loc[key])
        rows.append(
            {
                "userId": key[0],
                "topic": key[1],
                "cluster": cluster_id,
                "profile": PROFILE_LABELS[cluster_id],
                "turns": int(sum(lengths)),
                "trace_count": int(len(lengths)),
                "mean_trace_length": float(np.mean(lengths)),
                "unique_acts": int(len(acts)),
                "avg_branching": float(np.mean(np.sum(matrix > 0, axis=1))),
                "avg_entropy": entropy(matrix),
            }
        )
    return pd.DataFrame(rows).sort_values(["cluster", "userId", "topic"]).reset_index(drop=True)


def validate_user_metrics(reproduced: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path, dtype={"userId": str, "topic": str})
    reference = reference.sort_values(["cluster", "userId", "topic"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(reproduced, reference, check_dtype=False, rtol=1e-10, atol=1e-10)


def validate_cluster_metrics(units: list[dict[str, object]], labels: pd.DataFrame, reference_path: Path) -> None:
    lookup = labels.set_index(["userId", "topic"])["cluster"]
    rows = []
    for cluster_id in range(6):
        selected = [
            unit for unit in units
            if int(lookup.loc[(str(unit["userId"]), str(unit["topic"]))]) == cluster_id
        ]
        sequences = [sequence for unit in selected for sequence in unit["sequences"]]
        matrix, acts = local_transition_matrix(sequences)
        rows.append(
            {
                "cluster": cluster_id,
                "events": sum(map(len, sequences)),
                "traces": len(sequences),
                "unique_acts": len(acts),
                "avg_branching": round(float(np.mean(np.sum(matrix > 0, axis=1))), 3),
                "avg_entropy": round(entropy(matrix), 3),
                "mean_trace_length": round(float(np.mean(list(map(len, sequences)))), 3),
            }
        )
    reproduced = pd.DataFrame(rows)
    reference = pd.read_csv(reference_path).drop(columns=["suggested_label"])
    pd.testing.assert_frame_equal(reproduced, reference, check_dtype=False, rtol=1e-12, atol=1e-12)


def masked_hamming(values: np.ndarray) -> np.ndarray:
    distances = np.zeros((len(values), len(values)), dtype=float)
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            observed = ~np.isnan(values[left]) & ~np.isnan(values[right])
            distance = float(np.mean(values[left, observed] != values[right, observed])) if observed.any() else 1.0
            distances[left, right] = distances[right, left] = distance
    return distances


def reproduce_trajectories(labels: pd.DataFrame, data: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    trajectory = labels.pivot_table(index="userId", columns="topic", values="cluster", aggfunc="first").reindex(columns=TOPICS)
    distances = masked_hamming(trajectory.to_numpy(dtype=float))
    best: tuple[float, int, np.ndarray] | None = None
    for k in range(2, 11):
        candidate = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(distances)
        score = float(silhouette_score(distances, candidate, metric="precomputed"))
        if best is None or score > best[0]:
            best = (score, k, candidate)
    assert best is not None
    groups = pd.DataFrame({"userId": trajectory.index.astype(str), "trajectory_group": best[2].astype(int)})
    reference = pd.read_csv(data / "trajectory" / "user_trajectory_groups.csv", dtype={"userId": str})
    comparison = groups.merge(reference, on="userId", suffixes=("_reproduced", "_reference"), validate="one_to_one")
    ari = float(adjusted_rand_score(comparison["trajectory_group_reference"], comparison["trajectory_group_reproduced"]))
    nmi = float(normalized_mutual_info_score(comparison["trajectory_group_reference"], comparison["trajectory_group_reproduced"]))
    return groups, {
        "k": best[1],
        "silhouette": best[0],
        "ari_vs_reference": ari,
        "nmi_vs_reference": nmi,
        "cluster_sizes": dict(sorted(Counter(map(int, best[2])).items())),
    }


def assert_close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, obtained {actual}.")


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    units = json.loads((data / "da_artifacts" / "per_user_topic_sequences.json").read_text(encoding="utf-8"))
    axes = json.loads((data / "da_artifacts" / "axes_global.json").read_text(encoding="utf-8"))
    states = [str(value) for value in axes["states_global"]]
    superstates = [tuple(map(str, value)) for value in axes["superstates_global"]]
    first, second, keys, total_first, total_second = transition_vectors(units, states, superstates)

    structural = {
        "units": len(units),
        "users": len({str(unit["userId"]) for unit in units}),
        "traces": sum(len(unit["sequences"]) for unit in units),
        "events": sum(len(sequence) for unit in units for sequence in unit["sequences"]),
        "states": len(states),
        "superstates": len(superstates),
        "first_transitions": total_first,
        "second_transitions": total_second,
        "zero_second_order_units": int(np.all(second == 0, axis=1).sum()),
    }
    if structural != EXPECTED:
        raise AssertionError(f"Structural invariants changed: {structural}")

    first_labels, first_embedding, first_silhouette = cluster(first, 10, args.seed)
    second_labels, second_embedding, second_silhouette = cluster(second, 6, args.seed)
    ari1, nmi1, assignments1 = compare_partition(
        keys, first_labels, data / "da_artifacts" / "user_topic_cluster_labels_umap_order1.csv"
    )
    ari2, nmi2, assignments2 = compare_partition(
        keys, second_labels, data / "da_artifacts" / "user_topic_cluster_labels_umap_order2.csv"
    )
    order1_sizes = dict(sorted(Counter(map(int, first_labels)).items()))
    order2_sizes = dict(sorted(Counter(map(int, second_labels)).items()))
    if args.require_exact_umap:
        assert_close(ari1, 1.0, 1e-12)
        assert_close(nmi1, 1.0, 1e-12)
        assert_close(ari2, 1.0, 1e-12)
        assert_close(nmi2, 1.0, 1e-12)
        assert_close(first_silhouette, 0.42985427379608154)
        assert_close(second_silhouette, 0.46370798349380493)
        if order1_sizes != EXPECTED_CLUSTER_SIZES_ORDER1 or order2_sizes != EXPECTED_CLUSTER_SIZES_ORDER2:
            raise AssertionError("Cluster-size invariants changed.")

    reference_order2 = pd.read_csv(
        data / "da_artifacts" / "user_topic_cluster_labels_umap_order2.csv",
        dtype={"userId": str, "topic": str},
    )
    reference_order2["cluster"] = reference_order2["cluster"].astype(int)
    metrics = user_topic_metrics(units, reference_order2)
    validate_user_metrics(metrics, data / "process_mining_outputs" / "user_topic_metrics.csv")
    validate_cluster_metrics(units, reference_order2, data / "process_mining_outputs" / "cluster_metrics.csv")
    trajectory_groups, trajectory_summary = reproduce_trajectories(reference_order2, data)
    if trajectory_summary["k"] != 8 or trajectory_summary["cluster_sizes"] != EXPECTED_TRAJECTORY_SIZES:
        raise AssertionError("Trajectory-cluster invariants changed.")
    assert_close(float(trajectory_summary["silhouette"]), 0.2120734827994361)
    assert_close(float(trajectory_summary["ari_vs_reference"]), 1.0, 1e-12)

    assignments1.assign(umap_x=first_embedding[:, 0], umap_y=first_embedding[:, 1]).to_csv(
        output / "order1_assignments.csv", index=False
    )
    assignments2.assign(umap_x=second_embedding[:, 0], umap_y=second_embedding[:, 1]).to_csv(
        output / "order2_assignments.csv", index=False
    )
    metrics.to_csv(output / "user_topic_metrics.csv", index=False)
    trajectory_groups.to_csv(output / "trajectory_groups.csv", index=False)

    summary = {
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "umap": umap.__version__,
        },
        "seed": args.seed,
        "exact_umap_required": args.require_exact_umap,
        "structural_invariants": structural,
        "order1": {
            "k": 10,
            "silhouette": first_silhouette,
            "ari_vs_reference": ari1,
            "nmi_vs_reference": nmi1,
            "cluster_sizes": order1_sizes,
        },
        "order2": {
            "k": 6,
            "silhouette": second_silhouette,
            "ari_vs_reference": ari2,
            "nmi_vs_reference": nmi2,
            "cluster_sizes": order2_sizes,
        },
        "trajectory": trajectory_summary,
        "process_metrics_match_reference": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
