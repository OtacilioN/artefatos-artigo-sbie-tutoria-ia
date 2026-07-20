#!/usr/bin/env python3
"""Regenera as figuras do artigo a partir dos artefatos fixos do experimento.

O script não executa classificação de atos de fala nem altera rótulos, clusters,
métricas ou trajetórias. Por padrão, redesenha apenas as Figuras 2--4. A UMAP
só é sobrescrita com ``--with-umap`` quando a reconstrução reproduz a partição
salva (ARI=1) e o silhouette publicado, arredondado a três casas.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphviz import Digraph
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.heuristics.variants import classic as heuristics_classic
from pm4py.objects.log.obj import Event, EventLog, Trace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "data"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures" / "reproduced"

TOPICS = [f"a{i}" for i in range(1, 8)]
PROFILE_LABELS = {
    0: "Exploratório/investigativo A",
    1: "Localmente sistemático",
    2: "Linear/diretivo",
    3: "Estratégia mista",
    4: "Exploratório/investigativo B",
    5: "Semi-exploratório",
}

# Paleta tab10 da Figura 1 original. O mesmo cluster mantém a mesma cor.
CLUSTER_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
}
HATCHES = {0: "///", 1: "\\\\\\", 2: "|||", 3: "---", 4: "+++", 5: "xxx"}

EXPECTED_CLUSTER_COUNTS = {0: 191, 1: 117, 2: 43, 3: 197, 4: 184, 5: 184}
EXPECTED_EVENTS = {0: 4005, 1: 339, 2: 261, 3: 3492, 4: 4369, 5: 3898}
EXPECTED_TRACES = {0: 340, 1: 147, 2: 57, 3: 382, 4: 417, 5: 384}
EXPECTED_UNIQUE_ACTS = {0: 24, 1: 19, 2: 12, 3: 24, 4: 26, 5: 26}
EXPECTED_HEURISTIC_NETS = {
    0: {
        "nodes": 24,
        "edges": 117,
        "dfg_edges": 139,
        "edge_frequency": 3545,
        "display_nodes": 16,
        "display_edges": 96,
        "display_frequency": 3515,
        "display_events": 3986,
        "highlight_edges": 25,
        "highlight_frequency": 3186,
    },
    2: {
        "nodes": 10,
        "edges": 11,
        "dfg_edges": 17,
        "edge_frequency": 191,
        "display_nodes": 10,
        "display_edges": 11,
        "display_frequency": 191,
        "display_events": 251,
        "highlight_edges": 11,
        "highlight_frequency": 191,
    },
}
EXPECTED_LINKS = 168
EXPECTED_FLOW_SUM = 614
EXPECTED_STEP_TOTALS = {
    ("a1", "a2"): 54,
    ("a2", "a3"): 113,
    ("a3", "a4"): 127,
    ("a4", "a5"): 118,
    ("a5", "a6"): 104,
    ("a6", "a7"): 98,
}

DA_LABELS_PT = {
    "Statement-non-opinion": "Afirmação não opinativa",
    "Statement-opinion": "Afirmação opinativa",
    "Action-directive": "Diretiva",
    "Yes-No-Question": "Pergunta sim/não",
    "Declarative Yes-No-Question": "Pergunta declarativa",
    "Wh-Question": "Pergunta WH",
    "Open-Question": "Pergunta aberta",
    "Rhetorical-Question": "Pergunta retórica",
    "Quotation": "Citação",
    "Summarize/Reformulate": "Síntese/reformulação",
    "Collaborative Completion": "Completação",
    "Repeat-phrase": "Repetição",
    "Uninterpretable": "Não interpretável",
    "Conventional-opening": "Abertura convencional",
    "Conventional-closing": "Encerramento convencional",
    "Hold Before Answer/Agreement": "Pausa antes da resposta",
    "Acknowledge (Backchannel)": "Sinal de acompanhamento",
    "Signal-non-understanding": "Sinal de não entendimento",
    "Downplayer": "Atenuador",
    "Thanking": "Agradecimento",
    "No Answers": "Resposta negativa",
    "Appreciation": "Apreciação",
    "Agree/Accept": "Concordância/aceitação",
    "Other": "Outro",
}

# São os valores padrão do Heuristics Miner no PM4Py 2.7.18, explicitados para
# que uma mudança futura da biblioteca não altere silenciosamente os modelos.
HEURISTICS_PARAMETERS = {
    heuristics_classic.Parameters.DEPENDENCY_THRESH: 0.5,
    heuristics_classic.Parameters.AND_MEASURE_THRESH: 0.65,
    heuristics_classic.Parameters.MIN_ACT_COUNT: 1,
    heuristics_classic.Parameters.MIN_DFG_OCCURRENCES: 1,
    heuristics_classic.Parameters.DFG_PRE_CLEANING_NOISE_THRESH: 0.05,
    heuristics_classic.Parameters.LOOP_LENGTH_TWO_THRESH: 0.5,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def paths(source: Path) -> dict[str, Path]:
    return {
        "labels": source / "da_artifacts" / "user_topic_cluster_labels_umap_order2.csv",
        "sequences": source / "da_artifacts" / "per_user_topic_sequences.json",
        "axes": source / "da_artifacts" / "axes_global.json",
        "metrics": source / "process_mining_outputs" / "user_topic_metrics.csv",
        "cluster_metrics": source / "process_mining_outputs" / "cluster_metrics.csv",
        "flows": source / "trajectory" / "transition_counts_by_step.csv",
    }


def require_inputs(input_paths: dict[str, Path]) -> None:
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Artefatos ausentes:\n" + "\n".join(missing))


def normalized_counts(series: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in series.value_counts().sort_index().items()}


def validate_inputs(input_paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(input_paths["labels"], dtype={"userId": str, "topic": str})
    labels["cluster"] = labels["cluster"].astype(int)
    if len(labels) != 916:
        raise ValueError(f"Esperadas 916 unidades; encontradas {len(labels)}.")
    if normalized_counts(labels["cluster"]) != EXPECTED_CLUSTER_COUNTS:
        raise ValueError("A distribuição de clusters diverge do resultado publicado.")

    metrics = pd.read_csv(input_paths["metrics"], dtype={"userId": str, "topic": str})
    metrics["cluster"] = metrics["cluster"].astype(int)
    if len(metrics) != 916 or normalized_counts(metrics["cluster"]) != EXPECTED_CLUSTER_COUNTS:
        raise ValueError("As métricas estudante-atividade não correspondem aos 916 rótulos.")

    cluster_metrics = pd.read_csv(input_paths["cluster_metrics"])
    events = {
        int(row.cluster): int(row.events)
        for row in cluster_metrics[["cluster", "events"]].itertuples(index=False)
    }
    if events != EXPECTED_EVENTS or sum(events.values()) != 16364:
        raise ValueError("As contagens de mensagens por cluster foram alteradas.")
    traces = {
        int(row.cluster): int(row.traces)
        for row in cluster_metrics[["cluster", "traces"]].itertuples(index=False)
    }
    unique_acts = {
        int(row.cluster): int(row.unique_acts)
        for row in cluster_metrics[["cluster", "unique_acts"]].itertuples(index=False)
    }
    if traces != EXPECTED_TRACES or unique_acts != EXPECTED_UNIQUE_ACTS:
        raise ValueError("Os traços ou atos de fala por cluster divergem dos artefatos publicados.")

    return labels, metrics


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def profile_legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=CLUSTER_COLORS[c],
            edgecolor="black",
            hatch=HATCHES[c],
            label=f"C{c} — {PROFILE_LABELS[c]}",
        )
        for c in range(6)
    ]


def generate_boxplots(metrics: pd.DataFrame, output: Path) -> None:
    specs = [
        ("turns", "Mensagens por estudante-atividade", "Mensagens"),
        ("avg_entropy", "Entropia por estudante-atividade", "Entropia"),
        ("avg_branching", "Fator de ramificação por estudante-atividade", "Fator de ramificação"),
        ("mean_trace_length", "Comprimento médio das conversas", "Mensagens"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    fig.suptitle("Distribuição das métricas por perfil", fontsize=12, fontweight="bold", y=0.975)

    for ax, (column, title, ylabel) in zip(axes.flat, specs):
        values = [metrics.loc[metrics["cluster"] == c, column].astype(float).to_numpy() for c in range(6)]
        boxplot = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.58,
            whis=1.5,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "white"},
        )

        for c, box in enumerate(boxplot["boxes"]):
            box.set(facecolor=mcolors.to_rgba(CLUSTER_COLORS[c], 0.42), edgecolor=CLUSTER_COLORS[c], linewidth=1.2)
            box.set_hatch(HATCHES[c])
        for c in range(6):
            for line in (boxplot["whiskers"][2 * c], boxplot["whiskers"][2 * c + 1], boxplot["caps"][2 * c], boxplot["caps"][2 * c + 1]):
                line.set(color=CLUSTER_COLORS[c], linewidth=1.0)
            boxplot["fliers"][c].set(markeredgecolor=CLUSTER_COLORS[c])

        ax.set_title(title, pad=5, fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(
            range(1, 7),
            [f"C{c}\nn={EXPECTED_CLUSTER_COUNTS[c]}" for c in range(6)],
        )
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    fig.legend(
        handles=profile_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.04, 0.012, 0.92, 0.16),
        mode="expand",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.23, hspace=0.52, wspace=0.28)
    save_figure(fig, output, "boxplots_user_topic_metrics")


def readable_da_label(label: str, width: int = 20) -> str:
    translated = DA_LABELS_PT.get(label, label.replace("-", " "))
    return "\n".join(textwrap.wrap(translated, width=width))


def lighten(color: str, amount: float = 0.82) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def build_event_dataframe(labels: pd.DataFrame, sequences_path: Path) -> pd.DataFrame:
    """Reproduz a expansão e o merge realizados no notebook de processo."""
    raw = json.loads(sequences_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for item in raw:
        user_id = str(item["userId"])
        topic = str(item["topic"])
        for sequence_index, sequence in enumerate(item.get("sequences", [])):
            for turn, dialogue_act in enumerate(sequence, start=1):
                rows.append(
                    {
                        "userId": user_id,
                        "topic": topic,
                        "seq_index": sequence_index,
                        "turn": turn,
                        "dialogue_act": str(dialogue_act),
                    }
                )

    events = pd.DataFrame(rows).merge(labels, on=["userId", "topic"], how="inner")
    events["cluster"] = events["cluster"].astype(int)
    if len(events) != 16364 or normalized_counts(events["cluster"]) != EXPECTED_EVENTS:
        raise ValueError("A expansão das sequências não reproduziu as 16.364 mensagens publicadas.")
    return events


def build_pm4py_log(events: pd.DataFrame) -> EventLog:
    """Reproduz um traço por (estudante, atividade, sequência)."""
    log = EventLog()
    for (_, _, sequence_index), group in events.groupby(
        ["userId", "topic", "seq_index"], sort=True
    ):
        trace = Trace()
        for row in group.sort_values("turn").itertuples(index=False):
            trace.append(
                Event(
                    {
                        "concept:name": str(row.dialogue_act),
                        "userId": str(row.userId),
                        "topic": str(row.topic),
                        "cluster": int(row.cluster),
                        "turn": int(row.turn),
                        "seq_index": int(sequence_index),
                    }
                )
            )
        log.append(trace)
    return log


def heuristic_edges(heuristic_net: object) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for source_name, source_node in heuristic_net.nodes.items():
        for target_node, connections in source_node.output_connections.items():
            for edge in connections:
                edges.append(
                    {
                        "source": str(source_name),
                        "target": str(target_node.node_name),
                        "frequency": int(edge.dfg_value),
                        "dependency": float(edge.dependency_value),
                    }
                )
    return edges


def render_heuristic_net(
    heuristic_net: object,
    activity_counts: dict[str, int],
    cluster: int,
    panel: str,
    output: Path,
) -> tuple[int, int, int, int, float]:
    """Diagrama um recorte visual auditável do HeuristicsNet completo.

    Em C0, o recorte retém os 16 atos mais frequentes no log e todas as
    conexões do modelo entre eles. Empates na frequência do ato são resolvidos
    pela soma das frequências das arestas incidentes e, por fim, pelo nome.
    """
    edges = heuristic_edges(heuristic_net)
    expected = EXPECTED_HEURISTIC_NETS[cluster]
    if (
        len(heuristic_net.nodes) != expected["nodes"]
        or len(edges) != expected["edges"]
        or len(heuristic_net.dfg) != expected["dfg_edges"]
    ):
        raise ValueError(
            f"Heuristic Net C{cluster} divergente: {len(heuristic_net.nodes)} nós, "
            f"{len(edges)} arestas e {len(heuristic_net.dfg)} arestas no DFG limpo."
        )

    ordered = sorted(
        edges,
        key=lambda edge: (
            -int(edge["frequency"]),
            str(edge["source"]),
            str(edge["target"]),
        ),
    )
    incident_frequency: defaultdict[str, int] = defaultdict(int)
    for edge in edges:
        incident_frequency[str(edge["source"])] += int(edge["frequency"])
        incident_frequency[str(edge["target"])] += int(edge["frequency"])

    ranked_nodes = sorted(
        (str(node_name) for node_name in heuristic_net.nodes),
        key=lambda node_name: (
            -activity_counts.get(node_name, 0),
            -incident_frequency[node_name],
            node_name,
        ),
    )
    displayed_nodes = set(ranked_nodes[: expected["display_nodes"]])
    displayed = [
        edge
        for edge in ordered
        if str(edge["source"]) in displayed_nodes
        and str(edge["target"]) in displayed_nodes
    ]
    highlighted = displayed[: expected["highlight_edges"]]
    highlighted_keys = {
        (str(edge["source"]), str(edge["target"]))
        for edge in highlighted
    }
    edge_frequency = sum(int(edge["frequency"]) for edge in edges)
    display_frequency = sum(int(edge["frequency"]) for edge in displayed)
    highlight_frequency = sum(int(edge["frequency"]) for edge in highlighted)
    display_events = sum(activity_counts.get(node_name, 0) for node_name in displayed_nodes)
    if (
        edge_frequency != expected["edge_frequency"]
        or len(displayed_nodes) != expected["display_nodes"]
        or len(displayed) != expected["display_edges"]
        or display_frequency != expected["display_frequency"]
        or highlight_frequency != expected["highlight_frequency"]
        or display_events != expected["display_events"]
    ):
        raise ValueError(f"O recorte visual do Heuristic Net C{cluster} divergiu dos invariantes.")

    max_frequency = max(int(edge["frequency"]) for edge in displayed)
    color = CLUSTER_COLORS[cluster]
    engine = "circo" if cluster == 0 else "dot"
    graph_attributes = {
        "bgcolor": "white",
        "fontname": "Helvetica",
        "fontsize": "15",
        "label": f"({panel}) C{cluster} — {PROFILE_LABELS[cluster]}",
        "labelloc": "t",
        "margin": "0",
        "outputorder": "edgesfirst",
        "pad": "0.10",
    }
    if cluster == 0:
        graph_attributes.update(
            {
                "mindist": "0.5",
                "overlap": "false",
                "pad": "0.04",
                "ratio": "0.75",
                "splines": "line",
            }
        )
    else:
        graph_attributes.update(
            {
                "nodesep": "0.18",
                "rankdir": "TB",
                "ranksep": "0.30",
                "splines": "spline",
            }
        )

    graph = Digraph(
        name=f"heuristic_net_c{cluster}",
        engine=engine,
        graph_attr=graph_attributes,
        node_attr={
            "color": color,
            "fillcolor": mcolors.to_hex(lighten(color, 0.88)),
            "fontcolor": "#202124",
            "fontname": "Helvetica",
            "fontsize": "13",
            "margin": "0.06,0.04" if cluster == 0 else "0.08,0.05",
            "penwidth": "1.1",
            "shape": "box",
            "style": "rounded,filled",
        },
        edge_attr={
            "arrowhead": "normal",
            "arrowsize": "0.55",
            "fontname": "Helvetica",
        },
    )

    node_ids = {
        name: f"n{index}"
        for index, name in enumerate(sorted(displayed_nodes))
    }
    for name in sorted(displayed_nodes):
        graph.node(
            node_ids[name],
            label=readable_da_label(name, width=14 if cluster == 0 else 20),
        )

    for edge in displayed:
        source = str(edge["source"])
        target = str(edge["target"])
        frequency = int(edge["frequency"])
        dependency = float(edge["dependency"])
        normalized = math.log1p(frequency) / math.log1p(max_frequency)
        is_highlighted = (source, target) in highlighted_keys
        graph.edge(
            node_ids[source],
            node_ids[target],
            color=color if is_highlighted else "#9aa7b2",
            penwidth=f"{0.55 + 4.0 * normalized:.2f}",
            style="solid" if dependency >= 0.5 else "dashed",
        )

    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"cluster_{cluster}_heuristic_net"
    graph.format = "pdf"
    graph.render(filename=str(stem), cleanup=True)
    graph.attr(dpi="400")
    graph.format = "png"
    graph.render(filename=str(stem), cleanup=True)
    return (
        len(heuristic_net.nodes),
        len(edges),
        len(displayed_nodes),
        len(displayed),
        display_frequency / edge_frequency,
    )


def generate_process_network(
    events: pd.DataFrame,
    cluster: int,
    panel: str,
    output: Path,
) -> tuple[int, int, int, int, float]:
    cluster_events = events.loc[events["cluster"] == cluster].copy()
    log = build_pm4py_log(cluster_events)
    if len(cluster_events) != EXPECTED_EVENTS[cluster] or len(log) != EXPECTED_TRACES[cluster]:
        raise ValueError(f"O log C{cluster} não corresponde aos eventos e traços publicados.")
    heuristic_net = heuristics_miner.apply_heu(log, parameters=HEURISTICS_PARAMETERS)
    activity_counts = {
        str(dialogue_act): int(count)
        for dialogue_act, count in cluster_events["dialogue_act"].value_counts().items()
    }
    return render_heuristic_net(
        heuristic_net,
        activity_counts,
        cluster,
        panel,
        output,
    )


def ribbon_path(
    source_x: float,
    target_x: float,
    source_low: float,
    source_high: float,
    target_low: float,
    target_high: float,
) -> MplPath:
    mid_x = (source_x + target_x) / 2.0
    vertices = [
        (source_x, source_low),
        (mid_x, source_low),
        (mid_x, target_low),
        (target_x, target_low),
        (target_x, target_high),
        (mid_x, target_high),
        (mid_x, source_high),
        (source_x, source_high),
        (source_x, source_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def generate_longitudinal_flows(flows: pd.DataFrame, output: Path) -> None:
    flows = flows.copy()
    flows["from_cluster"] = flows["from_cluster"].astype(int)
    flows["to_cluster"] = flows["to_cluster"].astype(int)
    flows["count"] = flows["count"].astype(int)

    if len(flows) != EXPECTED_LINKS or int(flows["count"].sum()) != EXPECTED_FLOW_SUM:
        raise ValueError("As ligações longitudinais divergem dos artefatos publicados.")
    step_totals = {
        (str(a), str(b)): int(value)
        for (a, b), value in flows.groupby(["from_topic", "to_topic"])["count"].sum().items()
    }
    if step_totals != EXPECTED_STEP_TOTALS:
        raise ValueError(f"Totais por etapa inesperados: {step_totals}")

    incoming: defaultdict[tuple[str, int], int] = defaultdict(int)
    outgoing: defaultdict[tuple[str, int], int] = defaultdict(int)
    for row in flows.itertuples(index=False):
        outgoing[(str(row.from_topic), int(row.from_cluster))] += int(row.count)
        incoming[(str(row.to_topic), int(row.to_cluster))] += int(row.count)

    capacities = {
        (topic, cluster): max(incoming[(topic, cluster)], outgoing[(topic, cluster)])
        for topic in TOPICS
        for cluster in range(6)
    }
    if len(capacities) != 42 or any(value <= 0 for value in capacities.values()):
        raise ValueError("A figura longitudinal deve conter 42 nós não vazios.")

    gap = 0.014
    max_column_total = max(sum(capacities[(topic, c)] for c in range(6)) for topic in TOPICS)
    unit_height = (0.74 - 5 * gap) / max_column_total
    x_positions = {topic: x for topic, x in zip(TOPICS, np.linspace(0.05, 0.95, 7))}
    node_ranges: dict[tuple[str, int], tuple[float, float]] = {}

    for topic in TOPICS:
        column_height = sum(capacities[(topic, c)] for c in range(6)) * unit_height + 5 * gap
        cursor = 0.87 + column_height / 2.0 - 0.37
        for cluster in range(6):
            height = capacities[(topic, cluster)] * unit_height
            node_ranges[(topic, cluster)] = (cursor - height, cursor)
            cursor -= height + gap

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    node_width = 0.024
    for from_topic, to_topic in zip(TOPICS, TOPICS[1:]):
        step = flows.loc[
            (flows["from_topic"].astype(str) == from_topic)
            & (flows["to_topic"].astype(str) == to_topic)
        ].sort_values(["from_cluster", "to_cluster"])
        source_cursor = {
            (from_topic, c): node_ranges[(from_topic, c)][0]
            for c in range(6)
        }
        target_cursor = {
            (to_topic, c): node_ranges[(to_topic, c)][0]
            for c in range(6)
        }

        for row in step.itertuples(index=False):
            source = (from_topic, int(row.from_cluster))
            target = (to_topic, int(row.to_cluster))
            height = int(row.count) * unit_height
            source_low = source_cursor[source]
            target_low = target_cursor[target]
            source_cursor[source] += height
            target_cursor[target] += height
            path = ribbon_path(
                x_positions[from_topic] + node_width / 2,
                x_positions[to_topic] - node_width / 2,
                source_low,
                source_low + height,
                target_low,
                target_low + height,
            )
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=CLUSTER_COLORS[source[1]],
                    edgecolor="none",
                    alpha=0.26,
                    zorder=1,
                )
            )

    for topic_index, topic in enumerate(TOPICS, start=1):
        x = x_positions[topic]
        ax.text(x, 0.965, f"Atividade {topic_index}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        for cluster in range(6):
            low, high = node_ranges[(topic, cluster)]
            color = CLUSTER_COLORS[cluster]
            ax.add_patch(
                Rectangle(
                    (x - node_width / 2, low),
                    node_width,
                    high - low,
                    facecolor=color,
                    edgecolor="#333333",
                    linewidth=0.45,
                    zorder=3,
                )
            )
            luminance = np.dot(mcolors.to_rgb(color), [0.2126, 0.7152, 0.0722])
            ax.text(
                x,
                (low + high) / 2,
                f"C{cluster}",
                ha="center",
                va="center",
                fontsize=5.1,
                color="black" if luminance > 0.55 else "white",
                fontweight="bold",
                zorder=4,
            )

    fig.text(
        0.5,
        0.19,
        "Espessura dos fluxos proporcional ao número de estudantes",
        ha="center",
        va="center",
        fontsize=7.0,
        color="#444444",
    )
    fig.legend(
        handles=profile_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.03, 0.005, 0.94, 0.16),
        mode="expand",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.22)
    save_figure(fig, output, "sankey")


def second_order_probability_matrix(
    sequences: list[list[str]],
    state_index: dict[str, int],
    superstate_index: dict[tuple[str, str], int],
) -> np.ndarray:
    matrix = np.zeros((len(superstate_index), len(state_index)), dtype=np.float64)
    for sequence in sequences:
        for first, second, following in zip(sequence, sequence[1:], sequence[2:]):
            superstate = (first, second)
            if superstate in superstate_index and following in state_index:
                matrix[superstate_index[superstate], state_index[following]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    np.divide(matrix, row_sums, out=matrix, where=row_sums != 0)
    return matrix


def generate_umap_if_reproduced(
    labels: pd.DataFrame,
    input_paths: dict[str, Path],
    output: Path,
) -> tuple[float, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    axes = json.loads(input_paths["axes"].read_text(encoding="utf-8"))
    states = [str(value) for value in axes["states_global"]]
    superstates = [tuple(map(str, value)) for value in axes["superstates_global"]]
    if len(states) != 29 or len(superstates) != 276:
        raise ValueError("Os eixos globais não correspondem a 29 estados e 276 superestados.")

    sequences = json.loads(input_paths["sequences"].read_text(encoding="utf-8"))
    if len(sequences) != 916:
        raise ValueError("A reconstrução da UMAP exige as 916 unidades preservadas.")

    label_lookup = labels.set_index(["userId", "topic"])["cluster"]
    state_index = {state: index for index, state in enumerate(states)}
    superstate_index = {state: index for index, state in enumerate(superstates)}
    vectors: list[np.ndarray] = []
    saved_labels: list[int] = []
    for item in sequences:
        key = (str(item["userId"]), str(item["topic"]))
        if key not in label_lookup.index:
            raise ValueError(f"Unidade sem rótulo preservado: {key}")
        matrix = second_order_probability_matrix(item.get("sequences", []), state_index, superstate_index)
        vectors.append(matrix.ravel())
        saved_labels.append(int(label_lookup.loc[key]))

    features = StandardScaler().fit_transform(np.vstack(vectors))
    embedding = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42,
    ).fit_transform(features)
    reproduced = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(embedding)
    ari = float(adjusted_rand_score(saved_labels, reproduced))
    silhouette = float(silhouette_score(embedding, saved_labels))
    if not math.isclose(ari, 1.0, abs_tol=1e-12) or round(silhouette, 3) != 0.464:
        raise RuntimeError(
            "A UMAP não foi sobrescrita: a execução atual não reproduziu integralmente "
            f"a partição publicada (ARI={ari:.6f}; silhouette={silhouette:.6f})."
        )

    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    saved = np.asarray(saved_labels)
    for cluster in range(6):
        points = embedding[saved == cluster]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.82,
            color=CLUSTER_COLORS[cluster],
            edgecolors="none",
            label=f"C{cluster} — {PROFILE_LABELS[cluster]}",
        )
    ax.set_title("Projeção UMAP dos vetores de segunda ordem", fontweight="bold")
    ax.set_xlabel("Dimensão 1")
    ax.set_ylabel("Dimensão 2")
    ax.grid(color="#e6e6e6", linewidth=0.5)
    ax.legend(loc="best", frameon=True, fontsize=6.6)
    fig.tight_layout()
    save_figure(fig, output, "umap_clusters_2_ordem")
    return ari, silhouette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Diretório de dados derivados (padrão: data/ no repositório).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Diretório de saída das figuras PDF e PNG.",
    )
    parser.add_argument(
        "--with-umap",
        action="store_true",
        help="Tenta regenerar a Figura 1, abortando se os resultados não forem reproduzidos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    input_paths = paths(args.source.resolve())
    require_inputs(input_paths)
    labels, metrics = validate_inputs(input_paths)
    events = build_event_dataframe(labels, input_paths["sequences"])

    generate_boxplots(metrics, args.output)
    c0 = generate_process_network(events, cluster=0, panel="a", output=args.output)
    c2 = generate_process_network(events, cluster=2, panel="b", output=args.output)
    generate_longitudinal_flows(pd.read_csv(input_paths["flows"]), args.output)

    print("Figuras 2--4 regeneradas; 916 unidades e 16.364 mensagens preservadas.")
    print(
        "Heuristic Nets PM4Py 2.7.18: "
        f"C0={c0[0]} nós/{c0[1]} arestas, recorte={c0[2]}/{c0[3]} "
        f"({c0[4]:.1%} da frequência); C2={c2[0]} nós/{c2[1]} arestas, "
        f"recorte={c2[2]}/{c2[3]} ({c2[4]:.1%})."
    )
    print("Fluxos longitudinais: 42 nós, 168 ligações e soma 614.")
    if args.with_umap:
        ari, silhouette = generate_umap_if_reproduced(labels, input_paths, args.output)
        print(f"Figura 1 regenerada: ARI={ari:.6f}; silhouette={silhouette:.6f}.")


if __name__ == "__main__":
    main()
