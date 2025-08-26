# annotate_agreement_app.py
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


# -----------------------------
# Configuration (edit these)
# -----------------------------
DATASET_NAME = "BramVanroy/c5-topics-500k-full"
SPLIT = "train"
SAMPLE_SIZE = 250
SEED = 42

# If your columns differ, the app will try these fallbacks automatically.
text_col = "text_short"
topic_col = "topic"


# -----------------------------
# Utilities
# -----------------------------
def _wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class AppState:
    df: pd.DataFrame
    texts: List[str]
    gold_ids: List[int]
    id2label: Dict[int, str]
    label2id: Dict[str, int]
    topics_sorted: List[str]
    idx: int
    preds: List[Optional[int]]
    total: int


def _generate_balanced_sample(ds: Dataset, class_ids: list[int], samples_per_class: int) -> Dataset:
    """Generate a balanced sample from the dataset, i.e. each class should occur at most `samples_per_class` times."""
    sampled_dfs = []
    for c in class_ids:
        ds_c = ds.filter(lambda x: x[topic_col] == c, num_proc=4)
        num_c = min(len(ds_c), samples_per_class)
        if num_c == 0:
            continue

        ds_c_sampled = ds_c.shuffle(seed=SEED).select(range(num_c))
        sampled_dfs.append(ds_c_sampled)
    return concatenate_datasets(sampled_dfs).shuffle(seed=SEED)


def _prepare_state(dataset_name: str, split: str, sample_size: int = 250, seed: int = 42) -> AppState:
    ds = load_dataset(dataset_name, split=split)
    print(ds)
    ds = ds.filter(lambda lang: lang in ("nld",), input_columns="language", num_proc=32).select_columns(
        [text_col, topic_col]
    )
    categories = ds.unique(topic_col)
    ds = ds.cast_column(topic_col, ClassLabel(names=categories)).shuffle(seed=seed)
    category_ids = ds.unique(topic_col)
    samples_per_class = max(1, sample_size // len(categories))
    ds = _generate_balanced_sample(ds, category_ids, samples_per_class)

    df_sample = ds.to_pandas()
    texts = df_sample[text_col].astype(str).tolist()
    gold_ids = df_sample[topic_col]

    id2label = dict(enumerate(ds.features[topic_col]._int2str))
    label2id = ds.features[topic_col]._str2int

    print(id2label)
    print(label2id)

    return AppState(
        df=df_sample,
        texts=texts,
        gold_ids=gold_ids,
        id2label=id2label,
        label2id=label2id,
        topics_sorted=sorted(label2id.keys()),
        idx=0,
        preds=[None] * len(df_sample),
        total=len(df_sample),
    )


def _progress_text(state: AppState) -> str:
    annotated = sum(p is not None for p in state.preds)
    return f"**Sample {state.idx + 1} of {state.total}** &nbsp;|&nbsp; **Annotated:** {annotated}/{state.total}"


def _current_text(state: AppState) -> str:
    return state.texts[state.idx]


def _current_selection(state: AppState) -> Optional[str]:
    pi = state.preds[state.idx]
    if pi is None:
        return None
    return state.id2label[pi]


def _set_selection(state: AppState, choice: Optional[str]) -> None:
    # Update preds and update df with "human_label" column at right index
    if choice is None:
        state.preds[state.idx] = None
        state.df.at[state.idx, "human_label"] = None
    else:
        state.preds[state.idx] = state.label2id[str(choice)]
        state.df.at[state.idx, "human_label"] = state.label2id[str(choice)]


def _compute_agreement(state: AppState, mode: str = "display"):
    """
    mode='display' -> returns (summary_md:str, per_class_df:pd.DataFrame, cm_df:pd.DataFrame)
    mode='download' -> returns (stats_json_bytes:bytes, cm_txt_bytes:bytes)
    """
    # Use only annotated positions
    y_true = []
    y_pred = []
    for g, p in zip(state.gold_ids, state.preds):
        if p is not None:
            y_true.append(int(g))
            y_pred.append(int(p))
    n_total = state.total
    n_eval = len(y_true)

    if mode == "display" and n_eval == 0:
        summary_md = "No annotations yet. Annotate some samples and click **Agreement** again."
        return summary_md, pd.DataFrame(), pd.DataFrame()
    if mode == "download" and n_eval == 0:
        raise gr.Error("No annotations yet. Annotate some samples and try again.")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")

    # Wilson CI for accuracy
    k = int(round(acc * n_eval))
    lo, hi = _wilson_ci(k, n_eval, confidence=0.95)

    # Per-class report
    labels = sorted(state.id2label.keys())
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[state.id2label[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    per_class_recalls = []
    for lab_id in labels:
        name = state.id2label[lab_id]
        stats = report.get(name, {})
        per_class_rows.append(
            {
                "Topic": name,
                "Precision": round(float(stats.get("precision", 0.0)), 6),
                "Recall": round(float(stats.get("recall", 0.0)), 6),
                "F1": round(float(stats.get("f1-score", 0.0)), 6),
                "Support": int(stats.get("support", 0)),
            }
        )
        per_class_recalls.append(float(stats.get("recall", 0.0)))

    per_class_df = pd.DataFrame(per_class_rows)
    variance_across_topics = float(np.var(per_class_recalls)) if per_class_recalls else float("nan")

    # Confusion matrix (numpy + pretty DataFrame)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"gold:{state.id2label[i]}" for i in labels],
        columns=[f"pred:{state.id2label[i]}" for i in labels],
    )

    if mode == "display":
        summary_md = (
            f"### Inter-Annotator Agreement (vs. gold)\n"
            f"- **Coverage:** {n_eval}/{n_total} annotated\n"
            f"- **Accuracy:** {acc:.4f}  \n"
            f"  - **95% Wilson CI:** [{lo:.4f}, {hi:.4f}] (binomial, n={n_eval})\n"
            f"- **Balanced Accuracy:** {bal_acc:.4f}\n"
            f"- **Cohen's κ:** {kappa:.4f}\n"
            f"- **F1 (macro):** {f1_macro:.4f} &nbsp;&nbsp; **F1 (micro):** {f1_micro:.4f}\n"
            f"- **Variance across per-topic recalls (lower is more consistent):** {variance_across_topics:.6f}\n"
            f"\n**Per-topic metrics** appear in the table below, and the **confusion matrix** below that."
        )
        return summary_md, per_class_df, cm_df

    # mode == 'download' -> build JSON + TXT
    label_names = [state.id2label[i] for i in labels]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "coverage": {"annotated": n_eval, "total": n_total},
        "metrics": {
            "accuracy": round(float(acc), 6),
            "accuracy_ci_95": [round(float(lo), 6), round(float(hi), 6)],
            "balanced_accuracy": round(float(bal_acc), 6),
            "cohens_kappa": round(float(kappa), 6),
            "f1_macro": round(float(f1_macro), 6),
            "f1_micro": round(float(f1_micro), 6),
            "variance_recall_across_topics": round(float(variance_across_topics), 8),
        },
        "labels": label_names,
        "per-topic": per_class_rows,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
    }
    stats_json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    header = [""] + [f"pred:{n}" for n in label_names]
    lines = ["\t".join(header)]
    for i, row in enumerate(cm):
        lines.append("\t".join([f"gold:{label_names[i]}"] + [str(x) for x in row.tolist()]))
    cm_txt_bytes = "\n".join(lines).encode("utf-8")

    return stats_json_bytes, cm_txt_bytes


def _bytes_to_tmpfile(content: bytes, suffix: str, stem: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{stem}_")
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path


# -----------------------------
# Gradio App
# -----------------------------
with gr.Blocks(
    elem_id="app-root",
    title="Topics annotation & agreement",
    css="#app-root {max-width: 900px; margin: auto}; #short_text {overflow: auto !important;}",
) as demo:
    gr.Markdown("## Topics annotation & agreement")

    # State
    state = gr.State()  # will hold AppState

    with gr.Row():
        progress_md = gr.Markdown(value="")

    with gr.Row():
        with gr.Column(scale=3):
            text_out = gr.Textbox(elem_id="short_text", label="Text", lines=10, interactive=False, autoscroll=False)
        with gr.Column(scale=1):
            topics_radio = gr.Radio(choices=[], label="Your topic", interactive=True)
            with gr.Row():
                prev_btn = gr.Button("Previous", variant="secondary")
                next_btn = gr.Button("Next")

    with gr.Row():
        agree_btn = gr.Button("Agreement", variant="primary")
        # ↓ replace DownloadButtons with plain Buttons
        download_stats_btn = gr.Button("Create agreement stats (JSON)")
        download_cm_btn = gr.Button("Create confusion matrix (TXT)")
        download_df_btn = gr.Button("Create annotations (Excel)")

    # Add File outputs (start hidden so the UI stays clean until a file is ready)
    with gr.Row():
        stats_file = gr.File(label="agreement_stats.json", visible=False)
        cm_file = gr.File(label="confusion_matrix.txt", visible=False)
        ann_file = gr.File(label="annotations.xlsx", visible=False)

    agree_md = gr.Markdown()
    per_class_df_out = gr.Dataframe(label="Per-topic metrics", interactive=False)
    cm_df_out = gr.Dataframe(label="Confusion matrix (rows=gold, cols=pred)", interactive=False)

    # --------- Handlers ---------
    def _refresh_ui(st: AppState):
        """Return current UI values from state."""
        return (
            _progress_text(st),
            _current_text(st),
            gr.update(choices=st.topics_sorted, value=_current_selection(st)),
        )

    def init_app():
        st = _prepare_state(DATASET_NAME, SPLIT, SAMPLE_SIZE, SEED)
        return (st, *_refresh_ui(st))

    def reload_app(name, split, n_sample, seed):
        st = _prepare_state(str(name), str(split), int(n_sample), int(seed))
        return (st, *_refresh_ui(st))

    def set_choice(choice, st: AppState):
        _set_selection(st, choice)
        return st

    def go_next(choice, st: AppState):
        _set_selection(st, choice)
        if st.idx < st.total - 1:
            st.idx += 1
        return (st, *_refresh_ui(st))

    def go_prev(choice, st: AppState):
        _set_selection(st, choice)
        if st.idx > 0:
            st.idx -= 1
        return (st, *_refresh_ui(st))

    def do_agreement_display(st: AppState):
        return _compute_agreement(st, mode="display")

    def download_stats_json(st: AppState):
        stats_json, _ = _compute_agreement(st, mode="download")
        path = _bytes_to_tmpfile(stats_json, ".json", "agreement_stats")
        return gr.update(value=path, visible=True)

    def download_confusion_txt(st: AppState):
        _, cm_txt = _compute_agreement(st, mode="download")
        path = _bytes_to_tmpfile(cm_txt, ".txt", "confusion_matrix")
        return gr.update(value=path, visible=True)

    def download_annotations_xlsx(st: AppState):
        st.df["human_label"] = [st.id2label[p] if p is not None else None for p in st.preds]
        buf = io.BytesIO()
        st.df.to_excel(buf, index=False)
        xls_bytes = buf.getvalue()
        path = _bytes_to_tmpfile(xls_bytes, ".xlsx", "annotations")
        return gr.update(value=path, visible=True)

    # wire
    demo.load(fn=init_app, inputs=None, outputs=[state, progress_md, text_out, topics_radio])
    topics_radio.change(fn=set_choice, inputs=[topics_radio, state], outputs=[state])
    next_btn.click(fn=go_next, inputs=[topics_radio, state], outputs=[state, progress_md, text_out, topics_radio])
    prev_btn.click(fn=go_prev, inputs=[topics_radio, state], outputs=[state, progress_md, text_out, topics_radio])
    agree_btn.click(fn=do_agreement_display, inputs=[state], outputs=[agree_md, per_class_df_out, cm_df_out])
    download_stats_btn.click(fn=download_stats_json, inputs=[state], outputs=stats_file)
    download_cm_btn.click(fn=download_confusion_txt, inputs=[state], outputs=cm_file)
    download_df_btn.click(fn=download_annotations_xlsx, inputs=[state], outputs=ann_file)


if __name__ == "__main__":
    demo.queue().launch()
