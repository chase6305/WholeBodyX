"""Create a self-contained Plotly comparison report from H1 dynamics CSVs.

Requires the optional `plotly` package; the controller itself does not use it.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path):
    fields = ("time", "target_com_y_m", "com_y_m", "com_error_m", "left_fz_n", "right_fz_n",
              "contact_position_error_m", "applied_push_y_n")
    with Path(path).open() as stream:
        data = np.array([[float(row[key]) for key in fields] for row in csv.DictReader(stream)])
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("report CSV must contain at least two finite samples and increasing times")
    return {key: data[:, i] for i, key in enumerate(fields)}


def compare(wbc, mpc):
    """Reject incompatible experiments before reporting relative performance."""
    datasets = {"WBC": load(wbc), "MPC + WBC": load(mpc)}
    first, second = datasets.values()
    for key in ("time", "target_com_y_m", "applied_push_y_n"):
        if first[key].shape != second[key].shape or not np.allclose(first[key], second[key], rtol=0., atol=1e-9):
            raise ValueError(f"comparison requires matching {key}; use the same timing, target and disturbance")
    summary = {}
    for label, data in datasets.items():
        force_steps = np.diff(np.column_stack((data["left_fz_n"], data["right_fz_n"])), axis=0)
        intervals = np.diff(data["time"])
        summary[label] = dict(samples=len(data["time"]), duration_s=float(data["time"][-1]),
                              sample_interval_min_s=float(intervals.min()), sample_interval_max_s=float(intervals.max()),
                              com_rmse_mm=float(np.sqrt(np.mean(data["com_error_m"]**2)) * 1000),
                              max_com_error_mm=float(np.max(data["com_error_m"]) * 1000),
                              min_normal_force_n=float(min(data["left_fz_n"].min(), data["right_fz_n"].min())),
                              max_foot_drift_um=float(data["contact_position_error_m"].max() * 1e6),
                              max_force_step_n=float(np.abs(force_steps).max()),
                              force_step_rms_n=float(np.sqrt(np.mean(force_steps**2))))
    return datasets, summary


def report(wbc, mpc, output):
    datasets, summary = compare(wbc, mpc)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Lateral CoM", "Normal support forces",
                                                      "CoM tracking error", "Foot contact position drift"),
                        vertical_spacing=.14, horizontal_spacing=.10)
    for index, (label, data) in enumerate(datasets.items()):
        color = ("#2563eb", "#9333ea")[index]
        stride = max(1, len(data["time"]) // 4000)
        t = data["time"][::stride]
        def line(values, name, row, col, dash=None):
            fig.add_trace(go.Scatter(x=t, y=values[::stride], name=name, legendgroup=label,
                                    mode="lines", line=dict(color=color, width=2, dash=dash)), row=row, col=col)
        line(data["com_y_m"] * 100, label, 1, 1)
        line(data["left_fz_n"], label + " / left", 1, 2)
        line(data["right_fz_n"], label + " / right", 1, 2, "dot")
        line(data["com_error_m"] * 1000, label + " error", 2, 1)
        line(data["contact_position_error_m"] * 1e6, label + " foot drift", 2, 2)
    first = datasets["WBC"]
    stride = max(1, len(first["time"]) // 4000)
    fig.add_trace(go.Scatter(x=first["time"][::stride], y=first["target_com_y_m"][::stride] * 100,
                            name="Requested CoM", mode="lines", line=dict(color="#64748b", dash="dash")), row=1, col=1)
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="Simulation time (s)", row=row, col=col)
    for row, col, unit in ((1, 1, "cm"), (1, 2, "N"), (2, 1, "mm"), (2, 2, "µm")):
        fig.update_yaxes(title_text=unit, row=row, col=col)
    fig.update_layout(title=dict(text="H1: torque WBC and fixed-height CoM/ZMP preview MPC", x=.04),
                      height=950, template="plotly_white", font=dict(family="Arial, sans-serif", size=13),
                      legend=dict(orientation="h", y=-.16), margin=dict(l=70, r=40, t=100, b=130))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(include_plotlyjs=True, full_html=True, config={"displaylogo": False})
    note = ('<p style="font:14px Arial;padding:12px 40px;background:#f1f5f9">'
            'Fixed double support, no contact switching. MPC uses a constant-height approximation and may track '
            'more conservatively than direct WBC. These plots compare simulation traces, not hardware accuracy.</p>')
    note += ('<table style="font:14px Arial;margin:12px 40px;border-spacing:20px 8px">'
             '<caption>Matching timing, target and disturbance; force changes use every sample, including pushes.</caption>'
             '<tr><th>Controller</th><th>CoM RMSE (mm)</th><th>Max force step (N)</th>'
             '<th>Force step RMS (N)</th><th>Sample interval range (ms)</th></tr>')
    for label, metrics in summary.items():
        note += (f'<tr><th>{label}</th><td>{metrics["com_rmse_mm"]:.2f}</td>'
                 f'<td>{metrics["max_force_step_n"]:.2f}</td><td>{metrics["force_step_rms_n"]:.2f}</td>'
                 f'<td>{metrics["sample_interval_min_s"] * 1000:.3f}–'
                 f'{metrics["sample_interval_max_s"] * 1000:.3f}</td></tr>')
    note += '</table>'
    html = html.replace("<body>", "<body>" + note, 1)
    output.write_text(html)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wbc", type=Path, required=True)
    parser.add_argument("--mpc", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/h1-dynamics-report.html"))
    args = parser.parse_args()
    report(args.wbc, args.mpc, args.output)
