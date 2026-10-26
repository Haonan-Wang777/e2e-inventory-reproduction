"""Check the new replay against selected, already reviewed author notebook cells.

Only the two exact notebook hashes below are accepted. Import, plotting, private
path, notebook-output, and arbitrary document-instruction execution are excluded.
Historical output-cache agreement is recorded separately from source equivalence.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "author_package"
NOTEBOOK_HASHES = {
    "E2E_Section4.ipynb": "dcdb4d0366b512df9ef8cd8578825cf34367c2a0075fcfe27c08e560d0f9fc87",
    "Field_experiments_summary.ipynb": "fefc36407f0b06903c3d943529dd89eca5792af2bd9b8e235356bb10c13bacf1",
}
OFFLINE_CELLS = [1, 2, 4, 6, 8, 9, 11, 12, 14, 15]
FIELD_CELLS = [
    2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30,
    32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
]
METRICS = ["holding_cost", "stockout_cost", "total_cost", "turnover", "stockout_ratio"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_notebook(filename):
    path = DATA / filename
    digest = sha256(path)
    if digest != NOTEBOOK_HASHES[filename]:
        raise ValueError(f"Notebook changed; review its selected source before execution: {filename}")
    return json.loads(path.read_text(encoding="utf-8"))


class CompatibleDrop(ast.NodeTransformer):
    """Modernize only the reviewed DataFrame.drop('y', 1) call syntax."""

    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "drop"
                and len(node.args) == 2 and not node.keywords
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "y"
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == 1):
            node.args = node.args[:1]
            node.keywords = [ast.keyword(arg="axis", value=ast.Constant(value=1))]
            self.changed += 1
        return node


def run_cell(notebook, filename, index, namespace, compatibility=False):
    allowed = OFFLINE_CELLS if filename == "E2E_Section4.ipynb" else FIELD_CELLS
    if index not in allowed or notebook["cells"][index]["cell_type"] != "code":
        raise ValueError(f"Unreviewed cell: {filename}:{index}")
    tree = ast.parse("".join(notebook["cells"][index]["source"]))
    changes = 0
    if compatibility:
        adapter = CompatibleDrop()
        tree = adapter.visit(tree)
        ast.fix_missing_locations(tree)
        changes = adapter.changed
        if changes != 1:
            raise ValueError("Expected exactly one reviewed drop-axis compatibility change")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, f"{filename}:cell{index}", "exec"), namespace)
    return changes


def import_replay(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"verified_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simple_assignment(node, name):
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)


def main_body(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(node.body for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")


def execute_statements(statements, namespace, label):
    """Evaluate the replay's audited calculation segment without its output code."""
    tree = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, label, "exec"), namespace)


def check_equal(name, actual, expected, checks, exact=False):
    a, e = np.asarray(actual), np.asarray(expected)
    if exact:
        np.testing.assert_array_equal(a, e, err_msg=name)
    else:
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-9, err_msg=name)
    delta = float(np.max(np.abs(a - e))) if a.size else 0.0
    checks.append({"name": name, "values": int(a.size), "max_absolute_error": delta,
                   "exact": exact, "passed": True})


def verify_offline(module, notebook):
    original = {"np": np, "pd": pd, "gamma": gamma}
    for index in OFFLINE_CELLS:
        run_cell(notebook, "E2E_Section4.ipynb", index, original)
    # Evaluate the actual replay statements through creation of its cost table.
    # Parser setup and every filesystem/plot output statement are omitted.
    body = main_body(module)
    start = next(i for i, node in enumerate(body) if simple_assignment(node, "source"))
    end = next(i for i, node in enumerate(body) if simple_assignment(node, "table"))
    replay = dict(module.__dict__)
    execute_statements(body[start:end + 1], replay, "reproduce_offline:calculation_segment")
    checks = []
    expected_groups = original["df_test"]
    actual_groups = replay["grouped"]
    if not expected_groups.index.equals(actual_groups.index):
        raise AssertionError("Offline SKU group order differs")
    curves_checked = 0
    for method in module.METHODS:
        actual_curves = replay["paths"][method]
        expected_curves = expected_groups[f"{method}_agginv"].tolist()
        diag = {"orders": 0, "negative_replenishments": 0, "fallback_inventory_index": 0}
        for group_index, ((sku, row), expected, actual) in enumerate(
                zip(actual_groups.iterrows(), expected_curves, actual_curves)):
            expected_full = expected_groups.iloc[group_index][f"{method}_agginv_f"]
            actual_full, function_curve = module.inventory_path(row, method, diag)
            # Both inventory arrays are checked exactly; no result rounding.
            np.testing.assert_array_equal(actual_full, expected_full,
                                          err_msg=f"{method}/{sku}/full")
            np.testing.assert_array_equal(actual, expected,
                                          err_msg=f"{method}/{sku}/charged")
            np.testing.assert_array_equal(function_curve, expected,
                                          err_msg=f"{method}/{sku}/function")
            curves_checked += 1
    check_equal("offline.all_21_mean_costs", replay["table"].to_numpy(),
                original["df_aggcom"].to_numpy(), checks)
    reference_error = float(np.max(np.abs(original["df_aggcom"].to_numpy() - module.REFERENCE)))
    np.testing.assert_allclose(original["df_aggcom"].to_numpy(), module.REFERENCE,
                               rtol=0, atol=5.1e-7)
    return {
        "passed": True, "reviewed_cells_executed": OFFLINE_CELLS,
        "rows": 3000, "sku_groups": len(actual_groups), "methods": module.METHODS,
        "group_method_curves_checked_exactly": curves_checked,
        "full_and_charged_inventory_arrays_checked": curves_checked * 2,
        "mean_cost_values_checked": 21, "checks": checks,
        "cached_21_values_max_error_after_six_decimal_rounding": reference_error,
        "cost_table_from_current_original_source": original["df_aggcom"].to_dict(),
    }


def original_metrics(namespace, role):
    prefix = "e2e" if role == "treatment" else "jd"
    return pd.DataFrame({
        "holding_cost": namespace[f"{prefix}_holding_cost"],
        "stockout_cost": namespace[f"{prefix}_stockout_cost"],
        "total_cost": namespace[f"{prefix}_total_cost"],
        "turnover": namespace[f"{prefix}_turnover1"],
        "turnover_raw": namespace[f"{prefix}_turnover"],
        "stockout_ratio": namespace[f"{prefix}_stockout_ratio"],
    })


def verify_field(module, notebook):
    original = {"np": np, "pd": pd, "stats": stats, "sm": sm}
    for index in [2, 3, 5, 7, 8, 10, 11]:
        run_cell(notebook, "Field_experiments_summary.ipynb", index, original)
    expected_groups = {
        "treatment_post": original_metrics(original, "treatment"),
        "control_post": original_metrics(original, "control"),
    }
    expected_post = []
    for metric, index in zip(METRICS, [13, 14, 15, 16, 17]):
        run_cell(notebook, "Field_experiments_summary.ipynb", index, original)
        expected_post.append({"metric": metric, "t_statistic": original["t"],
                              "p_value": original["p"]})
    for index in [19, 20, 21, 22]:
        run_cell(notebook, "Field_experiments_summary.ipynb", index, original)
    expected_regression = []
    compatibility_changes = 0
    for metric, index in zip(METRICS, [23, 24, 25, 26, 27]):
        compatibility_changes += run_cell(notebook, "Field_experiments_summary.ipynb", index,
                                          original, compatibility=True)
        fit = original["lr"]
        for term in ["e2e", "vlt", "d"]:
            expected_regression.append({"metric": metric, "term": term,
                "coefficient": fit.params[term], "standard_error": fit.bse[term],
                "t_statistic": fit.tvalues[term], "p_value": fit.pvalues[term]})
    for index in [29, 30, 32, 33, 35, 36]:
        run_cell(notebook, "Field_experiments_summary.ipynb", index, original)
    expected_groups.update({
        "treatment_pre": original_metrics(original, "treatment"),
        "control_pre": original_metrics(original, "control"),
    })
    expected_did = []
    for metric, expression_cell, test_cell in zip(METRICS, [37, 39, 41, 43, 45], [38, 40, 42, 44, 46]):
        run_cell(notebook, "Field_experiments_summary.ipynb", expression_cell, original)
        run_cell(notebook, "Field_experiments_summary.ipynb", test_cell, original)
        expected_did.append({"metric": metric, "did": original["post"].mean() - original["pre"].mean(),
                             "t_statistic": original["t"], "p_value": original["p"]})
    groups = {}
    for role, code in [("treatment", "e2e"), ("control", "jd")]:
        for period in ["pre", "post"]:
            key = f"{role}_{period}"
            groups[key] = module.measures(pd.read_csv(DATA / f"order_{code}_{period}_500.csv"))
    # The actual replay source constructs statistics here, with all file-writing
    # statements before and after this selected block excluded.
    body = main_body(module)
    start = next(i for i, node in enumerate(body) if simple_assignment(node, "post"))
    end = next(i for i, node in enumerate(body[start + 1:], start + 1)
               if simple_assignment(node, "post"))
    replay = dict(module.__dict__)
    replay["groups"] = groups
    execute_statements(body[start:end], replay, "reproduce_field:statistics_segment")
    post = pd.DataFrame(replay["post"])
    regression = pd.DataFrame(replay["regression"])
    did = pd.DataFrame(replay["did"])
    checks = []
    metric_columns = METRICS + ["turnover_raw"]
    for key in expected_groups:
        check_equal(f"field.{key}.all_row_metrics", groups[key][metric_columns].to_numpy(),
                    expected_groups[key][metric_columns].to_numpy(), checks)
    check_equal("field.post_t_and_p", post[["t_statistic", "p_value"]].to_numpy(),
                pd.DataFrame(expected_post)[["t_statistic", "p_value"]].to_numpy(), checks)
    check_equal("field.ols_coefficients_standard_errors_t_p",
                regression[["coefficient", "standard_error", "t_statistic", "p_value"]].to_numpy(),
                pd.DataFrame(expected_regression)[["coefficient", "standard_error", "t_statistic", "p_value"]].to_numpy(), checks)
    check_equal("field.did_point_estimates_t_and_p", did[["did", "t_statistic", "p_value"]].to_numpy(),
                pd.DataFrame(expected_did)[["did", "t_statistic", "p_value"]].to_numpy(), checks)
    # Cached author outputs are evidence to audit, not the oracle for an altered
    # public dataset. A mismatch here does not negate current-source equivalence.
    cache = module.validate_notebook(DATA, groups, post, did, regression)
    if len(cache) != 55:
        raise AssertionError(f"Expected 55 historical-cache comparisons, found {len(cache)}")
    return {
        "passed": True, "reviewed_cells_executed": FIELD_CELLS,
        "compatibility_change": {"description": "DataFrame.drop('y',1) -> drop('y',axis=1); no statistical change",
                                 "cells_changed": compatibility_changes},
        "row_metric_values_checked": 4 * 500 * len(metric_columns),
        "post_t_and_p_values_checked": 10,
        "ols_coefficient_standard_error_t_p_values_checked": 60,
        "did_point_t_and_p_values_checked": 15, "checks": checks,
        "did_from_current_original_source": expected_did,
        "historical_cache_audit_separate_from_source_verification": {
            "comparisons": len(cache), "matched": sum(c["match"] for c in cache),
            "mismatches": [c for c in cache if not c["match"]],
            "interpretation": "Saved notebook outputs are not all reproducible from the released CSV and current reviewed source. The cause is not established; source-equivalent replay is retained.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "source_equivalence.json")
    args = parser.parse_args()
    record = {"passed": False, "python": platform.python_version(),
              "scope": "Reviewed public notebook source equivalence; not private-model training reproduction",
              "notebook_sha256": NOTEBOOK_HASHES,
              "script_sha256": {name: sha256(ROOT / "scripts" / f"{name}.py")
                                 for name in ["reproduce_offline", "reproduce_field"]},
              "data_sha256": {path.name: sha256(path) for path in DATA.glob("*.csv")}}
    previous_directory = Path.cwd()
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
        offline_notebook = reviewed_notebook("E2E_Section4.ipynb")
        field_notebook = reviewed_notebook("Field_experiments_summary.ipynb")
        offline_module = import_replay("reproduce_offline")
        field_module = import_replay("reproduce_field")
        os.chdir(DATA)
        record["offline"] = verify_offline(offline_module, offline_notebook)
        record["field"] = verify_field(field_module, field_notebook)
        record["passed"] = True
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        os.chdir(previous_directory)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = record["field"]["historical_cache_audit_separate_from_source_verification"]
    print(json.dumps({"source_equivalence_passed": True,
                      "offline_group_method_curves": record["offline"]["group_method_curves_checked_exactly"],
                      "offline_mean_cost_values": 21,
                      "field_historical_cache_matches": f"{cache['matched']}/{cache['comparisons']}"}))


if __name__ == "__main__":
    main()
