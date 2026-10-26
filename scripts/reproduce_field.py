"""Recompute the public-subset field experiment and matching demonstration.

This preserves the published notebooks' definitions (including their edge cases)
and distinguishes recomputation from notebook logs and full-paper results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

METRICS = ["holding_cost", "stockout_cost", "total_cost", "turnover", "stockout_ratio"]
LABELS = ["Holding cost", "Stockout cost", "Total cost", "Turnover", "Stockout ratio"]


def measures(frame: pd.DataFrame, h: float = 1, b: float = 9) -> pd.DataFrame:
    """Exact notebook metric convention: exclude terminal inventory, divide by T."""
    records = []
    for row in frame.itertuples(index=False):
        inventory = np.asarray(ast.literal_eval(row.test_inv), dtype=float)
        # Two released control paths contain a terminal point only. The original
        # notebook charges zero periods for those records; they remain in the sample.
        if len(inventory) < 1 or not np.all(np.isfinite(inventory)):
            raise ValueError("Invalid inventory path")
        charged = inventory[:-1]
        holding = float(h * np.maximum(charged, 0).sum())
        shortage = float(b * np.maximum(-charged, 0).sum())
        raw_turnover = float(row.ave_inv / row.ave_demand)
        records.append({
            "sku_dc_pair_index": row.sku_dc_pair_index,
            "holding_cost": holding, "stockout_cost": shortage,
            "total_cost": holding + shortage,
            "turnover": max(0.0, raw_turnover), "turnover_raw": raw_turnover,
            "stockout_ratio": float(np.count_nonzero(charged < 0) / len(inventory)),
            "inventory_path_length": len(inventory),
            "demand_path_length": len(ast.literal_eval(row.test_demand)),
            "ave_demand": float(row.ave_demand),
            "vlt_num": int(str(row.vlt).split()[0]),
        })
    result = pd.DataFrame(records)
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Non-finite metrics")
    return result


def output_text(notebook, cell):
    return "\n".join(
        "".join(o.get("text", o.get("data", {}).get("text/plain", [])))
        for o in notebook["cells"][cell].get("outputs", [])
    )


def validate_notebook(data_dir, groups, post, did, regression):
    """Compare current calculations with embedded historical outputs, not paper."""
    notebook = json.loads((data_dir / "Field_experiments_summary.ipynb").read_text(encoding="utf8"))
    checks = []

    def check(name, computed, saved, atol=1e-9):
        if name.endswith('.p_value'):
            # Absolute 1e-9 would incorrectly call distinct tiny p-values equal.
            atol = 1e-300
        checks.append({"name": name, "computed": float(computed),
                       "notebook_saved": float(saved), "absolute_error": float(abs(computed-saved)),
                       "atol": atol, "match": bool(np.isclose(computed, saved, atol=atol, rtol=1e-8))})

    for key, cell in [("treatment_post", 8), ("control_post", 11),
                      ("treatment_pre", 33), ("control_pre", 36)]:
        saved = [float(line.rsplit(":", 1)[-1]) for line in output_text(notebook, cell).splitlines() if ":" in line]
        for metric, value in zip(METRICS, saved):
            check(f"{key}.{metric}", groups[key][metric].mean(), value)
    for kind, table, cells in [("post_ttest", post, [13,14,15,16,17]),
                               ("did_ttest_notebook", did, [38,40,42,44,46])]:
        for metric, cell in zip(METRICS, cells):
            saved = re.findall(r"[tp]\s*=\s*([-+\deE.]+)", output_text(notebook, cell))
            row = table.loc[table.metric.eq(metric)].iloc[0]
            for col, value in zip(["t_statistic", "p_value"], saved):
                check(f"{kind}.{metric}.{col}", row[col], float(value))
    for metric, cell in zip(METRICS, [23,24,25,26,27]):
        for term in ["e2e", "vlt", "d"]:
            match = re.search(rf"^\s*{term}\s+([-+\d.]+)", output_text(notebook,cell), flags=re.M)
            if match:
                value = regression.loc[(regression.metric == metric) & (regression.term == term), "coefficient"].iloc[0]
                check(f"ols.{metric}.{term}", value, float(match.group(1)), atol=5.1e-5)
    if len(checks) != 55:
        raise AssertionError(f'Expected 55 saved-output comparisons; parsed {len(checks)}')
    return checks


def matching_demo(data_dir, out_dir):
    treatment = pd.read_csv(data_dir / "order_e2e_post_500.csv")
    candidates = pd.read_csv(data_dir / "control_demand_vlt.csv")
    t = treatment[["ave_demand", "vlt_num"]].copy()
    c = candidates[["ave_demand", "vlt_num"]].copy()
    origin = pd.concat([t,c],ignore_index=True)
    y = np.r_[np.ones(len(t)), np.zeros(len(c))]
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=100, tol=1e-4)
    model.fit(origin, y)
    # The notebook matches covariate vectors, not the estimated propensity score.
    # Its treatment-indicator column contributes the same constant 1 to every
    # squared distance, so it does not alter covariate-nearest-neighbor ranking.
    tx = t.assign(e2e=1.).to_numpy()
    cx = c.assign(e2e=0.).to_numpy()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(cx)
    distances, indices = nbrs.kneighbors(tx)
    idx = indices[:,0]
    matched = c.iloc[idx].reset_index(drop=True)
    score_t = model.predict_proba(t)[:,1]
    score_c = model.predict_proba(c)[:,1]
    score_m = model.predict_proba(matched)[:,1]
    pairs = pd.DataFrame({
        "treatment_row": np.arange(len(t)), "candidate_row": idx,
        "candidate_disguised_index": candidates.iloc[idx]["Unnamed: 0"].to_numpy(),
        "notebook_distance_including_indicator": distances[:,0],
        "covariate_distance": np.linalg.norm(t.to_numpy()-matched.to_numpy(),axis=1),
        "treatment_propensity": score_t, "matched_propensity": score_m,
        "treatment_ave_demand": t.ave_demand,
        "matched_ave_demand": matched.ave_demand,
        "treatment_vlt_num": t.vlt_num, "matched_vlt_num": matched.vlt_num,
    })
    pairs.to_csv(out_dir / "matching_pairs.csv",index=False)
    balance = []
    for feature in ["ave_demand", "vlt_num"]:
        scale = np.sqrt((t[feature].var(ddof=1)+c[feature].var(ddof=1))/2)
        balance.append({"feature":feature,"treatment_mean":t[feature].mean(),
                        "candidate_mean":c[feature].mean(),"matched_mean":matched[feature].mean(),
                        "standardized_mean_difference_before":(t[feature].mean()-c[feature].mean())/scale,
                        "standardized_mean_difference_after_fixed_denominator":(t[feature].mean()-matched[feature].mean())/scale})
    pd.DataFrame(balance).to_csv(out_dir / "matching_balance.csv", index=False)
    fig, axes = plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    for ax, control_score, title in [(axes[0],score_c,"Before covariate matching"),
                                      (axes[1],score_m,"After covariate matching")]:
        bins = np.linspace(min(score_t.min(),control_score.min()),max(score_t.max(),control_score.max()),15)
        ax.hist(score_t,bins=bins,density=True,alpha=.55,label="Treatment (n=500)",color="#2867a4")
        ax.hist(control_score,bins=bins,density=True,alpha=.45,label=f"Control (n={len(control_score):,})",color="#df9655")
        ax.set(title=title,xlabel="Estimated propensity score",ylabel="Density")
        ax.legend(fontsize=8);ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out_dir / "matching_propensity.png",dpi=170)
    plt.close(fig)
    return {
        "treatment_n":len(t),"candidate_n":len(c),"selected_n":len(idx),
        "unique_selected_candidates":len(np.unique(idx)),"with_replacement":True,
        "maximum_covariate_distance":float(pairs.covariate_distance.max()),
        "mean_covariate_distance":float(pairs.covariate_distance.mean()),
        "logistic_intercept":model.intercept_.tolist(),"logistic_coefficients":model.coef_.tolist(),
        "logistic_iterations":model.n_iter_.tolist(),
        "matching_definition":"NearestNeighbors on unstandardized ave_demand, vlt_num and treatment indicator, as notebook; not on propensity score.",
        "identity_limitation":"Anonymized candidate indices cannot be joined reliably to the separately published 500 control trajectories; no identity equivalence claimed.",
        "plot_note":"Histogram version of the Figure 3 diagnostic, not a pixel reproduction of seaborn distplot/KDE.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir",type=Path,default=root/"author_package")
    parser.add_argument("--output-dir",type=Path,default=root/"results"/"field")
    args=parser.parse_args()
    data_dir=args.data_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    groups={}
    files={}
    for role, code in [("treatment","e2e"),("control","jd")]:
        for period in ["pre","post"]:
            key=f"{role}_{period}"; filename=f"order_{code}_{period}_500.csv"
            original=pd.read_csv(data_dir/filename)
            if len(original)!=500 or not original.sku_dc_pair_index.is_unique:
                raise ValueError(f"Unexpected row count or duplicate identifiers: {filename}")
            groups[key]=measures(original)
            groups[key].to_csv(out/f"metrics_{key}.csv",index=False)
            files[filename]={"rows":len(original),"sha256":hashlib.sha256((data_dir/filename).read_bytes()).hexdigest(),
                             "missing_cells":int(original.isna().sum().sum())}
    summaries=pd.DataFrame([{ "group":key,"n":len(df),**df[METRICS].mean().to_dict()} for key,df in groups.items()])
    summaries.to_csv(out/"group_summary.csv",index=False)
    post=[];regression=[];did=[];did_consistent=[]
    a=groups["treatment_post"];b=groups["control_post"]
    X=pd.DataFrame({"e2e":np.r_[np.ones(len(a)),np.zeros(len(b))],
                    "vlt":np.r_[a.vlt_num,b.vlt_num],"d":np.r_[a.ave_demand,b.ave_demand]})
    for metric in METRICS:
        test=stats.ttest_ind(a[metric],b[metric],equal_var=True)
        post.append({"metric":metric,"treatment_mean":a[metric].mean(),"control_mean":b[metric].mean(),
                     "difference":a[metric].mean()-b[metric].mean(),
                     "relative_reduction_pct":100*(1-a[metric].mean()/b[metric].mean()),
                     "t_statistic":test.statistic,"p_value":test.pvalue})
        y=np.r_[a[metric],b[metric]]
        fit=sm.OLS(y,X).fit() # No intercept: exact notebook specification.
        for term in X.columns:
            regression.append({"metric":metric,"term":term,"coefficient":fit.params[term],
                               "standard_error":fit.bse[term],"t_statistic":fit.tvalues[term],
                               "p_value":fit.pvalues[term],"ci_low":fit.conf_int().loc[term,0],
                               "ci_high":fit.conf_int().loc[term,1],"nobs":fit.nobs,"intercept":False})
        # Author notebook uses raw post-control turnover only in DID. Preserve
        # that execution detail, and export a consistently clipped sensitivity.
        for legacy, rows in [(True,did),(False,did_consistent)]:
            control_post=b["turnover_raw" if legacy and metric=="turnover" else metric]
            treatment_pre=groups["treatment_pre"][metric]
            control_pre=groups["control_pre"][metric]
            post_difference=a[metric].to_numpy()-control_post.to_numpy()
            pre_difference=treatment_pre.to_numpy()-control_pre.to_numpy()
            test=stats.ttest_ind(post_difference,pre_difference,equal_var=True)
            rows.append({"metric":metric,"treatment_pre":treatment_pre.mean(),"treatment_post":a[metric].mean(),
                         "control_pre":control_pre.mean(),"control_post":control_post.mean(),
                         "treatment_change":a[metric].mean()-treatment_pre.mean(),
                         "control_change":control_post.mean()-control_pre.mean(),
                         "did":post_difference.mean()-pre_difference.mean(),
                         "t_statistic":test.statistic,"p_value":test.pvalue})
    post=pd.DataFrame(post);regression=pd.DataFrame(regression);did=pd.DataFrame(did)
    post.to_csv(out/"table2_public_subset.csv",index=False)
    regression.to_csv(out/"table3_public_subset_all_coefficients.csv",index=False)
    did.to_csv(out/"table4_public_subset_notebook_convention.csv",index=False)
    pd.DataFrame(did_consistent).to_csv(out/"table4_public_subset_consistent_turnover.csv",index=False)
    checks=validate_notebook(data_dir,groups,post,did,regression)
    known_cache_mismatches={f'did_ttest_notebook.{m}.{c}' for m in METRICS
                            for c in ['t_statistic','p_value']}
    actual_cache_mismatches={c['name'] for c in checks if not c['match']}
    unexpected_cache_mismatches=actual_cache_mismatches-known_cache_mismatches
    pd.DataFrame(checks).to_csv(out/"notebook_output_verification.csv",index=False)
    matching=matching_demo(data_dir,out)
    # Full-paper values are references only; supplied public data are subsamples.
    paper=pd.DataFrame({"metric":METRICS,
                        "paper_treatment_post":[598.20,496.34,1094.54,18.59,.17],
                        "paper_control_post":[809.54,1027.93,1837.47,20.39,.26],
                        "paper_e2e_ols_coefficient":[-211.33,-531.53,-742.86,-1.16,-.09],
                        "paper_did":[-112.79,-524.45,-637.24,-1.81,-.11]})
    paper=paper.merge(post[["metric","treatment_mean","control_mean","relative_reduction_pct"]],on="metric")
    paper["public_subset_e2e_ols_coefficient"]=regression[regression.term.eq("e2e")].coefficient.to_numpy()
    paper["public_subset_did_notebook"]=did.did.to_numpy()
    paper.to_csv(out/"paper_vs_public_subset.csv",index=False)
    # Supplementary specifications expose the paper/code differences. They are
    # not substituted into the notebook fidelity checks or presented as logs.
    regression_intercept=[]
    for metric in METRICS:
        fit=sm.OLS(np.r_[a[metric],b[metric]],sm.add_constant(X)).fit()
        for term in fit.params.index:
            regression_intercept.append({"metric":metric,"term":term,
                "coefficient":fit.params[term],"standard_error":fit.bse[term],
                "t_statistic":fit.tvalues[term],"p_value":fit.pvalues[term],
                "nobs":fit.nobs,"h":1,"b":9,"intercept":True})
    pd.DataFrame(regression_intercept).to_csv(out/"sensitivity_ols_with_intercept_h1_b9.csv",index=False)
    paper_cost_groups={}
    for role,code in [("treatment","e2e"),("control","jd")]:
        for period in ["pre","post"]:
            key=f"{role}_{period}"
            original=pd.read_csv(data_dir/f"order_{code}_{period}_500.csv")
            paper_cost_groups[key]=measures(original,h=12,b=88)
    pd.DataFrame([{ "group":key,"n":len(df),"h":12,"b":88,
                  **df[METRICS].mean().to_dict()} for key,df in paper_cost_groups.items()]).to_csv(
                      out/"sensitivity_group_summary_h12_b88.csv",index=False)
    paper_cost_post=[]
    for metric in METRICS:
        ta=paper_cost_groups["treatment_post"][metric]
        cb=paper_cost_groups["control_post"][metric]
        test=stats.ttest_ind(ta,cb,equal_var=True)
        paper_cost_post.append({"metric":metric,"h":12,"b":88,
            "treatment_mean":ta.mean(),"control_mean":cb.mean(),
            "relative_reduction_pct":100*(1-ta.mean()/cb.mean()),
            "t_statistic":test.statistic,"p_value":test.pvalue})
    pd.DataFrame(paper_cost_post).to_csv(out/"sensitivity_post_comparison_h12_b88.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
    for ax,metric,label in zip(axes,METRICS[:3],LABELS[:3]):
        row=post.loc[post.metric.eq(metric)].iloc[0]
        ax.bar(["E2E","JD control"],[row.treatment_mean,row.control_mean],color=["#2867a4","#df9655"])
        ax.set(title=label,ylabel="Mean accumulated cost (h=1, b=9)")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Field experiment: released subset, 500 pairs per group")
    fig.savefig(out/"post_experiment_costs.png",dpi=170);plt.close(fig)
    record={"scope":"Released 500-pair-per-group subset; not full paper dataset", "python":platform.python_version(),
            "package_versions":{n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","statsmodels","scikit-learn","matplotlib"]},
            "inputs":files,"notebook_checks":len(checks),"notebook_checks_passed":sum(c["match"] for c in checks),
            "supplementary_specifications":{"cost_sensitivity":{"h":12,"b":88,"scope":"Same public subset, notebook inventory-path convention; paper field cost rates substituted"}, "OLS_with_intercept":{"h":1,"b":9,"scope":"Same public subset and covariates; intercept added to match paper formula, not notebook execution"}},
            "all_notebook_checks_passed":all(c["match"] for c in checks),"matching":matching,
            "historical_cache_mismatches":sorted(actual_cache_mismatches),
            "unexpected_historical_cache_mismatches":sorted(unexpected_cache_mismatches),
            "cache_audit_note":"Historical DiD t/p outputs differ from executing the current public CSV and reviewed original notebook. Independent source-equivalence validation is separate; no algorithm or tolerance is changed to force agreement.",
            "definitions":{"h":1,"b":9,"cost_inventory_indices":"0..T-2 inclusive", "stockout_denominator":"T, including terminal inventory point", "turnover":"max(0, ave_inv/ave_demand) except notebook DID uses raw post-control turnover", "OLS":"e2e, vlt, d; no intercept; ordinary nonrobust standard errors", "t_test":"scipy.stats.ttest_ind(equal_var=True), two-sided", "DID_test":"independent t-test between row-wise treatment-minus-control differences in post and pre; reproduces notebook, not a clustered panel DID estimator"}}
    (out/"verification.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
    total=post.loc[post.metric.eq("total_cost")].iloc[0]
    turnover_coef=regression.loc[(regression.metric.eq("turnover")) & (regression.term.eq("e2e"))].iloc[0]
    report=f"""# 公开子样本实证复现

当前脚本重新读取四个原始 CSV（处理组/对照组、实验前/后，各 500 行），独立重算统计量。结果并非论文完整样本结果，论文原表仅在 paper_vs_public_subset.csv 中作为参照。

## 主要结果

- 实验后平均总成本：E2E={total.treatment_mean:.3f}，对照={total.control_mean:.3f}，公开子样本降幅={total.relative_reduction_pct:.2f}%。独立样本双侧 t 检验 p={total.p_value:.4g}。
- 与作者 Field_experiments_summary.ipynb 历史保存输出比较：{record['notebook_checks_passed']}/{len(checks)} 项匹配；其余为已确认的 DiD t/p 缓存差异，详见 notebook_output_verification.csv。直接执行原 notebook 当前源码与公开 CSV 得到与新脚本一致的结果，独立核验另存 source_equivalence.json；差异原因公开资料无法确定。
- 公开子样本 turnover 的 E2E 无截距 OLS 系数为 {turnover_coef.coefficient:.6f}（p={turnover_coef.p_value:.4f}），并未重现论文完整样本中该回归系数显著为负的结论。
- 候选对照 {matching['candidate_n']:,} 行，按作者代码为 500 个处理样本进行可重复使用的最近邻匹配，选中 {matching['unique_selected_candidates']} 个不同候选。图表展示匹配前后 propensity score 分布。

## 精确遵循作者 notebook 的口径

成本使用 h=1、b=9，只累计 test_inv[:-1]。缺货比例分子为上述区间库存小于零的点数，分母仍为完整库存路径 T。各条路径长度不统一，未自行改为日均成本。turnover 为 ave_inv/ave_demand，并在汇总和实验后回归中截断为非负。

回归包含 e2e、vlt、d 三列，无截距，采用普通标准误。DID 的点估计为处理组前后变化减去对照组前后变化，显著性则忠实重现原 notebook 对前后两个逐行组间差序列进行独立样本 t 检验的方式；未声称这是带个体固定效应或聚类标准误的面板 DID。

原 notebook DID 唯独对照组实验后 turnover 使用未截断值（post_exp_jd 存入 jd_turnover）。table4_public_subset_notebook_convention.csv 保留该行为，table4_public_subset_consistent_turnover.csv 则给出统一非负截断的敏感性结果。其他四项不变。

## 论文口径敏感性（独立附表）

论文现场成本单价 h=12、b=88，与 notebook 的 h=1、b=9 不同；这不是同一倍数缩放。sensitivity_group_summary_h12_b88.csv 与 sensitivity_post_comparison_h12_b88.csv 使用同一公开子样本及库存区间定义替换成本单价，供比较。论文回归公式含截距，而 notebook 未加截距；sensitivity_ols_with_intercept_h1_b9.csv 为 notebook 成本单价下加截距的对照回归。它们均为本次额外计算，未替代主结果，也未当作作者保存日志。

## 匹配演示限制

作者先用 LogisticRegression 估算 propensity score，但实际 NearestNeighbors 输入是未标准化的 ave_demand、vlt_num、e2e 三列，而不是 propensity score。e2e 指示列对所有候选距离贡献相同常数，不改变排名。未加入 caliper、未禁止同一候选重复使用。由于匿名索引不能可靠跨文件映射，重新匹配结果未被冒充为已发布 order_jd_post_500.csv 的身份重建；候选数据也没有完整库存路径，无法重新生成这些候选的实证轨迹。matching_propensity.png 采用直方图展示分布，不是原 seaborn KDE 图的像素级复刻。

## 运行

从项目根目录运行 `.venv/Scripts/python.exe scripts/reproduce_field.py`。所有作者文件保持原样；结果写入 results/field。verification.json 记录依赖版本、输入哈希、定义和核验状态。
"""
    (out/"field_report.md").write_text(report,encoding="utf8")
    print(json.dumps({"checks":len(checks),"passed":record["notebook_checks_passed"],"total_cost_reduction_pct":total.relative_reduction_pct,"matching_unique":matching["unique_selected_candidates"]}))
    if unexpected_cache_mismatches:
        raise AssertionError(f"Unexpected saved-output differences: {sorted(unexpected_cache_mismatches)}")


if __name__=="__main__":
    main()
