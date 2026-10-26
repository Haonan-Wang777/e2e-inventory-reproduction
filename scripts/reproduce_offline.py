"""Replay the released Section 4 notebook; do not retrain its stored predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['OPT', 'E2E_RNN', 'Bm1', 'Bm2', 'Normal', 'Gamma', 'gbm']
# Values saved in author_package/E2E_Section4.ipynb, rounded to six decimals.
REFERENCE = np.array([
    [2572.964327, 3036.758343, 3842.305332, 3837.828539, 3961.691600, 4099.949751, 3424.869965],
    [1363.218642, 1883.591484, 1709.364020, 1541.630610, 2317.097814, 1947.838128, 1424.929804],
    [1209.745685, 1153.166858, 2132.941312, 2296.197929, 1644.593786, 2152.111623, 1999.940161],
])


def inventory_path(x, name, diagnostics):
    """Faithful translation of the author's get_agginv, including its fallbacks.

    The author's routine subtracts initial inventory from stored target predictions,
    permits negative replenishments, and resets negative inventory at arrivals.
    These choices are retained for comparability, not endorsed as a general simulator.
    """
    inv1, inv2 = [x['initial_stock'][0]], []
    rd = len(x['OPT_pred'])
    for r in range(rd):
        len_day = len(x['demand_hist'][r]) - (1 if r < rd - 1 else 0)
        for t in range(len_day):
            if t == 0:
                if r == 0:
                    replen = int(round(x[name + '_pred'][r] - inv1[0]))
                else:
                    try:
                        replen = int(round(x[name + '_pred'][r] - inv1[-int(round(x['vlt'][r]))-1]))
                    except IndexError:
                        replen = int(round(x[name + '_pred'][r] - inv1[1]))
                        diagnostics['fallback_inventory_index'] += 1
                diagnostics['orders'] += 1
                diagnostics['negative_replenishments'] += int(replen < 0)
            if t < int(round(x['vlt'][r])):
                if r == 0:
                    inv1.append(inv1[-1] - x['demand_hist'][r][t])
            elif t == int(round(x['vlt'][r])):
                inv_ = max(inv1[-1], 0) + replen - x['demand_hist'][r][t]
                inv1.append(inv_)
                inv2.append(inv_)
            else:
                inv_ = inv1[-1] - x['demand_hist'][r][t]
                inv1.append(inv_)
                inv2.append(inv_)
    return inv1[1:], inv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'offline')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'author_package' / 'df_test.csv'
    df = pd.read_csv(source)
    df['sf_rnn'] = df['sf_rnn'].map(lambda s: pd.to_numeric(s.strip('[]').split(', ')))
    df['demand_hist'] = df['demand_hist'].map(lambda s: pd.to_numeric(s.strip('[]').split()))
    assert len(df) == 3000 and df['SKU'].nunique() == 2607
    assert all(len(x) > 0 for x in df['demand_hist'])
    forecast = df['sf_rnn'].map(np.mean)
    df['Bm1_pred'] = forecast * (df['review_period'] + df['vlt']).astype(int)

    def bm2(row):
        rl = row['review_period'] + row['vlt']
        days = int(rl) if rl <= 9 else int(rl) - rl // 10
        return row['Bm2_pred'] * days

    df['Bm2_pred'] = forecast
    df['Bm2_pred'] = df.apply(bm2, axis=1)
    # Preserve the notebook's Z90=1.64 and variance expression in the main replay.
    df['Normal_pred'] = df.apply(lambda x: int(
        x['demand_mean'] * (x['review_period'] + x['vendor_vlt_mean'])
        + 1.64 * np.sqrt((x['review_period'] + x['vendor_vlt_mean']) * x['demand_std']**2
                         + x['demand_std']**2 * x['vendor_vlt_std'])), axis=1)

    def gamma_base(x):
        theta = x['demand_std']**2 / (x['demand_mean'] + 1e-4)
        k = x['demand_mean'] / (theta + 1e-4)
        result = gamma.ppf(.9, a=int(x['review_period'] + x['vendor_vlt_mean']) * k, scale=theta)
        return 0 if np.isnan(result) else int(result)

    df['Gamma_pred'] = df.apply(gamma_base, axis=1)
    # Separate sensitivity calculation using equation (7), while keeping simulation fixed.
    df['Normal_equation7_pred'] = df.apply(lambda x: int(
        x['demand_mean'] * (x['review_period'] + x['vendor_vlt_mean'])
        + norm.ppf(.9) * np.sqrt((x['review_period'] + x['vendor_vlt_mean']) * x['demand_std']**2
                               + x['demand_mean']**2 * x['vendor_vlt_std']**2)), axis=1)
    grouped = df.groupby('SKU', sort=True).agg(lambda x: x.tolist())
    costs, diagnostics, paths = {}, {}, {}
    for method in METHODS + ['Normal_equation7']:
        diag = dict(orders=0, negative_replenishments=0, fallback_inventory_index=0)
        curves = [inventory_path(row, method, diag)[1] for _, row in grouped.iterrows()]
        holding = np.array([sum(v for v in a if v > 0) for a in curves])
        shortage = np.array([-9 * sum(v for v in a if v < 0) for a in curves])
        costs[method] = np.array([holding + shortage, holding, shortage]).T
        diagnostics[method] = diag
        paths[method] = curves
    table = pd.DataFrame({m: costs[m].mean(axis=0) for m in METHODS},
                         index=['Total cost', 'Holding cost', 'Stockout cost'])
    table.to_csv(args.out / 'cost_comparison.csv', encoding='utf-8-sig')
    errors = table.to_numpy() - REFERENCE
    max_error = float(np.max(np.abs(errors)))
    # Six-decimal saved reference only supports a 0.5e-6 rounding tolerance.
    np.testing.assert_allclose(table.to_numpy(), REFERENCE, rtol=0, atol=5.1e-7)
    pd.DataFrame(errors, index=table.index, columns=table.columns).to_csv(
        args.out / 'notebook_reference_errors.csv', encoding='utf-8-sig')
    records = []
    for m in costs:
        for sku, a in zip(grouped.index, costs[m]):
            records.append(dict(SKU=sku, method=m, total_cost=a[0], holding_cost=a[1], stockout_cost=a[2]))
    pd.DataFrame(records).to_csv(args.out / 'per_sku_costs.csv', index=False, encoding='utf-8-sig')
    sensitivity = pd.DataFrame({m: costs[m].mean(axis=0) for m in ['Normal', 'Normal_equation7']},
                               index=table.index)
    sensitivity.to_csv(args.out / 'normal_formula_sensitivity.csv', encoding='utf-8-sig')
    reductions = {m: float(1 - table.loc['Total cost', 'E2E_RNN'] / table.loc['Total cost', m])
                  for m in METHODS if m not in ['E2E_RNN', 'OPT']}
    metadata = dict(
        data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=len(df),
        unique_sku_dc=int(df['SKU'].nunique()), notebook_check_passed=True,
        max_absolute_error_to_six_decimal_notebook=max_error,
        cost_weights=dict(h=1, b=9), e2e_total_cost_reduction=reductions,
        diagnostics=diagnostics,
        notes=[
            'E2E_RNN, GBM, OPT and demand forecasts are provided predictions, not fitted here.',
            'The released CSV has 3000 rows but 2607 unique SKU identifiers.',
            'Rows within each SKU retain CSV order; no timestamp is available to validate chronology.',
            'Costs use inv2 as in the notebook: pre-first-arrival inventory points are excluded.',
            'Negative orders and negative-stock reset are preserved from the notebook.',
            'Normal equation7 is a formula sensitivity only, not a new full simulation model.',
            'CSV vlt semantics are ambiguous: README says prediction, original log uses vlt_actual.',
        ])
    (args.out / 'verification.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    xs = np.arange(len(METHODS))
    hold, short = table.loc['Holding cost'].values, table.loc['Stockout cost'].values
    ax.bar(xs, hold, color='#447AA3', label='Holding cost')
    ax.bar(xs, short, bottom=hold, color='#D6A064', label='Stockout cost')
    for i, total in enumerate(hold + short):
        ax.text(i, total + 60, f'{total:,.2f}', ha='center', fontsize=10)
    ax.set_xticks(xs, ['OPT', 'E2E', 'BM1', 'BM2', 'Normal', 'Gamma', 'GBM'])
    ax.set_ylim(0, max(hold + short) * 1.18)
    ax.set_ylabel('Mean cost per SKU group (h=1, b=9)')
    ax.set_title('Offline replay on released data: 3,000 rows / 2,607 SKU groups', loc='left', pad=15)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(args.out / 'offline_costs.png', dpi=180)
    fig.savefig(args.out / 'offline_costs.svg')
    plt.close(fig)
    print(table.to_string(float_format=lambda v: f'{v:.6f}'))
    print(f'PASS: all 21 notebook reference values match; maximum absolute error {max_error:.3g}')


if __name__ == '__main__':
    main()
