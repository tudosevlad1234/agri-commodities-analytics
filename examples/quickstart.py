from agri.pipeline import analyse_contract
from agri.synthetic import generate

bundle = generate(seed=7)
report = analyse_contract(bundle, "wheat")
fv, crop, regime, score = report.fair_value, report.crop, report.regime, report.score

print(f"price {fv.price_now:.1f}  fair value {fv.fv_now:.1f}  gap {fv.gap_pct:+.1f}%")
print(f"crop nowcast: world supply {crop.supply_shock_pct:+.1f}%  price impact {crop.price_impact_pct:+.1f}%")
print(f"regime {regime.state} ({regime.certainty:.0%})")
print(f"score {score.score:+.0f}  P(up) {score.p_up:.0%}  trust {score.trust:.0f}  {score.verdict}")
