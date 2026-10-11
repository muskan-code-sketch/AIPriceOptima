# AI Price Optima

Amazon sales analytics and pricing scenarios, built with Streamlit, XGBoost and LightGBM.

**Run locally**

```bash
pip install -r requirements.txt
streamlit run streamlit_app_FIXED.py
```

## What the app does

| Tab | Question it answers |
|---|---|
| Overview | How is the business doing? KPIs, trend, seasonality, top products and cities |
| Discount Insights | What do discounts cost, and do they sell more units? Includes a discount-cap calculator with a break-even point |
| Price Simulator | What happens to revenue, units and profit if I change the price? |
| Recommendations | Which products should I change, and how fragile is that plan? |
| Revenue Predictor | What will this order bring in? Single order and batch CSV |
| Model Health | How accurate are the models on unseen orders? |

Sidebar filters (period, products, cities) apply to every tab. Business assumptions
(price sensitivity, product cost, max price move) drive the Simulator and Recommendations.

## Important: what the data can and cannot tell you

* `TotalAmount` is almost `UnitPrice x Quantity x (1 - Discount)` plus about 8% fees. The models
  therefore score ~99% R² (99.4% on a held-out 20% split) - they are accurate **revenue estimators**.
* In this dataset, units sold show **no relationship with price or discount** (measured price
  sensitivity ~0, discount volume lift +0.2%, not significant). A demand curve cannot be learned
  from it, so the Simulator and Recommendations use an explicit, adjustable price-sensitivity
  assumption. Results are scenarios, not forecasts - validate with a pilot.
* There is no cost column, so product cost is an assumption (default 60% of list price).

## Files

* `streamlit_app_FIXED.py` - app entry point (layout, sidebar, tabs)
* `aipo/` - data loading, pricing engine, model helpers and one module per tab
* `xgboost_revenue_model.json` - version-proof export of the XGBoost model (the `.pkl` is kept as fallback)
* `Milestone_5_FULLY_CORRECTED.ipynb` - model training notebook
