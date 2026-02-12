import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import heapq
from heapq import heappush, heappop
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict

# параметры

CONSTRAINTS = {
    "TOTAL_BUDGET": 100_000,
    "DRR_TARGET": 0.15,
    "STEP": 500,
    "MAX_DAILY_CHANGE": 10_000,
}

BUSINESS_PARAMS = {
    "RISK_AVERSION": 0.05,
    "STABILITY_PREFERENCE": 0.7,
    "FUTURE_DISCOUNT": 0.9,
}

DAYS = 5
np.random.seed(42)

# выгрузка в датафрейм

df = pd.read_csv("./content/test.csv")
df.columns = df.columns.str.lower()

def pick(cols):
    for c in cols:
        if c in df.columns:
            return c
    raise KeyError(cols)

CID = pick(["campaign_id", "campaign", "id"])
SPEND = pick(["money_spent", "cost"])
REV = pick(["orders_money", "revenue"])

df["profit"] = df[REV] - df[SPEND]

# оценка
def revenue_fn(x, a, b):
    return a * np.log1p(b * x)

rows = []

for cid, g in df.groupby(CID):
    x, y = g[SPEND].values, g[REV].values
    if len(np.unique(x)) < 3:
        continue
    try:
        (a, b), _ = curve_fit(revenue_fn, x, y, bounds=(0, np.inf))
    except Exception:
        continue

    profits = g["profit"].values
    q = np.quantile(profits, 0.1)
    cvar = profits[profits <= q].mean() if np.any(profits <= q) else 0

    # оценка стабильности по волатильности
    stability = 1.0
    if len(profits) > 3 and profits.mean() > 0:
        cv = profits.std() / profits.mean()  # коэффициент вариации
        if profits.mean() <= 0:  # убыточные кампании
            stability = 0.1  # минимальная стабильность
        else:
            cv = profits.std() / profits.mean()
            stability = max(0.1, 1 - min(cv, 2.0))  # стабильность от 0.1 до 1

    # аномалии
    is_anomaly = False
    if len(profits) > 5:
        # проверяем на выбросы
        q25, q75 = np.percentile(profits, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 3 * iqr
        if profits.min() < lower_bound:
            is_anomaly = True

    rows.append({
        "campaign_id": cid,
        "alpha": a,
        "beta": b,
        "cvar": cvar,
        "stability": stability,
        "is_anomaly": is_anomaly,
        "min_spend": g[SPEND].quantile(0.2),
        "max_spend": g[SPEND].quantile(0.9),
        "data_points": len(g)
    })

campaigns = pd.DataFrame(rows)

# классификация и маржа

beta_med = campaigns.beta.median()
alpha_med = campaigns.alpha.median()


campaigns["type"] = "tractor"

# балласт: низкий alpha
alpha_q25 = campaigns.alpha.quantile(0.25)
cvar_q25 = campaigns.cvar.quantile(0.25)
ballast_mask = (campaigns.alpha < alpha_q25) & (campaigns.cvar < cvar_q25)
campaigns.loc[ballast_mask, "type"] = "ballast"

# снайпер: высокий beta
sniper_mask = (~ballast_mask) & (campaigns.beta > beta_med)
campaigns.loc[sniper_mask, "type"] = "sniper"

# назначение маржи
campaigns["margin"] = np.where(
    campaigns.type == "ballast", 0.3,      # низкая маржа
    np.where(campaigns.type == "sniper", 1.2, 1.0)  # высокая для снайперов
)

# маржа для аномальных кампаний
campaigns.loc[campaigns.is_anomaly, "margin"] *= 0.5

# min/max spend по типам
campaigns["min_spend"] = np.where(
    campaigns.type == "ballast", 0,
    np.where(campaigns.type == "sniper",
             CONSTRAINTS["STEP"] * 3,
             CONSTRAINTS["STEP"] * 5)
)

campaigns["max_spend"] = np.where(
    campaigns.type == "ballast", 0,
    np.where(campaigns.type == "sniper",
             CONSTRAINTS["TOTAL_BUDGET"] * 0.25,
             CONSTRAINTS["TOTAL_BUDGET"] * 0.30)
)

# многодневная оптимизация
def multi_day_score(row, budget_series):
    """Рассчитывает дисконтированную прибыль на несколько дней"""
    total = 0
    gamma = BUSINESS_PARAMS["FUTURE_DISCOUNT"]

    for t, x in enumerate(budget_series):
        # прибыль в день t
        daily_profit = row.margin * revenue_fn(x, row.alpha, row.beta) - x
        discount_factor = gamma ** t
        total += daily_profit * discount_factor

    return total

# стимуляция за несколько дней

def get_dynamic_revenue(row, budget, day):
    base_rev = revenue_fn(budget, row.alpha, row.beta)
    seed_value = int(row.campaign_id) + day * 1000
    np.random.seed(seed_value)
    
    # динамика по типам
    if row['type'] == "sniper":
        # снайперы теряют эффективность
        multiplier = max(0.8, 1.0 - 0.03 * (day-1))
    elif row['type'] == "tractor":
        # тягачи стабильны
        multiplier = 1.0 + np.random.uniform(-0.05, 0.05)
    else:
        # балласты ухудшаются
        multiplier = max(0.6, 1.0 - 0.07 * (day-1))
    
    # шум
    if row['is_anomaly']:
        noise = np.random.uniform(0.7, 1.3)
    else:
        noise = np.random.uniform(0.9, 1.1)
    
    return base_rev * multiplier * noise

def optimize_budget_dynamic(campaigns_df, prev_budgets, constraints, day):
    
    current_budgets = prev_budgets.copy()
    total_spent = sum(current_budgets.values())
    
    if day == 1 and not current_budgets:
        # начинаем с топ-кампаний
        campaigns_df = campaigns_df.copy()
        campaigns_df['initial_roi'] = campaigns_df.apply(
            lambda row: (row.margin * revenue_fn(constraints["STEP"]*5, row.alpha, row.beta) - constraints["STEP"]*5) / (constraints["STEP"]*5),
            axis=1
        )
        top_campaigns = campaigns_df.nlargest(20, 'initial_roi')
        
        for _, row in top_campaigns.iterrows():
            if total_spent < constraints["TOTAL_BUDGET"]:
                budget = min(constraints["STEP"] * 5, constraints["TOTAL_BUDGET"] - total_spent)
                if budget >= row.min_spend:
                    current_budgets[row.campaign_id] = budget
                    total_spent += budget
    else:
        # оцениваем ROI активных кампании
        roi_data = []
        for cid, x in list(current_budgets.items()):
            if x > 0:
                row = campaigns_df[campaigns_df.campaign_id == cid].iloc[0]
                pred_rev = get_dynamic_revenue(row, x, day)
                roi = (row.margin * pred_rev - x) / x if x > 0 else 0
                roi_data.append((roi, cid, x, row))
        
        # сортируем по ROI
        roi_data.sort(reverse=True, key=lambda x: x[0])
        
        # отключаем 10% худших
        num_to_disable = max(1, len(roi_data) // 10)
        for i in range(-1, -num_to_disable-1, -1):
            if abs(i) <= len(roi_data):
                roi, cid, x, row = roi_data[i]
                if roi < 0.05:
                    current_budgets[cid] = 0
                    total_spent -= x
        
        # включаем новые кампании
        active_ids = set(current_budgets.keys())
        all_ids = set(campaigns_df.campaign_id)
        inactive_ids = all_ids - active_ids
        
        # потенциальные кандидатов
        candidates = []
        for cid in inactive_ids:
            row = campaigns_df[campaigns_df.campaign_id == cid].iloc[0]
            if row.type == "ballast":
                continue
                
            test_budget = max(constraints["STEP"], row.min_spend)
            if test_budget <= constraints["TOTAL_BUDGET"] - total_spent:
                pred_rev = get_dynamic_revenue(row, test_budget, day)
                roi = (row.margin * pred_rev - test_budget) / test_budget if test_budget > 0 else 0
                candidates.append((roi, cid, test_budget, row))
        
        # берём лучших
        candidates.sort(reverse=True, key=lambda x: x[0])
        for roi, cid, test_budget, row in candidates[:5]:
            if roi > 0.1 and test_budget <= constraints["TOTAL_BUDGET"] - total_spent:
                current_budgets[cid] = test_budget
                total_spent += test_budget
    
    # проверяем ограничения
    final_budgets = {}
    final_total = 0
    
    for cid, x in current_budgets.items():
        if x > 0:
            row = campaigns_df[campaigns_df.campaign_id == cid].iloc[0]
            # округляем до шага
            x = round(x / constraints["STEP"]) * constraints["STEP"]
            x = max(0, min(row.max_spend, x))
            
            if x > 0 and final_total + x <= constraints["TOTAL_BUDGET"]:
                final_budgets[cid] = x
                final_total += x
    
    return final_budgets

budgets = {}
stats = []
history = []
campaign_scores = []

for day in range(1, DAYS + 1):
    print(f"Day {day}: ", end="")
    
    # оптимизируем
    budgets = optimize_budget_dynamic(campaigns, budgets, CONSTRAINTS, day)
    
    # считаем результаты
    spend = sum(budgets.values())
    revenue = 0
    daily_profits = []
    
    for cid, x in budgets.items():
        if x > 0:
            row = campaigns[campaigns.campaign_id == cid].iloc[0]
            rev = get_dynamic_revenue(row, x, day)
            revenue += rev
            profit = row.margin * rev - x
            daily_profits.append(profit)
            
            discount = BUSINESS_PARAMS["FUTURE_DISCOUNT"] ** (day - 1)
            campaign_scores.append({
                "day": day,
                "campaign_id": cid,
                "budget": x,
                "profit": profit,
                "discounted_score": profit * discount,
                "type": row.type
            })
    
    total_profit = revenue - spend
    
    stats.append({
        "day": day,
        "spend": spend,
        "revenue": revenue,
        "profit": total_profit,
        "drr": spend / revenue if revenue > 0 else 0,
        "active_campaigns": sum(1 for x in budgets.values() if x > 0),
        "avg_stability": np.mean([campaigns[campaigns.campaign_id == cid].iloc[0].stability
                                  for cid in budgets if budgets[cid] > 0]) if budgets else 0
    })
    
    # История
    for cid, x in budgets.items():
        if x > 0:
            history.append({
                "day": day,
                "campaign_id": cid,
                "budget": x,
                "type": campaigns[campaigns.campaign_id == cid].iloc[0].type
            })
    
    print(f"Active: {stats[-1]['active_campaigns']}, Profit: {total_profit:.0f}")

metrics = pd.DataFrame(stats)
hist = pd.DataFrame(history)
scores_df = pd.DataFrame(campaign_scores)

# Отчёт (результаты)

print("Параметры:")
for param, value in BUSINESS_PARAMS.items():
    print(f"  {param}: {value}")


print("\n Статистика компаний по их типу")
type_stats = campaigns.groupby("type").agg({
    "campaign_id": "count",
    "margin": "mean",
    "stability": "mean",
    "cvar": "mean"
}).round(3)
print(type_stats)

print("\n Вычисление аномалий")
anomaly_stats = campaigns.groupby("is_anomaly").agg({
    "campaign_id": "count",
    "margin": "mean"
})
print(anomaly_stats)

print("\n Динамика за несколько дней")
print(metrics.round(3))

# итоговый score
total_score = scores_df.groupby("campaign_id")["discounted_score"].sum().sum()

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Прибыль & DRR Динамика
axes[0, 0].plot(metrics.day, metrics.profit, marker="o", label="Прибыль")
axes[0, 0].plot(metrics.day, metrics.drr * 10000, "--", label="DRR")
axes[0, 0].set_title("Прибыль & DRR Динамика")
axes[0, 0].set_xlabel("День")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Бюджет стабльных компаний
top_campaigns = hist.groupby("campaign_id")["budget"].std().nsmallest(5).index
for cid in top_campaigns[:5]:
    s = hist[hist.campaign_id == cid]
    axes[0, 1].plot(s.day, s.budget, marker="o", label=f"Компания {cid}")
axes[0, 1].set_title("5 стабльных компаний")
axes[0, 1].set_xlabel("День")
axes[0, 1].set_ylabel("Бюджет")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. распределение компаний
camp_type_counts = campaigns["type"].value_counts()
axes[1, 0].pie(camp_type_counts.values, labels=camp_type_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title("Распределение типов кампаний")

# 4. стабильность vs профит
campaign_summary = scores_df.groupby("campaign_id").agg({
    "profit": "mean",
    "discounted_score": "sum"
}).reset_index()
campaign_summary = campaign_summary.merge(
    campaigns[["campaign_id", "stability", "type"]],
    on="campaign_id"
)

for camp_type in ["sniper", "tractor", "ballast"]:
    mask = campaign_summary["type"] == camp_type
    axes[1, 1].scatter(
        campaign_summary.loc[mask, "stability"],
        campaign_summary.loc[mask, "profit"],
        label=camp_type,
        alpha=0.6
    )
axes[1, 1].set_xlabel("Стабильность")
axes[1, 1].set_ylabel("Средний дневной профит")
axes[1, 1].set_title("Стабильность vs профит по типу команий")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Ключевые итоги

# 1. Эффективность по типам
print("\n Эффективность компаний по типам:")
for camp_type in ["sniper", "tractor", "ballast"]:
    type_campaigns = campaigns[campaigns.type == camp_type]
    if len(type_campaigns) > 0:
        avg_stability = type_campaigns.stability.mean()
        avg_margin = type_campaigns.margin.mean()
        print(f"   {camp_type.upper():10} | Count: {len(type_campaigns):2} | "
              f"Сред. стабильность: {avg_stability:.2f} | Маржа: {avg_margin:.2f}")

# 2. Аномалии
anomalies = campaigns[campaigns.is_anomaly]
if len(anomalies) > 0:
    print(f"\n {len(anomalies)} Компаний-аномалий")
    print(f"Их средняя маржа снизилась до: {anomalies.margin.mean():.2f}")

# 3. Итоговая эффективность
print(f"\n Итоговая эффективность:")
print(f"   Прибыль за {DAYS} дней: {metrics.profit.sum():.2f}")
print(f"   Средняя DRR: {metrics.drr.mean():.3f}")
print(f"   Средние показатели активности кампаний: {metrics.active_campaigns.mean():.1f}")
print(f"   Общий Discounted: {total_score:.2f}")