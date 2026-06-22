"""SARS Tax Engine - South African Revenue Service tax calculations.

Brackets and rebates for the 2024/2025 tax year (1 March 2024 - 28 Feb 2025).
Source: SARS official tax tables.
"""
from typing import List, Dict

# Tax brackets: (upper_bound, base_tax_at_lower, marginal_rate, lower_bound)
# Upper bound of None means no limit (top bracket).
SARS_BRACKETS_2024_2025 = [
    {"lower": 0,        "upper": 237_100,    "base": 0,        "rate": 0.18},
    {"lower": 237_100,  "upper": 370_500,    "base": 42_678,   "rate": 0.26},
    {"lower": 370_500,  "upper": 512_800,    "base": 77_362,   "rate": 0.31},
    {"lower": 512_800,  "upper": 673_000,    "base": 121_475,  "rate": 0.36},
    {"lower": 673_000,  "upper": 857_900,    "base": 179_147,  "rate": 0.39},
    {"lower": 857_900,  "upper": 1_817_000,  "base": 251_258,  "rate": 0.41},
    {"lower": 1_817_000,"upper": None,       "base": 644_489,  "rate": 0.45},
]

# Primary rebate (under 65)
PRIMARY_REBATE = 17_235
SECONDARY_REBATE = 9_444   # 65+
TERTIARY_REBATE = 3_145    # 75+

# Tax thresholds (below this no tax payable)
THRESHOLD_UNDER_65 = 95_750


def calculate_annual_tax(taxable_income: float, age: int = 30) -> Dict:
    """Calculate annual tax liability using SARS brackets + age rebate."""
    if taxable_income <= 0:
        return {
            "taxable_income": 0.0,
            "tax_before_rebate": 0.0,
            "rebate": 0.0,
            "tax_payable": 0.0,
            "bracket": None,
            "effective_rate": 0.0,
        }

    bracket_used = None
    tax_before_rebate = 0.0
    for b in SARS_BRACKETS_2024_2025:
        if b["upper"] is None or taxable_income <= b["upper"]:
            tax_before_rebate = b["base"] + (taxable_income - b["lower"]) * b["rate"]
            bracket_used = b
            break

    rebate = PRIMARY_REBATE
    if age >= 65:
        rebate += SECONDARY_REBATE
    if age >= 75:
        rebate += TERTIARY_REBATE

    tax_payable = max(0.0, tax_before_rebate - rebate)
    effective_rate = (tax_payable / taxable_income) if taxable_income > 0 else 0.0

    return {
        "taxable_income": round(taxable_income, 2),
        "tax_before_rebate": round(tax_before_rebate, 2),
        "rebate": round(rebate, 2),
        "tax_payable": round(tax_payable, 2),
        "bracket": {
            "lower": bracket_used["lower"],
            "upper": bracket_used["upper"],
            "rate": bracket_used["rate"],
        } if bracket_used else None,
        "effective_rate": round(effective_rate, 4),
    }


def annual_summary(monthly_records: List[Dict], expenses: List[Dict], age: int = 30) -> Dict:
    """Aggregate the user's year.

    monthly_records: list of {income, tax_paid, month, year}
    expenses: list of {amount, category, date, deductible}
    """
    total_income = sum(float(r.get("income", 0)) for r in monthly_records)
    total_paye = sum(float(r.get("tax_paid", 0)) for r in monthly_records)

    # Only Medical + Business expenses count as deductible in our simple model
    deductible_categories = {"Medical", "Business"}
    total_deductions = sum(
        float(e.get("amount", 0))
        for e in expenses
        if e.get("category") in deductible_categories
    )

    taxable_income = max(0.0, total_income - total_deductions)
    tax = calculate_annual_tax(taxable_income, age=age)
    refund_or_owed = round(total_paye - tax["tax_payable"], 2)  # positive = refund

    # By category
    category_breakdown: Dict[str, float] = {}
    for e in expenses:
        cat = e.get("category", "Other")
        category_breakdown[cat] = category_breakdown.get(cat, 0.0) + float(e.get("amount", 0))

    return {
        "total_income": round(total_income, 2),
        "total_paye_paid": round(total_paye, 2),
        "total_deductions": round(total_deductions, 2),
        "taxable_income": tax["taxable_income"],
        "tax_payable": tax["tax_payable"],
        "tax_before_rebate": tax["tax_before_rebate"],
        "rebate": tax["rebate"],
        "bracket": tax["bracket"],
        "effective_rate": tax["effective_rate"],
        "refund_or_owed": refund_or_owed,
        "status": "refund" if refund_or_owed > 0 else ("owed" if refund_or_owed < 0 else "balanced"),
        "category_breakdown": {k: round(v, 2) for k, v in category_breakdown.items()},
    }
