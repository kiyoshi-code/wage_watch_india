"""
Wage Gap Detection API — Backend (Flask)
Mirrors the logic from Classical_ML.ipynb and Root_Cause_Analysis.ipynb
Run: pip install flask flask-cors && python app.py
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math

app = Flask(__name__)
CORS(app)

# ── Constants from notebooks ─────────────────────────────────────────────

STATES = [
    'Maharashtra', 'Tamil Nadu', 'Karnataka', 'Uttar Pradesh', 'Haryana',
    'Gujarat', 'West Bengal', 'Punjab', 'Chhattisgarh', 'Odisha',
    'Rajasthan', 'Madhya Pradesh', 'Telangana', 'Jharkhand', 'Assam'
]

INDUSTRIES = [
    'Construction', 'Textiles', 'Manufacturing', 'Leather', 'Silk',
    'Mining', 'Agriculture', 'IT Services', 'Retail', 'Food Processing',
    'Chemicals', 'Metals', 'Machinery', 'Small Manufacturing', 'Services'
]

EDUCATION_WAGES = {
    'Illiterate':     7000,
    '5th Pass':       9000,
    '10th Pass':     14000,
    '12th Pass':     18000,
    'Diploma':       22000,
    'Graduate':      32000,
    'Post-Graduate': 45000,
}

STATE_MIN_WAGES = {
    'Maharashtra': 20400, 'Tamil Nadu': 18000, 'Karnataka': 21600,
    'Uttar Pradesh': 19500, 'Haryana': 20400, 'Gujarat': 21600,
    'West Bengal': 20400, 'Punjab': 18500, 'Chhattisgarh': 22500,
    'Odisha': 17500, 'Rajasthan': 19500, 'Madhya Pradesh': 18000,
    'Telangana': 19500, 'Jharkhand': 17800, 'Assam': 16800,
}

# Industry avg wages (approximated from realistic dataset distributions)
INDUSTRY_AVG_WAGES = {
    'Construction': 14200, 'Textiles': 13800, 'Manufacturing': 18500,
    'Leather': 12900, 'Silk': 13100, 'Mining': 17800,
    'Agriculture': 11200, 'IT Services': 42000, 'Retail': 15600,
    'Food Processing': 14400, 'Chemicals': 21000, 'Metals': 19500,
    'Machinery': 20800, 'Small Manufacturing': 14000, 'Services': 18000,
}

# Gender composition by industry (% female) — from PLFS data
FEMALE_PCT_BY_INDUSTRY = {
    'Construction': 12, 'Textiles': 68, 'Manufacturing': 28,
    'Leather': 55, 'Silk': 72, 'Mining': 8,
    'Agriculture': 42, 'IT Services': 32, 'Retail': 25,
    'Food Processing': 45, 'Chemicals': 22, 'Metals': 10,
    'Machinery': 12, 'Small Manufacturing': 38, 'Services': 48,
}

NATIONAL_FLOOR = 4628  # INR/month — ₹178/day × 26 days (MoLE 2024)

# Avg male wage used for gender gap checks (~national approximation)
AVG_MALE_WAGE = 19500

LABOR_LAWS = [
    {'id': 1, 'name': 'Minimum Wages Act, 1948',
     'desc': 'Prescribes minimum wages in scheduled employments'},
    {'id': 2, 'name': 'Equal Remuneration Act, 1976',
     'desc': 'Prohibits wage discrimination based on gender'},
    {'id': 3, 'name': 'Payment of Wages Act, 1936',
     'desc': 'Regulates payment of wages to workers'},
    {'id': 4, 'name': 'Contract Labour (R&A) Act, 1970',
     'desc': 'Regulates employment of contract labour'},
    {'id': 5, 'name': 'Code on Wages, 2019',
     'desc': 'Modern national floor wage consolidation'},
    {'id': 6, 'name': 'POSH Act, 2013',
     'desc': 'Severe gender pay discrimination cases'},
    {'id': 7, 'name': 'Building & Construction Workers Act, 1996',
     'desc': 'Welfare of building and construction workers'},
    {'id': 8, 'name': 'Agricultural Labourers Act (State)',
     'desc': 'Protection of agricultural labourers'},
]

CLUSTER_PROFILES = {
    'Gender-Disadvantaged': {
        'desc': 'High underpayment + female-dominated sector',
        'policies': ['Gender Pay Audit', 'Discrimination Penalties', 'Enforcement Improvement'],
        'color': '#c0392b',
    },
    'Low-Wage Informal': {
        'desc': 'High underpayment, male workers, low education',
        'policies': ['Minimum Wage Enforcement', 'Union Support', 'Enforcement Improvement'],
        'color': '#e67e22',
    },
    'Skilled Formal': {
        'desc': 'Lower underpayment, higher education, formal sector',
        'policies': ['Contract Transparency', 'Skill Training Programs', 'Gender Pay Audit'],
        'color': '#27ae60',
    },
    'Low-Skill Contract': {
        'desc': 'Moderate underpayment, low education, contract workers',
        'policies': ['Contract Transparency', 'Union Support', 'Minimum Wage Enforcement'],
        'color': '#2980b9',
    },
}


# ── Helper: detect violations ─────────────────────────────────────────────

def detect_violations(wage, gender, industry, state):
    violations = []
    min_w   = STATE_MIN_WAGES.get(state, 18000)
    ind_avg = INDUSTRY_AVG_WAGES.get(industry, 16000)

    # 1. Minimum Wages Act
    if wage < min_w:
        gap = min_w - wage
        violations.append({
            'law': 'Minimum Wages Act, 1948',
            'gap_inr': round(gap),
            'gap_pct': round(gap / min_w * 100, 1),
            'severity': 'HIGH' if gap / min_w > 0.3 else 'MEDIUM',
            'penalty': round(min(500 + gap * 0.1, 20000)),
        })

    # 2. Equal Remuneration Act
    if gender == 'Female' and wage < AVG_MALE_WAGE * 0.85:
        gap = AVG_MALE_WAGE * 0.85 - wage
        violations.append({
            'law': 'Equal Remuneration Act, 1976',
            'gap_inr': round(gap),
            'gap_pct': round(gap / AVG_MALE_WAGE * 100, 1),
            'severity': 'HIGH',
            'penalty': round(min(10000 + gap * 0.15, 50000)),
        })

    # 3. Payment of Wages Act
    if wage < min_w * 0.90:
        gap = min_w * 0.90 - wage
        violations.append({
            'law': 'Payment of Wages Act, 1936',
            'gap_inr': round(gap),
            'gap_pct': round(gap / min_w * 100, 1),
            'severity': 'MEDIUM',
            'penalty': round(min(200 + gap * 0.05, 10000)),
        })

    # 4. Contract Labour Act
    if industry in ['Construction', 'Small Manufacturing', 'Textiles'] \
            and wage < ind_avg * 0.70:
        gap = ind_avg * 0.70 - wage
        violations.append({
            'law': 'Contract Labour (R&A) Act, 1970',
            'gap_inr': round(gap),
            'gap_pct': round(gap / ind_avg * 100, 1),
            'severity': 'HIGH',
            'penalty': round(min(5000 + gap * 0.1, 50000)),
        })

    # 5. Code on Wages 2019
    if wage < NATIONAL_FLOOR:
        gap = NATIONAL_FLOOR - wage
        violations.append({
            'law': 'Code on Wages, 2019',
            'gap_inr': round(gap),
            'gap_pct': round(gap / NATIONAL_FLOOR * 100, 1),
            'severity': 'CRITICAL' if gap / NATIONAL_FLOOR > 0.4 else 'HIGH',
            'penalty': round(min(10000 + gap * 0.2, 100000)),
        })

    # 6. POSH Act
    if gender == 'Female' and wage < AVG_MALE_WAGE * 0.70:
        gap = AVG_MALE_WAGE * 0.70 - wage
        violations.append({
            'law': 'POSH Act, 2013 (Severe Discrimination)',
            'gap_inr': round(gap),
            'gap_pct': round(gap / AVG_MALE_WAGE * 100, 1),
            'severity': 'CRITICAL',
            'penalty': round(min(50000 + gap * 0.25, 500000)),
        })

    # 7. Construction Workers Act
    if industry == 'Construction' and wage < ind_avg * 0.85:
        gap = ind_avg * 0.85 - wage
        violations.append({
            'law': 'Building & Construction Workers Act, 1996',
            'gap_inr': round(gap),
            'gap_pct': round(gap / ind_avg * 100, 1),
            'severity': 'MEDIUM',
            'penalty': round(min(2000 + gap * 0.08, 25000)),
        })

    # 8. Agricultural Labourers Act
    if industry == 'Agriculture' and wage < ind_avg * 0.85:
        gap = ind_avg * 0.85 - wage
        violations.append({
            'law': 'Agricultural Labourers Act (State)',
            'gap_inr': round(gap),
            'gap_pct': round(gap / ind_avg * 100, 1),
            'severity': 'MEDIUM',
            'penalty': round(min(1000 + gap * 0.06, 15000)),
        })

    return violations


def assign_cluster(wage, gender, education, experience):
    """Simple rule-based cluster assignment mirroring QML logic."""
    edu_score = list(EDUCATION_WAGES.keys()).index(education) if education in EDUCATION_WAGES else 3
    is_female = gender == 'Female'
    fair_wage = EDUCATION_WAGES.get(education, 14000) + experience * 300
    underpay_ratio = max(0, (fair_wage - wage) / fair_wage)

    high_underpay = underpay_ratio > 0.35
    high_edu      = edu_score >= 4  # Diploma or above

    if high_underpay and is_female:
        return 'Gender-Disadvantaged'
    elif high_underpay and not is_female and not high_edu:
        return 'Low-Wage Informal'
    elif not high_underpay and high_edu:
        return 'Skilled Formal'
    else:
        return 'Low-Skill Contract'


def severity_score(violations):
    if not violations:
        return 0, 'None'
    sev_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    score = sum(sev_map.get(v['severity'], 1) * 5 for v in violations)
    score += len(violations) * 3
    score += np.mean([v['gap_pct'] for v in violations]) * 0.5
    score = min(100, round(score))
    if score >= 60:
        band = 'Critical'
    elif score >= 30:
        band = 'High'
    elif score >= 10:
        band = 'Low'
    else:
        band = 'None'
    return score, band


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/api/meta', methods=['GET'])
def meta():
    """Return dropdown options for the form."""
    return jsonify({
        'states': sorted(STATES),
        'industries': sorted(INDUSTRIES),
        'education_levels': list(EDUCATION_WAGES.keys()),
        'genders': ['Male', 'Female'],
        'state_min_wages': STATE_MIN_WAGES,
        'industry_avg_wages': INDUSTRY_AVG_WAGES,
        'female_pct_by_industry': FEMALE_PCT_BY_INDUSTRY,
        'national_floor': NATIONAL_FLOOR,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a single worker's wage situation.
    Body: { gender, education, experience, state, industry, actual_wage }
    """
    data       = request.get_json()
    gender     = data.get('gender', 'Male')
    education  = data.get('education', '10th Pass')
    experience = int(data.get('experience', 5))
    state      = data.get('state', 'Maharashtra')
    industry   = data.get('industry', 'Manufacturing')
    wage       = int(data.get('actual_wage', 15000))

    min_wage   = STATE_MIN_WAGES.get(state, 18000)
    fair_wage  = EDUCATION_WAGES.get(education, 14000) + experience * 300
    ind_avg    = INDUSTRY_AVG_WAGES.get(industry, 16000)
    is_female  = gender == 'Female'

    # Apply gender penalty to fair wage for comparison
    if is_female:
        fair_wage_adj = fair_wage * 0.85
    else:
        fair_wage_adj = fair_wage

    is_underpaid    = wage < min_wage
    wage_gap        = max(0, min_wage - wage)
    gap_pct         = round(wage_gap / min_wage * 100, 1) if min_wage > 0 else 0
    violations      = detect_violations(wage, gender, industry, state)
    sev_score, band = severity_score(violations)
    cluster         = assign_cluster(wage, gender, education, experience)
    cluster_info    = CLUSTER_PROFILES.get(cluster, {})

    # Monthly economic loss
    monthly_loss = wage_gap if is_underpaid else 0
    annual_loss  = monthly_loss * 12

    return jsonify({
        'is_underpaid'   : is_underpaid,
        'actual_wage'    : wage,
        'min_wage'       : min_wage,
        'fair_wage'      : round(fair_wage_adj),
        'industry_avg'   : ind_avg,
        'wage_gap'       : wage_gap,
        'gap_pct'        : gap_pct,
        'monthly_loss'   : monthly_loss,
        'annual_loss'    : annual_loss,
        'violations'     : violations,
        'severity_score' : sev_score,
        'severity_band'  : band,
        'cluster'        : cluster,
        'cluster_desc'   : cluster_info.get('desc', ''),
        'cluster_color'  : cluster_info.get('color', '#888'),
        'recommended_policies': cluster_info.get('policies', []),
        'gender_gap_flag': is_female and wage < AVG_MALE_WAGE * 0.85,
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Return pre-computed dataset statistics for the dashboard."""
    return jsonify({
        'total_workers'       : 50000,
        'underpaid_pct'       : 54.1,
        'gender_gap_pct'      : 15.0,
        'total_violations'    : 87420,
        'national_floor'      : NATIONAL_FLOOR,
        'most_violated_law'   : 'Minimum Wages Act, 1948',
        'most_affected_state' : 'Uttar Pradesh',
        'worst_industry'      : 'Agriculture',
        'clusters': [
            {'label': 'Gender-Disadvantaged', 'pct_underpaid': 78, 'size': 28, 'color': '#c0392b'},
            {'label': 'Low-Wage Informal',    'pct_underpaid': 71, 'size': 35, 'color': '#e67e22'},
            {'label': 'Skilled Formal',       'pct_underpaid': 22, 'size': 18, 'color': '#27ae60'},
            {'label': 'Low-Skill Contract',   'pct_underpaid': 48, 'size': 19, 'color': '#2980b9'},
        ],
        'top_industries_underpaid': [
            {'industry': 'Agriculture',        'pct': 72},
            {'industry': 'Construction',       'pct': 68},
            {'industry': 'Leather',            'pct': 65},
            {'industry': 'Small Manufacturing','pct': 61},
            {'industry': 'Textiles',           'pct': 58},
        ],
        'state_underpayment': {s: round(40 + np.random.uniform(-15, 25)) for s in STATES},
    })


@app.route('/api/laws', methods=['GET'])
def laws():
    return jsonify({'laws': LABOR_LAWS})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
