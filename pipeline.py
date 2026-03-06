import pandas as pd
import random

selected_features = [
    'avg_spend_ratio',
    'std_spend_ratio',
    'months_over_spending',
    'avg_food_pct_salary',
    'avg_shopping_pct_salary'
]

baseline_categories = [
    'avg_food_pct_salary',
    'avg_drink_pct_salary',
    'avg_shopping_pct_salary',
    'avg_transport_pct_salary',
    'avg_bills_pct_salary',
    'avg_health_pct_salary',
    'avg_entertainment_pct_salary'
]


def build_features(salary, spend_dict):

    salary = float(salary) if salary else 0.0
    salary_safe = salary if salary > 0 else 0.0

    total_spend = sum(float(v) for v in spend_dict.values())

    avg_spend_ratio = (total_spend / salary_safe) if salary_safe > 0 else 0.0
    std_spend_ratio = 0.0
    months_over_spending = 1 if (salary_safe > 0 and total_spend > salary_safe) else 0

    data = {
        "avg_spend_ratio": avg_spend_ratio,
        "std_spend_ratio": std_spend_ratio,
        "months_over_spending": months_over_spending,

        "avg_food_pct_salary": (spend_dict["food"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_drink_pct_salary": (spend_dict["drink"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_shopping_pct_salary": (spend_dict["shopping"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_transport_pct_salary": (spend_dict["transport"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_bills_pct_salary": (spend_dict["bills"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_health_pct_salary": (spend_dict["health"] / salary_safe) if salary_safe > 0 else 0.0,
        "avg_entertainment_pct_salary": (spend_dict["entertainment"] / salary_safe) if salary_safe > 0 else 0.0,
    }

    return pd.DataFrame([data])


def prepare_for_clustering(df):
    return df[selected_features]


def generate_final_recommendation(row, baseline_mean_df, threshold=0.05):

    categories_info = {
        'avg_food_pct_salary': "الأكل",
        'avg_drink_pct_salary': "المشروبات",
        'avg_shopping_pct_salary': "التسوق",
        'avg_transport_pct_salary': "المواصلات",
        'avg_bills_pct_salary': "الفواتير",
        'avg_health_pct_salary': "الصحة",
        'avg_entertainment_pct_salary': "الترفيه"
    }

    # النصائح الجديدة (معدلة لتكون مقنعة وعامية)
    smart_tips = {
        "الأكل": [
            "الأكل من برا ممكن يزود مصاريفك، فحاول تعتمد على أكل البيت أيام أكتر.",
            "الطلب من المطاعم مريح طبعًا، بس تقليله شوية خلال الأسبوع ممكن يوفر معاك مبلغ كويس.",
            "حتى لو خليت الأكل من برا مرة أو مرتين في الشهر، ده ممكن يقلل مصاريفك بشكل ملحوظ."
        ],
        "المشروبات": [
            "القعدة في الكافيهات لطيفة، بس تقليلها شوية خلال الأسبوع ممكن يوفر معاك فلوس.",
            "تجربة تعمل القهوة أو المشروبات في البيت أحيانًا ممكن تكون أوفر بكتير.",
            "المشروبات الجاهزة بتبان حاجة بسيطة، لكن تكرارها ممكن يزود المصاريف من غير ما تحس."
        ],
        "التسوق": [
            "تحديد ميزانية للتسوق كل شهر ممكن يساعدك تتحكم في مصاريفك.",
            "حاول تشتري الحاجات اللي محتاجها بس وتسيب المشتريات غير الضرورية.",
            "التفكير قبل أي شراء جديد ممكن يقلل المصاريف اللي ملهاش لازمة."
        ],
        "المواصلات": [
            "محاولة تجمع مشاويرك في مرة واحدة ممكن تقلل التكلفة شوية.",
            "استخدام المواصلات العامة أحيانًا ممكن يساعدك توفر جزء من المصاريف.",
            "تقليل استخدام التطبيقات أو المشاوير الفردية ممكن يوفر معاك فلوس."
        ],
        "الفواتير": [
            "الفواتير بتكون مصاريف ثابتة غالبًا، لكن تقليل الاستهلاك شوية ممكن يعمل فرق.",
            "اقفل الأجهزة اللي مش بتستخدمها ممكن يقلل استهلاك الكهرباء.",
            "متابعة استهلاك الكهرباء والإنترنت كل فترة ممكن يساعدك تتحكم في الفاتورة."
        ],
        "الصحة": [
            "مقارنة أسعار الأدوية بين الصيدليات ممكن تساعدك توفر شوية.",
            "شراء الأدوية الضرورية بس ممكن يقلل المصاريف.",
            "استخدام التأمين الصحي لو متوفر ممكن يقلل جزء من تكلفة العلاج."
        ],
        "الترفيه": [
            "الترفيه مهم طبعًا، لكن الخروجات الكتير ممكن تزود المصاريف بسرعة.",
            "اختيار أنشطة بسيطة أحيانًا ممكن يخليك تستمتع من غير ما تصرف كتير.",
            "تحديد ميزانية للترفيه كل شهر ممكن يساعدك توازن بين المتعة والمصاريف."
        ]
    }

    cluster_id = row['cluster']
    spend_ratio = float(row['avg_spend_ratio'])
    ratio_pct = round(spend_ratio * 100)

    financial_states = [
        f"بتصرف حوالي {ratio_pct}% من مرتبك.",
        f"إنفاقك حوالي {ratio_pct}% من دخلك.",
        f"مصاريفك الشهرية {ratio_pct}% من المرتب."
    ]

    if spend_ratio < 0.75:
        level_msg = "الوضع المالي عندك مستقر 👍"
    elif spend_ratio < 1:
        level_msg = "مصاريفك قريبة من الحد الأعلى ⚠️"
    else:
        level_msg = "مصاريفك أعلى من مرتبك 🚨"

    deviations = []

    for col, name in categories_info.items():

        current = float(row[col])
        baseline = float(baseline_mean_df.loc[cluster_id, col])

        if baseline == 0:
            continue

        relative_diff = (current - baseline) / baseline

        if relative_diff > threshold:
            percent = round(relative_diff * 100)
            deviations.append((name, percent))

    if not deviations:
        return f"{random.choice(financial_states)} {level_msg}. مصروفاتك متوازنة."

    deviations_sorted = sorted(deviations, key=lambda x: x[1], reverse=True)

    top_deviations = deviations_sorted[:2]

    names = [d[0] for d in top_deviations]

    categories_text = " و ".join(names)

    tips = [random.choice(smart_tips[n]) for n in names]

    tips_text = " ".join(tips)

    return (
        f"{random.choice(financial_states)} {level_msg}. "
        f"أعلى إنفاق عندك في {categories_text}. "
        f"{tips_text}"
    )