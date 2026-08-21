import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Return-Risk Scorer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Return-Risk Scorer & Financial Optimizer")
st.caption(
    "AI Risk Manager | Cost-Sensitive Return Detection & "
    "Real-Time Transaction Risk Scoring"
)


# =========================================================
# DATA + MODEL
# =========================================================

@st.cache_resource
def load_data_and_train(csv_path="train.csv"):

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    data = pd.read_csv(csv_path)

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # -----------------------------------------------------
    # EXPECTED COLUMNS
    # -----------------------------------------------------

    required_columns = [
        "customer_age",
        "product_price",
        "discount_percent",
        "product_rating",
        "past_purchase_count",
        "past_return_rate",
        "delivery_delay_days",
        "session_length_minutes",
        "num_product_views",
        "device_type",
        "product_category",
        "shipping_method",
        "payment_method",
        "used_coupon",
        "returned"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    numeric_features = [
        "customer_age",
        "product_price",
        "discount_percent",
        "product_rating",
        "past_purchase_count",
        "past_return_rate",
        "delivery_delay_days",
        "session_length_minutes",
        "num_product_views",
        "used_coupon"
    ]

    categorical_features = [
        "device_type",
        "product_category",
        "shipping_method",
        "payment_method"
    ]

    # -----------------------------------------------------
    # NUMERIC CONVERSION
    # -----------------------------------------------------

    for col in numeric_features + ["returned"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # -----------------------------------------------------
    # DATA CLEANING
    # -----------------------------------------------------

    # Number of views cannot be negative
    data["num_product_views"] = (
        data["num_product_views"]
        .clip(lower=0)
    )

    # Delivery delay cannot be negative
    data["delivery_delay_days"] = (
        data["delivery_delay_days"]
        .clip(lower=0)
    )

    # Purchase count cannot be negative
    data["past_purchase_count"] = (
        data["past_purchase_count"]
        .clip(lower=0)
    )

    # Return rate must be 0-1
    data["past_return_rate"] = (
        data["past_return_rate"]
        .clip(lower=0, upper=1)
    )

    # Discount percentage must be 0-100
    data["discount_percent"] = (
        data["discount_percent"]
        .clip(lower=0, upper=100)
    )

    # Product rating must be 0-5
    data["product_rating"] = (
        data["product_rating"]
        .clip(lower=0, upper=5)
    )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    for col in numeric_features:

        data[col] = data[col].fillna(
            data[col].median()
        )

    for col in categorical_features:

        data[col] = (
            data[col]
            .fillna("unknown")
            .astype(str)
        )

    data["returned"] = (
        data["returned"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------------------
    # REMOVE INVALID TARGET ROWS
    # -----------------------------------------------------

    data = data[
        data["returned"].isin([0, 1])
    ].copy()

    # -----------------------------------------------------
    # X / y
    # -----------------------------------------------------

    X = data[
        numeric_features +
        categorical_features
    ]

    y = data["returned"]

    # =====================================================
    # IMPORTANT:
    #
    # 60% TRAIN
    # 20% VALIDATION
    # 20% TEST
    #
    # TEST SET REMAINS UNTOUCHED UNTIL FINAL EVALUATION.
    # =====================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # -----------------------------------------------------
    # ENCODER
    # -----------------------------------------------------

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        # Older sklearn compatibility
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features
            ),
            (
                "categorical",
                encoder,
                categorical_features
            )
        ]
    )

    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                xgb_model
            )
        ]
    )

    # -----------------------------------------------------
    # TRAIN ONLY ON TRAINING DATA
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # VALIDATION PROBABILITIES
    #
    # Used ONLY for threshold selection.
    # -----------------------------------------------------

    validation_probs = (
        model.predict_proba(X_val)[:, 1]
    )

    # -----------------------------------------------------
    # TEST PROBABILITIES
    #
    # These are kept untouched for final evaluation.
    # -----------------------------------------------------

    test_probs = (
        model.predict_proba(X_test)[:, 1]
    )

    return (
        model,
        X_val,
        y_val,
        validation_probs,
        X_test,
        y_test,
        test_probs,
        numeric_features,
        categorical_features
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:

    (
        model,
        X_val,
        y_val,
        validation_probs,
        X_test,
        y_test,
        test_probs,
        numeric_features,
        categorical_features
    ) = load_data_and_train("train.csv")

except Exception as e:

    st.error(
        f"Failed to load/train model: {e}"
    )

    st.stop()


# =========================================================
# SIDEBAR — COST MATRIX
# =========================================================

st.sidebar.header("💰 Financial Cost Matrix")

st.sidebar.write(
    "Define the estimated business impact of model errors."
)

cost_per_fn = st.sidebar.number_input(
    "Estimated Loss per Missed Return / FN ($)",
    min_value=1.0,
    value=15.0,
    step=1.0
)

cost_per_fp = st.sidebar.number_input(
    "Estimated Customer Friction Cost / FP ($)",
    min_value=1.0,
    value=5.0,
    step=1.0
)


# =========================================================
# COST FUNCTION
# =========================================================

def calculate_cost(
    y_true,
    probabilities,
    threshold,
    fn_cost,
    fp_cost
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = np.sum(
        (predictions == 1) &
        (y_true == 1)
    )

    fp = np.sum(
        (predictions == 1) &
        (y_true == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (y_true == 1)
    )

    tn = np.sum(
        (predictions == 0) &
        (y_true == 0)
    )

    # Total business cost
    total_cost = (
        fn * fn_cost +
        fp * fp_cost
    )

    # The amount of loss avoided compared with
    # allowing every return to pass through.
    avoided_loss = (
        (tp + fn) * fn_cost
        - total_cost
    )

    return (
        total_cost,
        avoided_loss,
        tp,
        fp,
        fn,
        tn
    )


# =========================================================
# FIND COST-OPTIMAL THRESHOLD
#
# IMPORTANT:
# This happens ONLY on validation data.
# =========================================================

threshold_range = np.arange(
    0.05,
    0.96,
    0.01
)

validation_costs = []
validation_savings = []


for threshold in threshold_range:

    (
        total_cost,
        avoided_loss,
        tp,
        fp,
        fn,
        tn
    ) = calculate_cost(
        y_val,
        validation_probs,
        threshold,
        cost_per_fn,
        cost_per_fp
    )

    validation_costs.append(
        total_cost
    )

    validation_savings.append(
        avoided_loss
    )


optimal_index = np.argmin(
    validation_costs
)

optimal_threshold = round(
    float(threshold_range[optimal_index]),
    2
)


# =========================================================
# ACTIVE THRESHOLD
# =========================================================

st.sidebar.markdown("---")

st.sidebar.header("⚡ Decision Threshold")

if "threshold_val" not in st.session_state:

    st.session_state.threshold_val = (
        optimal_threshold
    )


if st.sidebar.button(
    "Reset to Validation-Optimal Threshold"
):

    st.session_state.threshold_val = (
        optimal_threshold
    )


selected_threshold = st.sidebar.slider(
    "Active Classification Threshold",
    min_value=0.05,
    max_value=0.95,
    value=st.session_state.threshold_val,
    step=0.01,
    key="threshold_val"
)


st.sidebar.success(
    f"Validation-optimal threshold: "
    f"**{optimal_threshold:.2f}**"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(
    [
        "📊 Model Evaluation & Cost Trade-offs",
        "🛒 Live Transaction Simulator"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader(
        "🔒 Final Evaluation on Untouched Test Set"
    )

    st.info(
        "The model was trained on the training set. "
        "The decision threshold was selected using the "
        "validation set. The metrics below are calculated "
        "only on the untouched test set."
    )

    # -----------------------------------------------------
    # FINAL TEST PREDICTIONS
    # -----------------------------------------------------

    test_preds = (
        test_probs >= selected_threshold
    ).astype(int)

    # -----------------------------------------------------
    # CONFUSION MATRIX VALUES
    # -----------------------------------------------------

    (
        test_cost,
        test_avoided_loss,
        tp,
        fp,
        fn,
        tn
    ) = calculate_cost(
        y_test,
        test_probs,
        selected_threshold,
        cost_per_fn,
        cost_per_fp
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    precision = precision_score(
        y_test,
        test_preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_preds,
        zero_division=0
    )

    total_actual_returns = tp + fn
    total_actual_clean = tn + fp

    if total_actual_clean > 0:

        false_positive_rate = (
            fp / total_actual_clean
        )

    else:

        false_positive_rate = 0.0

    if total_actual_returns > 0:

        false_negative_rate = (
            fn / total_actual_returns
        )

    else:

        false_negative_rate = 0.0

    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Precision",
        f"{precision:.2%}"
    )

    m2.metric(
        "Recall",
        f"{recall:.2%}"
    )

    m3.metric(
        "False Positive Rate",
        f"{false_positive_rate:.2%}"
    )

    m4.metric(
        "False Negative Rate",
        f"{false_negative_rate:.2%}"
    )


    st.markdown("---")


    # -----------------------------------------------------
    # BUSINESS METRICS
    # -----------------------------------------------------

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Active Threshold",
        f"{selected_threshold:.2f}"
    )

    b2.metric(
        "True Positives",
        int(tp)
    )

    b3.metric(
        "False Positives",
        int(fp)
    )

    b4.metric(
        "False Negatives",
        int(fn)
    )


    st.markdown("---")


    # =====================================================
    # VALIDATION COST CURVE
    # =====================================================

    col_left, col_right = st.columns(2)


    with col_left:

        st.write(
            "### 💰 Validation Cost vs Threshold"
        )

        fig_cost = px.line(
            x=threshold_range,
            y=validation_costs,
            labels={
                "x": "Classification Threshold",
                "y": "Estimated Business Cost ($)"
            },
            title=(
                "Threshold Optimization "
                "(Validation Set)"
            )
        )

        fig_cost.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text="Selected Threshold"
        )

        st.plotly_chart(
            fig_cost,
            use_container_width=True
        )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    with col_right:

        st.write(
            "### 🔍 Test-Set Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            test_preds
        )

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[
                    "Predicted Clean",
                    "Predicted Returned"
                ],
                y=[
                    "Actual Clean",
                    "Actual Returned"
                ],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={
                    "size": 18
                }
            )
        )

        fig_cm.update_layout(
            xaxis_title="Prediction",
            yaxis_title="Actual"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )


    # =====================================================
    # BUSINESS INTERPRETATION
    # =====================================================

    st.markdown("---")

    st.subheader(
        "💼 Business Impact"
    )

    st.write(
        f"""
        At the selected threshold of **{selected_threshold:.2f}**:

        - **{tp}** actual returns were correctly identified.
        - **{fn}** returns were missed.
        - **{fp}** legitimate transactions were incorrectly flagged.
        - **{tn}** legitimate transactions were correctly approved.

        Estimated cost:

        **({fn} × ${cost_per_fn:.2f}) + "
        f"({fp} × ${cost_per_fp:.2f}) = "
        f"${test_cost:,.2f}**

        The system therefore turns the ML prediction into a "
        **cost-aware business decision**, rather than simply "
        "using the default 0.50 probability threshold.
        """
    )


    # =====================================================
    # DATASET SPLIT
    # =====================================================

    st.markdown("---")

    st.subheader(
        "📚 Evaluation Methodology"
    )

    st.write(
        """
        **Training set:** 60% — used to train XGBoost.

        **Validation set:** 20% — used to select the
        cost-optimal classification threshold.

        **Held-out test set:** 20% — never used for training
        or threshold selection. Precision, recall, false
        positive rate, false negative rate and business cost
        are reported from this set.
        """
    )


# =========================================================
# TAB 2 — LIVE SIMULATOR
# =========================================================

with tab2:

    st.subheader(
        "🛒 Real-Time Return Risk Simulator"
    )

    st.write(
        "Enter a customer's transaction and behavioral "
        "attributes to simulate a live risk decision."
    )


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # CUSTOMER + PRODUCT
    # -----------------------------------------------------

    with col1:

        input_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=30
        )

        input_price = st.number_input(
            "Product Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=50.0
        )

        input_discount = st.slider(
            "Discount Percentage",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0
        )

        input_rating = st.slider(
            "Product Rating",
            min_value=0.0,
            max_value=5.0,
            value=3.5,
            step=0.1
        )


    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    with col2:

        input_purchases = st.number_input(
            "Past Purchase Count",
            min_value=0,
            max_value=500,
            value=10
        )

        input_return_rate = st.slider(
            "Past Return Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01
        )

        input_delay = st.number_input(
            "Delivery Delay (Days)",
            min_value=0.0,
            max_value=30.0,
            value=1.0,
            step=0.5
        )

        input_session = st.number_input(
            "Session Length (Minutes)",
            min_value=0.0,
            max_value=1000.0,
            value=30.0
        )


    # -----------------------------------------------------
    # BEHAVIOR + CATEGORIES
    # -----------------------------------------------------

    with col3:

        input_views = st.number_input(
            "Number of Product Views",
            min_value=0,
            max_value=1000,
            value=10
        )

        input_device = st.selectbox(
            "Device Type",
            [
                "mobile",
                "desktop",
                "tablet"
            ]
        )

        input_category = st.selectbox(
            "Product Category",
            [
                "electronics",
                "clothing",
                "sports",
                "toys"
            ]
        )

        input_shipping = st.selectbox(
            "Shipping Method",
            [
                "standard",
                "express"
            ]
        )

        input_payment = st.selectbox(
            "Payment Method",
            [
                "credit_card",
                "paypal",
                "apple_pay"
            ]
        )

        input_coupon = st.selectbox(
            "Used Coupon?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


    # =====================================================
    # CREATE TRANSACTION
    # =====================================================

    input_data = pd.DataFrame(
        [{
            "customer_age": float(input_age),
            "product_price": float(input_price),
            "discount_percent": float(input_discount),
            "product_rating": float(input_rating),
            "past_purchase_count": float(input_purchases),
            "past_return_rate": float(input_return_rate),
            "delivery_delay_days": float(input_delay),
            "session_length_minutes": float(input_session),
            "num_product_views": float(input_views),
            "device_type": input_device,
            "product_category": input_category,
            "shipping_method": input_shipping,
            "payment_method": input_payment,
            "used_coupon": float(input_coupon)
        }]
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        risk_score = float(
            model.predict_proba(
                input_data
            )[0, 1]
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

        risk_score = 0.0


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🎯 Real-Time Risk Decision"
    )


    result_left, result_right = st.columns(
        [1, 2]
    )


    with result_left:

        st.metric(
            "Return Risk Score",
            f"{risk_score:.1%}"
        )

        st.metric(
            "Decision Threshold",
            f"{selected_threshold:.1%}"
        )


        if risk_score >= selected_threshold:

            st.error(
                "🚨 HIGH RETURN RISK"
            )

            st.write(
                "**Recommended defensive action:**"
            )

            st.write(
                "Step-up verification / "
                "return-policy warning"
            )

        else:

            st.success(
                "✅ LOW RETURN RISK"
            )

            st.write(
                "**Recommended action:**"
            )

            st.write(
                "Standard frictionless checkout"
            )


    # =====================================================
    # GAUGE
    # =====================================================

    with result_right:

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                domain={
                    "x": [0, 1],
                    "y": [0, 1]
                },
                title={
                    "text": (
                        "Return Risk Score "
                        "vs Decision Threshold"
                    )
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": (
                            "darkred"
                            if risk_score >= selected_threshold
                            else "darkgreen"
                        )
                    },
                    "steps": [
                        {
                            "range": [
                                0,
                                selected_threshold * 100
                            ],
                            "color": "lightgreen"
                        },
                        {
                            "range": [
                                selected_threshold * 100,
                                100
                            ],
                            "color": "pink"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "red",
                            "width": 4
                        },
                        "thickness": 0.75,
                        "value": (
                            selected_threshold * 100
                        )
                    }
                }
            )
        )

        fig_gauge.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=50,
                b=10
            )
        )

        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🛡️ Defense-only AI Risk Management | "
    "XGBoost + Behavioral Features + "
    "Cost-Sensitive Threshold Optimization"
)