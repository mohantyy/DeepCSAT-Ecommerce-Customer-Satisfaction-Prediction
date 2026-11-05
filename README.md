# 🎯 DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

> **End-to-End Deep Learning (Artificial Neural Network) & NLP Pipeline for Predicting Real-Time Customer Satisfaction (CSAT) Scores on E-Commerce Support Interactions (`Shopzilla`)**
>
> **Author:** Abhilash Mohanty (`@mohantyy`)
>
> **Program:** AlmaBetter Masters in Data Science — AI Engineering v1 (Capstone Project: Deep Learning)

## 📌 1. Executive Summary & Business Problem

In high-volume e-commerce platforms like **Shopzilla**, post-interaction Customer Satisfaction (CSAT) surveys suffer from low response rates (<15%) and delayed feedback, preventing support operations from intervening before dissatisfied customers churn.

**DeepCSAT** solves this operational gap by predicting **CSAT Scores (1 to 5)** and **Dissatisfaction Risk** in real time at the moment a support interaction concludes. By combining structured operational telemetry (response time latency, connected handling duration, support channel, product category, item price, agent tenure, shift timing) with Natural Language Processing (NLP sentiment polarity & TF-IDF embeddings) on unstructured customer remarks, our 3-layer regularized **Artificial Neural Network (ANN)** empowers customer service teams to trigger immediate supervisor escalations and proactive retention workflows.

## 📊 2. Dataset Overview (`Shopzilla` Customer Support Dataset — 20 Columns)

| # | Column Name | Type | Description |
| :-: | :--- | :--- | :--- |
| 1 | `Unique id` | UUID String | Unique identifier for each support interaction record |
| 2 | `channel_name` | Categorical | Customer service channel (`Inbound`, `Outcall`, `Email`) |
| 3 | `category` | Categorical | High-level interaction category (`Returns`, `Order Related`, `Refund Related`, `Product Queries`, etc.) |
| 4 | `Sub-category` | Categorical | Granular ticket sub-category (`Reverse Pickup Enquiry`, `Order status enquiry`, `Delayed`, etc.) |
| 5 | `Customer Remarks` | Text / String | Qualitative feedback text provided by the customer |
| 6 | `Order_id` | UUID String | Associated Shopzilla order identifier |
| 7 | `order_date_time` | Datetime | Timestamp when the customer placed the order |
| 8 | `Issue_reported at` | Datetime | Timestamp when the customer reported the support issue |
| 9 | `issue_responded` | Datetime | Timestamp when the support agent responded to the issue |
| 10 | `Survey_response_Date` | Date | Date when the customer completed the survey |
| 11 | `Customer_City` | Categorical | Geographic city of the customer |
| 12 | `Product_category` | Categorical | Product line (`Electronics`, `LifeStyle`, `Mobile`, `Home`, `Furniture`, etc.) |
| 13 | `Item_price` | Float (INR) | Monetary price of the ordered item |
| 14 | `connected_handling_time` | Float (Sec) | Total duration in seconds taken to handle the interaction |
| 15 | `Agent_name` | Categorical | Customer service agent identifier |
| 16 | `Supervisor` | Categorical | Team supervisor overseeing the support agent |
| 17 | `Manager` | Categorical | Operations manager overseeing the supervisory pod |
| 18 | `Tenure Bucket` | Ordinal Cat | Agent experience tier (`On Job Training`, `0-30`, `31-60`, `61-90`, `>90` days) |
| 19 | `Agent Shift` | Categorical | Agent shift timing (`Morning`, `Afternoon`, `Evening`, `Split`, `Night`) |
| 20 | `CSAT Score` | Integer (1–5) | **Target Variable** — Customer Satisfaction Score (`1` = Very Dissatisfied to `5` = Very Satisfied) |

## 🏗️ 3. System Architecture & Methodology

1. **Data Integrity & Cleaning (`src/data_preprocessing.py`):** UUID deduplication, multi-format datetime parsing, category-conditioned median imputation with binary missingness indicator flags (`has_customer_remark`, `has_item_price`, `has_handling_time`), and IQR outlier Winsorization with `log1p` transformation.
2. **Feature Engineering & NLP (`src/feature_engineering.py`):** Temporal SLA latency (`response_time_mins` and `sla_breach_flag` for latency > 60 mins), cyclical hour and weekend flags, negation-preserving stopword filtering, lexicon sentiment polarity scoring, L2-normalized TF-IDF vectorization (25 features), Ordinal Tenure encoding, One-Hot categorical encoding, and Z-score Standardization (68 total input dimensions).
3. **Deep Learning ANN Architecture (`src/model_architecture.py`):**
   - `Input(68)` -> `Dense(128, HeNormal, L2=1e-4)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.25)`
   - `Dense(64, HeNormal, L2=1e-4)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.20)`
   - `Dense(32, HeNormal, L2=1e-4)` -> `BatchNormalization` -> `ReLU` -> `Dropout(0.10)`
   - `Output: Dense(5, Softmax)` trained with training-only **SMOTE**, mini-batch **Adam** (`batch_size=128`, `lr=0.004` with step decay), and **EarlyStopping** (`patience=10`).
4. **Real-Time Local Deployment (`app.py` & `src/predict.py`):** Interactive Streamlit web dashboard and CLI inference engine with automated high-risk detractor escalation alerts.

## 📈 4. Statistical Hypothesis Validation & Model Benchmark Results

### Statistical Hypothesis Test Outcomes
1. **Hypothesis 1 — Issue Response Latency vs. CSAT Tier (One-Way ANOVA):** $F = 2495.28, p < 10^{-100}$ (**Reject $H_0$**). Satisfied customers (`CSAT 4–5`) experience a median response time of **22–25 minutes**, whereas dissatisfied customers (`CSAT 1–2`) experience median response latencies exceeding **360 minutes**.
2. **Hypothesis 2 — Support Category Distribution vs. Satisfaction Group (Chi-Square Test):** `Returns` accounts for **51.2%** and `Order Related` accounts for **27.0%** of total ticket volume (**78.2% combined**).
3. **Hypothesis 3 — Connected Call Handling Time Burden (Mann-Whitney U Test):** $U = 2,084,408.0, p = 3.09 \times 10^{-58}$ (**Reject $H_0$**). Dissatisfied interactions exhibit significantly longer connected handling durations (`> 800 seconds`) compared to satisfied interactions (`~280 seconds`).

### Model Evaluation Benchmark Table

| Model Architecture | 5-Class Test Accuracy | Weighted F1-Score | CSAT-1 (Detractor) F1 | CSAT-5 (Promoter) F1 | Binary Satisfaction Accuracy | ROC-AUC Score |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Model 1:** Baseline Shallow MLP (1-Layer) | 71.40% | 0.6620 | 0.8120 | 0.8310 | 91.20% | 0.8950 |
| **Model 2:** 2-Layer ANN + 3-Fold CV | 76.95% | 0.7185 | 0.8840 | 0.8650 | 95.60% | 0.9580 |
| **Model 3: DeepCSAT 3-Layer Regularized ANN (Final)** | **79.01%** | **0.7335** | **0.9231** | **0.8806** | **98.40%** | **0.9806** |

## 💡 5. Key Business Recommendations for Shopzilla

1. **Enforce a 45-Minute First-Response SLA:** Tickets breaching the 60-minute response threshold experience a steep drop in average CSAT from **4.52 to 2.18**. Automated queue alerts at the 30-minute mark prevent SLA breaches.
2. **Automate Reverse-Pickup Tracking for `Returns`:** Because `Returns` (51.2%) and `Order Related` (27.0%) drive **78.2%** of all support traffic, proactive reverse-pickup status updates can deflect up to 30% of inbound inquiries.
3. **Tiered Routing for High-Value Orders & Novices:** Route orders with `Item_price > ₹10,000` and repeat refund escalations directly to tenured agents (`>90` days) rather than `On Job Training` cohorts.
4. **Live Supervisor Whisper Alerts at 8 Minutes Handling Time:** Trigger a live floor-supervisor assist alert whenever `connected_handling_time` exceeds 480 seconds or when real-time `DeepCSAT` dissatisfaction probability exceeds `0.70`.) scores in real time from customer support interactions, feedback, and operational telemetry. Built with TensorFlow/Keras, NumPy, SciPy, and Streamlit (AlmaBetter AI Engineering Capstone).
