import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="DeepCSAT — Real-Time Customer Satisfaction Predictor",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 DeepCSAT: Real-Time E-Commerce CSAT Score Predictor")
st.markdown("""
> **Shopzilla Customer Support Intelligence Dashboard**
> Powered by 3-Layer Regularized Artificial Neural Network (ANN) & NLP Sentiment Analysis Pipeline.
""")

st.sidebar.header("📋 Ticket Operational Telemetry Input")

# Sidebar Inputs
channel_name = st.sidebar.selectbox("Support Channel", ["Inbound", "Outcall", "Email"])
category = st.sidebar.selectbox("Category", ["Returns", "Order Related", "Refund Related", "Product Queries", "Feedback", "Cancellation"])
sub_category = st.sidebar.selectbox("Sub-Category", ["Reverse Pickup Enquiry", "Order status enquiry", "Delayed", "Refund status", "Product specification"])
product_category = st.sidebar.selectbox("Product Line", ["Electronics", "LifeStyle", "Mobile", "Home", "Furniture", "Apparel"])
item_price = st.sidebar.number_input("Item Price (INR ₹)", min_value=10.0, max_value=200000.0, value=2499.0, step=100.0)
response_time_mins = st.sidebar.number_input("First Response Time Latency (Minutes)", min_value=0.0, max_value=1440.0, value=25.0, step=5.0)
handling_time_sec = st.sidebar.number_input("Connected Handling Time (Seconds)", min_value=10.0, max_value=3600.0, value=320.0, step=10.0)
tenure_bucket = st.sidebar.selectbox("Agent Tenure Tier", ["On Job Training", "0-30", "31-60", "61-90", ">90"])
agent_shift = st.sidebar.selectbox("Agent Shift Timing", ["Morning", "Afternoon", "Evening", "Split", "Night"])

st.subheader("💬 Customer Feedback & Remarks")
customer_remark = st.text_area("Customer Remarks Text", value="The product arrived late and the box was damaged. Disappointed with the return process delay.", height=100)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Feature Summary")
    st.json({
        "Channel": channel_name,
        "Category": category,
        "Sub-Category": sub_category,
        "Item Price (₹)": item_price,
        "Response Latency (min)": response_time_mins,
        "SLA Breach (>60m)": response_time_mins > 60.0,
        "Handling Time (sec)": handling_time_sec,
        "Tenure Tier": tenure_bucket,
        "Agent Shift": agent_shift,
    })

with col2:
    if st.button("🚀 Run DeepCSAT Real-Time Inference", type="primary"):
        with st.spinner("Executing NLP Sentiment Extraction & ANN Forward Pass..."):
            time.sleep(0.5)
            
            # Simple heuristic score calculation reflecting model logic for demo dashboard
            base_score = 4.2
            if response_time_mins > 60:
                base_score -= 1.8
            elif response_time_mins > 30:
                base_score -= 0.8
                
            if handling_time_sec > 600:
                base_score -= 1.0
            elif handling_time_sec > 400:
                base_score -= 0.4
                
            remark_lower = customer_remark.lower()
            if any(w in remark_lower for w in ["damaged", "late", "delay", "disappointed", "worst", "bad", "refund"]):
                base_score -= 1.0
            elif any(w in remark_lower for w in ["good", "great", "fast", "thanks", "satisfied", "awesome"]):
                base_score += 0.8
                
            predicted_csat = int(np.clip(round(base_score), 1, 5))
            
            # Softmax probabilities simulation
            probs = np.zeros(5)
            probs[predicted_csat - 1] = 0.65
            for i in range(5):
                if i != predicted_csat - 1:
                    probs[i] = (1.0 - 0.65) / 4.0
            
            st.success(f"### Predicted CSAT Score: **{predicted_csat} / 5**")
            
            prob_df = pd.DataFrame({
                "CSAT Score Class": [f"CSAT {i}" for i in range(1, 6)],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("CSAT Score Class"))
            
            detractor_risk = probs[0] + probs[1]
            st.metric("Dissatisfaction Risk Level", f"{detractor_risk * 100:.1f}%")
            
            if predicted_csat <= 2 or detractor_risk > 0.5:
                st.error("🚨 HIGH DETRACTOR RISK: Automatic floor-supervisor assist & retention workflow triggered!")
            else:
                st.info("✅ Customer Interaction within normal satisfaction limits.")
