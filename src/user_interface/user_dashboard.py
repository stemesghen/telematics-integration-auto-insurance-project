#!/usr/bin/env python3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UBI Telematics Demo", layout="wide")

@st.cache_data
def load_df():
    pp = pd.read_csv("data/pricing_preview.csv", parse_dates=["period_start"])
    dp = pd.read_csv("data/driver_period.csv", parse_dates=["period_start"])
    return pp, dp

st.title("Usage-Based Pricing Preview")
pp, dp = load_df()

colA, colB, colC = st.columns(3)
colA.metric("Policies", f"{pp['policy_id'].nunique()}")
colB.metric("Avg factor", f"{pp['telematics_factor'].mean():.3f}")
colC.metric("Avg risk_p", f"{pp['risk_p'].mean():.3f}")

policy = st.text_input("Lookup policy_id")
if policy:
    rowp = pp[pp.policy_id == policy]
    if not rowp.empty:
        st.subheader(f"Policy {policy}")
        st.write(rowp)
        st.write("Latest features")
        latest = dp[dp.policy_id==policy].sort_values("period_start").tail(1)
        st.write(latest.drop(columns=["policy_id"]))
    else:
        st.info("No matching policy in pricing_preview.csv")

st.subheader("Leaderboard (by factor)")
st.dataframe(pp.sort_values("telematics_factor", ascending=False).head(25), use_container_width=True)

