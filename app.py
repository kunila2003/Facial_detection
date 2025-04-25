# Import necessary libraries
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os

# Set the page configuration
st.set_page_config(page_title="Live Attendance Viewer", layout="centered")

# Title for the web app
st.title("📋 Live Attendance Dashboard")

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Path to today's attendance file
file_path = f"Attendance/Attendance_{date}.csv"

# Show current date and time
st.markdown(f"**📅 Date:** {date}")
st.markdown(f"**⏰ Time:** {timestamp}")

# Try to read the CSV and display it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    # Show number of records
    st.success(f"✅ Total Records Found: {len(df)}")

    # Display DataFrame with highlights
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
else:
    st.warning(f"⚠️ No attendance record found for {date}")

# Add a refresh button (manual)
if st.button("🔄 Refresh Now"):
    st.rerun()  # ✅ This works in Streamlit v1.30+

# Auto-refresh every 5 seconds
time.sleep(5)
st.rerun()
