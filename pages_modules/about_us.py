import streamlit as st
import json
from pathlib import Path

def app():
    # Load data from JSON file
    json_file_path = Path(__file__).parent / "about_us.json"
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)["about_us"]

    st.title("👥 The Team")
    st.markdown("### CSE Students @ Vietnamese-German University")
    
    st.info("We’re five CSE students brought together by chance on the very first day at VGU. We’ve collaborated on numerous school projects and share a passion for tech.")

    st.markdown("---")

    # Display team members in rows of two
    for i in range(0, len(data), 2):
        cols = st.columns(2)
        for col, member in zip(cols, data[i:i+2]):
            with col:
                st.subheader(member["name"])
                st.caption(member["job_title"])
                st.write(member["description"])

    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")