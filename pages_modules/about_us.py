import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

def app():
    st.title("👥 The Team")
    st.markdown("### CSE Students @ Vietnamese-German University")
    
    st.info("We’re five CSE students brought together by chance on the very first day at VGU. We’ve collaborated on numerous school projects and share a passion for tech.")

    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Châu Minh Quân")
        st.caption("Backend & TypeScript Enthusiast")
        st.write("First-year undergraduate. Passionate about coding for over 8 years (Java, PHP, Python, Go).")

    with col2:
        st.subheader("Hồng Nguyên Phúc")
        st.caption("Frontend Developer")
        st.write("Member of the Gulag team. Loves basketball, games, and music. Responsible for UI/UX.")

    st.markdown("---")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Hồ Nguyễn Phú")
        st.caption("Backend & Database")
        st.write("Dreaming of being an inventor like Dr. Heinz Doofenschmirtz. Handles real-time effects.")

    with col4:
        st.subheader("Cao Tuệ Anh")
        st.caption("Data Engineering Interest")
        st.write("First-year student with a strong interest in Data Engineering. Designed the user experience.")

    st.markdown("---")

    # Row 3
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Phạm Trọng Quý")
        st.caption("Frontend & Data Science")
        st.write("Dreams of being a Data Scientist and Singer. Loves games and music.")
    
    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by Streamlit, OpenCV, and MobileNet SSD.")