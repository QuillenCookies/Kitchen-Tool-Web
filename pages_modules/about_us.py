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
        st.write(""" Hi, I'm Chau Minh Quan (Qan), a second-year undergraduate student and
          intern at the Vietnamese-German University. I've been passionate about
          coding for over 8 years, starting with Java, PHP, Python, and
          JavaScript, and now focusing on TypeScript and Go.""")

    with col2:
        st.subheader("Hồng Nguyên Phúc")
        st.caption("Frontend Developer")
        st.write("""Hi, I'm Hong Nguyen Phuc, a second-year undergraduate student. I love
          to play games, basketball and listen to music. I am a member of the
          Gulag team, I am responsible for the ... development of the
          project.""")

    st.markdown("---")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Hồ Nguyễn Phú")
        st.caption("Backend & Database")
        st.write("""I am a VGU sophomore. My dream is to be a great AI scientist and
          inventor. Dr. Heinz Doofenschmirtz is my role model. In this project,
          I take responsibility in back-end, database management and real-time
          effects.""")

    with col4:
        st.subheader("Cao Tuệ Anh")
        st.caption("Data Engineering Interest")
        st.write("""I’m currently in my first year at VGU, where I’m pursuing my studies
          with a strong interest in the field of data engineering. In this
          project, I took part in UX/UI design.""")

    st.markdown("---")

    # Row 3
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Phạm Trọng Quý")
        st.caption("Frontend & Data Science Enthusiast")
        st.write("""Hi, I'm Pham Trong Quy, a second-year undergraduate student. My dream
          is to be a Data scientist and singer. I love to play games and listen
          to music. I am a member of the Gulag team, and I am responsible for
          the front-end development of the project.""")
    
    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")