import streamlit as st
from streamlit_option_menu import option_menu
import pages_modules.home as home
import pages_modules.wikipedia as wikipedia
import pages_modules.about_us as about_us

# 1. Page Configuration
st.set_page_config(
    page_title="Object Detection AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stVerticalBlock"] > div:has(div.stOptionMenu) { margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. LOGIC GHI NHỚ VỊ TRÍ (Session State Logic)
# ------------------------------------------------------------------

# Khởi tạo biến ghi nhớ nếu chưa có
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = 0

# A. Kiểm tra nếu có lệnh điều hướng từ URL (Ví dụ: từ Home bấm sang)
query_params = st.query_params
if "nav" in query_params:
    target_nav = query_params["nav"]
    
    if target_nav == "Wiki Search":
        st.session_state["selected_index"] = 1
    elif target_nav == "About Us":
        st.session_state["selected_index"] = 2
    else:
        st.session_state["selected_index"] = 0
        
    # Xóa nav khỏi URL để tránh bị dính chặt vào đó
    del query_params["nav"]

# ------------------------------------------------------------------
# 4. Hiển thị Menu (Dùng biến đã ghi nhớ)
# ------------------------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Wiki Search", "About Us"],
    icons=["camera-video", "book", "people"], 
    menu_icon="cast",
    
    # QUAN TRỌNG: Dùng 'manual_select' để ép Menu theo ý mình
    manual_select=st.session_state["selected_index"],
    
    default_index=0,
    orientation="horizontal",
    key="main_menu",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
        "nav-link-selected": {"background-color": "#FF4B4B"},
    }
)

# B. Cập nhật lại biến ghi nhớ khi người dùng bấm chuột trực tiếp lên Menu
# Map từ tên tab sang số thứ tự
mapping = {"Home": 0, "Wiki Search": 1, "About Us": 2}
if mapping[selected] != st.session_state["selected_index"]:
    st.session_state["selected_index"] = mapping[selected]
    st.rerun() # Tải lại trang để cập nhật trạng thái ngay lập tức

# ------------------------------------------------------------------
# 5. Load Pages
# ------------------------------------------------------------------
if selected == "Home":
    home.app()
elif selected == "Wiki Search":
    wikipedia.app()
elif selected == "About Us":
    about_us.app()