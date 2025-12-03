import json
from pathlib import Path
import streamlit as st

def app():
    # Load Wiki Data from JSON file
    json_file_path = Path(__file__).parent / "wiki_data.json"
    with open(json_file_path, "r", encoding="utf-8") as file:
        wiki_data = json.load(file)["wiki_data"]
    
    # Load CSS File
    css_file_path = Path(__file__).parent / "static" / "wiki.css"
    with open(css_file_path, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

    # ---------------------------------------

    left_spacer, content, right_spacer = st.columns([1, 4, 1])

    with content:
        st.title("📚 Wiki Knowledge Base")

        tab_names = list(wiki_data.keys())
        query_params = st.query_params
        target_tab = query_params.get("tab", None)

        if target_tab and target_tab in tab_names:
            st.info(f"🔍 Showing results for: **{target_tab}**")
            
            # Show content directly
            st.subheader(target_tab)
            st.write(wiki_data[target_tab]["content"])
            st.video(wiki_data[target_tab]["video"])
            st.caption(wiki_data[target_tab].get("caption", ""))
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⬅️ Return to Full Wiki", use_container_width=True):
                    st.query_params.clear()
                    st.rerun()
            
            with col2:
                if st.button("📹 Continue Detection", type="primary", use_container_width=True):
                    st.query_params.clear()
                    st.session_state["selected_index"] = 0
                    st.session_state["main_menu_selected"] = "Home"
                    st.session_state["auto_start_trigger"] = True 
                    st.rerun()
        else:
            st.write("### Information")
            tabs = st.tabs(tab_names)
            for i, name in enumerate(tab_names):
                with tabs[i]:
                    st.subheader(name)
                    c1, c2 = st.columns([0.6, 0.4]) 
                    with c1:
                        st.write(wiki_data[name]["content"])
                    
                    with c2:
                        st.video(wiki_data[name]["video"])
                        st.caption(wiki_data[name].get("caption", ""))
    
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")