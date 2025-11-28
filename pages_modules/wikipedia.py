import streamlit as st

def app():
    # --- CHÈN CSS ĐỂ TĂNG KÍCH THƯỚC TAB ---
    st.markdown("""
    <style>
        /* 1. Tác động vào khung chứa các Tab */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important; /* Bắt buộc chiếm hết chiều ngang */
            gap: 8px; /* Khoảng cách giữa các nút */
            
            /* Khoảng đệm để hiệu ứng nút bay lên không bị mất ngọn */
            padding-top: 10px !important;
            padding-bottom: 5px !important;
        }

        /* 2. Tác động vào từng nút Tab */
        button[data-baseweb="tab"] {
            /* THẦN CHÚ: flex: 1 giúp các nút tự chia nhau khoảng trống để lấp đầy dòng */
            flex: 1 !important; 
            
            background-color: #f0f2f6; 
            border-radius: 20px !important; 
            border: 1px solid #e0e0e0 !important; 
            
            /* Canh chỉnh chữ */
            font-size: 16px !important; 
            font-weight: 700 !important;    
            color: #666 !important;         
            
            /* Giảm padding ngang một chút để đủ chỗ cho nhiều tab */
            padding: 8px 5px !important; 
            margin: 0 !important;
            
            white-space: nowrap !important; /* Giữ chữ trên 1 dòng, không bị xuống dòng */
            transition: all 0.2s ease;
        }

        /* 3. Hiệu ứng Hover (Giờ sẽ không bị cắt nữa) */
        button[data-baseweb="tab"]:hover {
            background-color: #e0e2e6;
            transform: translateY(-3px); /* Bay cao hơn xíu cho đẹp */
            color: #333 !important;
            border-color: #aaa !important;
        }
        
        /* 4. Tab đang chọn */
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #FF4B4B !important; 
            color: white !important;              
            border-color: #FF4B4B !important;
            box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4); /* Bóng đổ mềm mại hơn */
            transform: translateY(-2px); /* Tab đang chọn cũng nổi lên chút */
        }
        
        /* 5. Ẩn gạch chân */
        div[data-baseweb="tab-highlight"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # ---------------------------------------

    # 1. Create three columns to center the content
    # [1, 4, 1] ratio keeps the content in the middle 60% of the screen
    left_spacer, content, right_spacer = st.columns([1, 4, 1])

    # EVERYTHING must be indented inside this 'with' block
    with content:
        st.title("📚 Wiki Knowledge Base")

        # 2. Define Data
        wiki_data = {
            ### Chopsticks
            "bottle": {
                "content": """
            **Chopsticks** are shaped pairs of equal-length sticks that have been used as kitchen and eating utensils in most countries of Sinosphere for over three millennia. They are held in the dominant hand, secured by fingers, and wielded as extensions of the hand, to pick up food.
            
            Originating in **China**, chopsticks later spread to other parts of Sinosphere such as Korea, Japan and Vietnam. Chopsticks have become more accepted in connection with East Asian food in the West, especially in cities with significant East Asian diaspora communities. The use of chopsticks has also spread to the Southeast Asia either via the Chinese diaspora or through some dishes such as noodles that may require chopsticks.

            #### History
            **Chopsticks** have been around and used since at least the Shang dynasty (1766–1122 BCE). However, the Han dynasty historian Sima Qian wrote that it is likely that chopsticks were also used in the preceding Xia dynasty and even the earlier Erlitou culture, although finding archeological evidence from this era is difficult.
            """,
                "video": "https://www.youtube.com/watch?v=xFRzzSF_6gk",
                "caption": "How to use Chopsticks: https://www.youtube.com/watch?v=xFRzzSF_6gk"
                },
            
            ### Fork
            "person": {
                "content": """
            In cutlery or kitchenware, a **fork** (from Latin: furca 'pitchfork') is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines with which one can spear foods either to hold them to cut with a knife or to lift them to the mouth.
            
            #### History
            The **fork** originated as a two-pronged serving tool in ancient Egypt, Greece, and Rome, became a personal eating utensil in Persia and Byzantium, spread slowly to Europe—especially Italy—in the Middle Ages, and became common across Europe and North America by the 19th century as manufacturing improved.
            """,
                "video": "https://www.youtube.com/watch?v=Sq-AhuwJCIY",
                "caption": "How to use a Fork: https://www.youtube.com/watch?v=Sq-AhuwJCIY"
            },

            ### Butter knife
            "chair": {
                "content": """
            A **butter knife** is a small table knife designed specifically for spreading butter, jam, cream cheese, or other soft spreads.
            
            #### History
            The **butter knife** originated in 17th-century Europe when dining etiquette began discouraging sharp knives at the table, leading to the creation of blunt utensils for spreading softened butter. By the 18th and 19th centuries—especially during the Victorian era—it became a formalized piece of tableware, often made of silver and included in elaborate place settings, with both individual and communal versions. With industrial manufacturing in the 19th and 20th centuries, butter knives became affordable and common, evolving into the simple stainless-steel spreaders widely used today.
            """,
                "video": "https://www.youtube.com/watch?v=YrHpeEwk_-U",
                "caption": "How to use butter knife: https://www.youtube.com/watch?v=YrHpeEwk_-U"
            },

            ### Knife (Dao bep)
            "Knife (Dao bep)": {
                "content": """
            A **kitchen knife** is a tool used in food preparation, designed with a sharp blade and a handle to cut, slice, chop, or mince ingredients such as vegetables, meat, fruits, and herbs. It comes in many types (chef’s knife, paring knife, cleaver, etc.), each made for specific cooking tasks.
            
            #### History
            The **kitchen knife** has its origins in prehistoric times, when early humans used sharpened stones and bones to cut food. With the discovery of metalworking around 6000–3000 BCE, knives began to be made from copper, then bronze, and later from iron and steel, making them stronger and sharper. In ancient civilizations such as Egypt, China, and Rome, knives became essential cooking tools and were crafted in different shapes for specific tasks. During the Middle Ages, blacksmiths refined forging techniques, leading to harder, more durable steel blades. By the 18th and 19th centuries, regions like Solingen (Germany) and Sakai (Japan) became famous for high-quality kitchen knives. Modern kitchen knives now use advanced stainless steel, carbon steel, and composite materials, evolving into the specialized chef’s knives, paring knives, and cleavers used in kitchens around the world today.
            """,
                "video": "https://www.youtube.com/watch?v=20gwf7YttQM",
                "caption": "How to use Knife (Dao bep): https://www.youtube.com/watch?v=20gwf7YttQM"
            },

            ### Knife (Dao got)
            "Knife (Dao Got)": {
                "content": """
            A **paring knife (dao gọt)** is a small, lightweight kitchen knife designed for tasks that require precision and control. It typically has a short, sharp blade—usually 2.5 to 4 inches long—making it ideal for peeling fruits and vegetables, removing seeds, trimming stems, and performing delicate cuts that are difficult with larger knives. Because of its maneuverability, the paring knife is essential for fine kitchen work such as shaping, carving small details, or handling food directly in the hand rather than on a cutting board. It is one of the most versatile tools in food preparation and a staple in both home and professional kitchens.
            
            #### History
            The **paring knife (dao gọt)** originated from small utility knives used in ancient times for peeling and trimming food. As metalworking improved through the Bronze and Iron Ages, these small blades became sharper and more refined. By the Middle Ages, they evolved into dedicated kitchen tools, and by the 18th–19th centuries they were standardized by European cutlery makers. Today, the paring knife is an essential tool for precise, detailed kitchen work.
            """,
                "video": "https://www.youtube.com/watch?v=aoqVGdmVlKk",
                "caption": "How to use Knife (Dao got): https://www.youtube.com/watch?v=aoqVGdmVlKk"
            },

            ### Pot
            "Pot": {
                "content": """
            A **pot** is a deep, usually round cooking vessel with high sides and a flat or slightly rounded bottom, designed to hold liquids and ingredients for cooking. Made from materials like stainless steel, aluminum, cast iron, or nonstick coatings, pots are used on stoves, ovens, or open fires to boil, simmer, stew, or braise food. They often come with a lid to trap heat and moisture, helping cook food evenly and retain flavor. Pots vary in size and type, including stockpots, saucepans, and Dutch ovens, making them versatile tools in both home and professional kitchens.
            
            #### History
            Cooking pots are among the oldest kitchen tools in human history. Early humans used hollowed-out stones, shells, or clay vessels over open fires to cook food. Around 10,000 years ago, during the Neolithic period, people began making pottery pots, which allowed them to boil grains, soups, and stews. With the Bronze and Iron Ages, metal pots made of bronze, copper, and later iron became common, offering greater durability and heat control. By the Middle Ages, pots were a staple in European kitchens, often heavy and designed for open hearth cooking. The Industrial Revolution in the 18th–19th centuries led to mass-produced metal pots, making them affordable and widespread. Today, pots come in a variety of materials like stainless steel, aluminum, and nonstick surfaces, remaining essential for boiling, simmering, and stewing in kitchens worldwide.
            """,
                "video": "https://www.youtube.com/watch?v=ZXAd9BYDPJQ",
                "caption": "How to use Pot: https://www.youtube.com/watch?v=ZXAd9BYDPJQ"
            },

            ### Plate
            "Plate": {
                "content": """
            A **plate** is a flat, usually round dish used for serving or eating food. It can be made from materials like ceramic, glass, metal, or plastic and is a common item in both everyday dining and formal table settings.
            
            #### History
            **Plates** have been used for thousands of years as vessels for serving and eating food. Early plates were made from wood, stone, or clay in ancient civilizations such as Egypt, Greece, and China. By the Middle Ages, metal plates made of pewter or silver were common among the wealthy, while ordinary people often used wooden or ceramic dishes. In the 16th–18th centuries, fine porcelain plates from China and later Europe became popular for their beauty and durability, especially among the aristocracy. The Industrial Revolution made mass-produced ceramic and metal plates affordable for everyday use, and today, plates come in a wide variety of materials, shapes, and designs for both practical and decorative purposes.
            """,
                "video": "https://www.youtube.com/watch?v=-iZGoUNUwbc",
                "caption": "How to use Plate: https://www.youtube.com/watch?v=-iZGoUNUwbc"
            },

            ### Spoon
            "Spoon": {
                "content": """
            A **metal spoon** is a utensil made of metal, typically used for eating, stirring, or serving food and liquids. It usually has a shallow bowl at one end and a handle at the other, and can be made from materials like stainless steel, iron, or aluminum, making it durable and reusable.
            
            #### History
            **Spoons** are among the oldest eating utensils, with early versions made from wood, bone, or shells. The first metal spoons appeared in ancient civilizations such as Egypt, Greece, and Rome, where they were crafted from bronze, silver, or gold and often decorated with engravings. During the Middle Ages in Europe, pewter and silver spoons became common among the wealthy, while ordinary people still used wooden or horn spoons. The Industrial Revolution in the 18th–19th centuries allowed mass production of durable stainless steel spoons, making them affordable and widespread. Today, metal spoons are standard in households worldwide for eating, cooking, and serving.
            """,
                "video": "https://www.youtube.com/watch?v=U0Tp8-NjsvM",
                "caption": "How to use Spoon: https://www.youtube.com/watch?v=U0Tp8-NjsvM"
            }
        }

        tab_names = list(wiki_data.keys())

        # 3. Handle URL Navigation (?tab=...)
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
            
            # CREATE BUTTONS
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Button 1: Return to Wiki
                if st.button("⬅️ Return to Full Wiki", use_container_width=True):
                    st.query_params.clear()
                    st.rerun()
            
            with col2:
                # Button 2: Return to Home to recognize
                if st.button("📹 Continue Detection", type="primary", use_container_width=True):
                    # 1. Delect URL
                    st.query_params.clear()
                    
                    # 2. Update Session State for Menu.py tab 0 (Home)
                    st.session_state["selected_index"] = 0
                    st.session_state["main_menu_selected"] = "Home" # For sure
                    
                    # 3. Turn camera automatically
                    st.session_state["auto_start_trigger"] = True 
                    
                    # 4. Reload
                    st.rerun()

        else:
            # 4. Default View (Show All Tabs)
            st.write("### Information")
            
            # Create the tabs
            tabs = st.tabs(tab_names)

            # Fill tabs with content
            for i, name in enumerate(tab_names):
                with tabs[i]:
                    st.subheader(name)
                    
                    # IMPROVEMENT: Split text and video side-by-side inside the tab
                    # This makes it look much cleaner!
                    c1, c2 = st.columns([0.6, 0.4]) 
                    
                    with c1:
                        st.write(wiki_data[name]["content"])
                    
                    with c2:
                        st.video(wiki_data[name]["video"])
                        st.caption(wiki_data[name].get("caption", ""))
    
    # Footer (Full Width)
    st.markdown("---")
    st.caption("Powered by 5 anh em siu nhan.")