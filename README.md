# 🎈 Uncertainty-aware AI system in ED

This is the prototype for the thesis: Developing an Uncertainty-Aware Ex-plainable Clinical Decision Support System Prototype: A Design Science Approach

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
 

   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

Project structure
uncertaintyproject/
├─ streamlit_app.py                # Landing Page
├─ utils.py                        # shared styles, data, sidebar, header
└─ pages/
   ├─ 1_ED_Track_Board.py
   ├─ 2_Patient_Chart.py
   ├─ 3_Registration.py
   ├─ 4_Visits_and_Complaints.py
   ├─ 5_Clinical_Decision_Log.py
   ├─ 6_Audit_Log.py
