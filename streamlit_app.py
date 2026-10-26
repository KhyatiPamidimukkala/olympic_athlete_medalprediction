"""
Streamlit entrypoint for Streamlit Community Cloud.
Redirects execution to app.py so both 'app.py' and 'streamlit_app.py' work as entry points.
"""
import os
import runpy

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
runpy.run_path(app_path, run_name="__main__")
