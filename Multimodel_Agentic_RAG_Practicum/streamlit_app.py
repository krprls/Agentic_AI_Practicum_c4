"""Alias entrypoint so both `streamlit run app.py` and `streamlit run streamlit_app.py` work."""
from app import main

if __name__ == "__main__":
    main()

