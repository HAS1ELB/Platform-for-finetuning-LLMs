import streamlit as st
import requests
import os
import streamlit.components.v1 as components
import pandas as pd
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Simple local token caching to persist logins across Streamlit reloads (local dev only)
TOKEN_CACHE_FILE = os.path.join(os.path.dirname(__file__), '.token')

def save_token_to_file(token: str):
    try:
        with open(TOKEN_CACHE_FILE, 'w') as f:
            f.write(token)
    except Exception:
        pass

def load_token_from_file():
    try:
        if os.path.exists(TOKEN_CACHE_FILE):
            with open(TOKEN_CACHE_FILE, 'r') as f:
                return f.read().strip()
    except Exception:
        pass
    return None

def delete_token_file():
    try:
        if os.path.exists(TOKEN_CACHE_FILE):
            os.remove(TOKEN_CACHE_FILE)
    except Exception:
        pass

st.set_page_config(page_title="Mini Cloud Training", page_icon="☁️", layout="wide")

# Initialize session keys
if "show_register" not in st.session_state:
    st.session_state["show_register"] = False
if "token" not in st.session_state:
    # Try to load a cached token to persist login across reloads
    st.session_state["token"] = load_token_from_file()

# If a token is present, hide the login/register form immediately (covers race conditions)
if st.session_state.get("token"):
    st.markdown("<style>.form-card{display:none !important;}</style>", unsafe_allow_html=True)

# Small custom CSS for compact card layout for login/register forms
st.markdown(
    """
    <style>
    .form-card { max-width: 420px; margin: 4px auto; padding: 18px 18px 12px; border-radius: 12px; background-color: transparent; border: none; }
    /* Ensure consistent spacing between title and form */
    .form-title { text-align:center; margin: 6px 0 12px; font-weight:600; }
    .small-caption { font-size:14px; color:var(--secondary-text-color, #B0B0B0); text-align:right; }
    .inline-row{ display:flex; align-items:center; justify-content:flex-end; }
    .link-style { background: none; border: none; color: var(--primary-color, #1f77b4); text-decoration: underline; cursor: pointer; padding:0; }
    .header-bar { display:flex; justify-content:space-between; align-items:center; width:100%; padding: 8px 12px; }
    .username-button { border-radius: 10px; padding:6px 10px; background: rgba(255,255,255,0.02); color: var(--secondary-text-color, #B0B0B0); border: none; cursor: pointer; white-space:nowrap; }
    .header-bar button { white-space:nowrap; }
    .header-bar h1 { margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .section-card { padding:12px; border-radius:10px; background-color: rgba(255,255,255,0.01); margin-bottom: 10px; }
    /* Remove focus/active shadow on Streamlit buttons to avoid the ghost/shadow artifact when clicking */
    .stButton>button:focus, .stButton>button:active, .stButton>button:focus-visible, .stButton>button:active::after, .stButton>button::after, .stButton>button::before {
        box-shadow: none !important;
        outline: none !important;
        -webkit-box-shadow: none !important;
        -moz-box-shadow: none !important;
    }
    /* Broad override - some Streamlit versions use different class names */
    button:focus, button:active, button:focus-visible, input[type='button']:focus, input[type='button']:active {
        outline: none !important;
        box-shadow: none !important;
        -webkit-box-shadow: none !important;
    }
    /* Remove shadows from disabled buttons and any aria-disabled attribute variants */
    button[disabled], button[disabled]:before, button[disabled]:after, [aria-disabled='true'], [aria-disabled='true']::before, [aria-disabled='true']::after {
        box-shadow: none !important;
        outline: none !important;
        opacity: 0.6 !important;
        -webkit-box-shadow: none !important;
    }
    /* Prevent tap highlight on mobile devices for consistency */
    .stButton>button, button { -webkit-tap-highlight-color: transparent; }
    /* Remove inner highlight for Firefox */
    button::-moz-focus-inner { border: 0 !important; }
    /* Remove transitions and visual filters on buttons to prevent ghosting overlays during re-render */
    .stButton>button, button { transition: none !important; -webkit-transition: none !important; }
    .stButton>button[disabled], button[disabled] { filter: none !important; backdrop-filter: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

def check_backend_connection():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


def get_user_info():
    """Return current logged-in user's info (username) or None"""
    if not st.session_state.get('token'):
        return None
    if st.session_state.get('username'):
        return st.session_state.get('username')
    try:
        response = requests.get(f"{API_URL}/session", headers=get_headers(), timeout=6)
        if response.status_code == 200:
            data = response.json()
            st.session_state['username'] = data.get('username')
            return st.session_state['username']
    except Exception:
        return None
    return None

def get_train_env():
    """Return dict with trainer availability for the current backend environment."""
    try:
        response = requests.get(f"{API_URL}/train_env", headers=get_headers(), timeout=4)
        if response.status_code == 200:
            data = response.json()
            st.session_state['trainer_available'] = data.get('trainer_available', False)
            st.session_state['trainer_message'] = data.get('message')
            return data
    except Exception:
        # Cannot reach backend training info; assume false
        st.session_state['trainer_available'] = False
    st.session_state['trainer_available'] = False
    st.session_state['trainer_message'] = 'Could not reach backend to detect Trainer availability.'
    return {"trainer_available": False, "message": st.session_state['trainer_message']}

def login_page():
    # Center the top-level title for the login page
    st.markdown("<h1 style='text-align: center; margin-bottom: 8px;'>☁️ Welcome to Mini Cloud Training Platform</h1>", unsafe_allow_html=True)
    
    # Check backend status
    if not check_backend_connection():
        st.error("⚠️ Backend server is not running!")
        st.info("Please start the backend server first:")
        st.code("cd backend\npy -m uvicorn main:app --reload", language="bash")
        st.stop()
    
    # Use a centered subheader inside the form area

    # If user wants to register, show the register page and return early
    if st.session_state.get("show_register"):
        register_page()
        return
    # Center the login form and limit width for better UX
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # No placeholder image; small top spacer keeps the layout compact
        st.write("")
        # Create a container with a max-width for the form using inline CSS
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)

        # Centered subheader above the login form
        st.markdown("<h3 class='form-title'>Login to Your Account</h3>", unsafe_allow_html=True)

        # Login form
        with st.form("login_form"):
            st.text_input("Username", key="login_username", placeholder="Enter your username")
            st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                username = st.session_state.login_username
                password = st.session_state.login_password
                with st.spinner("Logging in..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/token",
                            data={"username": username, "password": password},
                            timeout=8
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.token = data["access_token"]
                            # Persist token to a local file so reloading Streamlit won't log us out.
                            save_token_to_file(st.session_state.token)
                            # Set a browser cookie (for direct browser clients)
                            # Instruct the backend to set a httpOnly cookie using the token we just received.
                            # This avoids sharing the raw token with JavaScript; it's a local development convenience.
                            components.html(
                                f"<script>fetch('{API_URL}/session/set', {{ method: 'POST', credentials: 'include', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{'access_token':'{st.session_state.token}','token_type':'bearer'}}) }}).then(()=>{{console.log('cookie set')}}).catch(()=>{{}});</script>",
                                height=0,
                            )
                            st.success("Login successful! Redirecting...")
                            st.rerun()
                        else:
                            st.error("Invalid username or password. Please try again.")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Please try again.")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        # Close the CSS wrapper
        st.markdown("</div>", unsafe_allow_html=True)

        # Inline phrase and Register button on the same line (right aligned phrase + button)
        inline_left, inline_right = st.columns([4, 1])
        with inline_left:
            st.markdown("<p style='text-align:right; margin:6px 0 4px 0;'>Don't have an account?</p>", unsafe_allow_html=True)
        with inline_right:
            if st.button("Register"):
                st.session_state.show_register = True
                st.rerun()

    # Show registration page if requested
    if st.session_state.get("show_register"):
        register_page()

def register_page():
    # Removed the large heading to keep the register page compact
    # Center the registration form for better UX
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        # Put the register form title inside the card so it matches the login layout
        st.markdown("<h3 class='form-title'>Register to Get Started</h3>", unsafe_allow_html=True)
        with st.form("register_form"):
            st.text_input("Choose a Username", key="register_username", placeholder="e.g., johndoe")
            st.text_input("Choose a Password", type="password", key="register_password", placeholder="Enter password")
            st.text_input("Confirm Password", type="password", key="register_password_confirm", placeholder="Confirm password")
            submit_button = st.form_submit_button("Register")

            if submit_button:
                username = st.session_state.register_username
                password = st.session_state.register_password
                confirm = st.session_state.register_password_confirm
                valid = True
                if password != confirm:
                    st.error("Passwords do not match. Please try again.")
                    valid = False
                elif not username or not password:
                    st.error("Please fill in both username and password.")
                    valid = False

                if valid:
                    with st.spinner("Creating account..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/register",
                                json={"username": username, "password": password},
                                timeout=10
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.token = data["access_token"]
                                save_token_to_file(st.session_state.token)
                                components.html(
                                    f"<script>fetch('{API_URL}/session/set', {{ method: 'POST', credentials: 'include', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify({{'access_token':'{st.session_state.token}','token_type':'bearer'}}) }}).then(()=>{{console.log('cookie set')}}).catch(()=>{{}});</script>",
                                    height=0,
                                )
                                st.success("Registration successful! Redirecting...")
                                st.rerun()
                            else:
                                st.error("Registration failed. Username may already exist.")
                        except requests.exceptions.Timeout:
                            st.error("Request timed out. Please try again.")
                        except Exception as e:
                            st.error(f"Connection error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Back to Login"):
            st.session_state.show_register = False
            st.rerun()

def main_page():
    # Header: title left, logout button on the far right in a single line
    st.markdown("<div class='header-bar'>", unsafe_allow_html=True)
    # Use columns to place the title and logout button inline; give more width to the button so it won't wrap
    cols = st.columns([10, 2])
    with cols[0]:
        st.markdown(
            "<h1 style='margin:0; padding:0; font-size:32px;'>☁️ Mini Cloud Training Platform</h1>",
            unsafe_allow_html=True,
        )
    # Place logout in the far right column for maximum right alignment
    with cols[1]:
        username = get_user_info()
        # Show username button which acts as logout when clicked
        if username:
            # Use a pill button with 'Logout from {username}' label and prevent wrapping
            # Use non-breaking spaces to keep label on one line
            label = f"Logout\u00A0from\u00A0{username}"
            if st.button(label, key="logout-username"):
                # Clear session keys for a clean logout
                st.session_state["token"] = None
                st.session_state["show_register"] = False
                if 'username' in st.session_state:
                    del st.session_state['username']
                # Reset UI pagination/caches
                for key in ['models', 'datasets', 'model_page', 'dataset_page']:
                    if key in st.session_state:
                        del st.session_state[key]
                # Remove cookie from browser; make a best effort using a JS snippet
                # Make a best-effort to remove the httpOnly cookie by asking the backend to clear it,
                # also remove local cached token file to avoid reusing it on reload.
                try:
                    requests.post(f"{API_URL}/logout", timeout=4)
                except Exception:
                    pass
                delete_token_file()
                components.html("<script>document.cookie='auth_token=; Max-Age=0; path=/;';</script>", height=0)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    # Hide login/register form card if user is authenticated to prevent it appearing in the main layout
    if st.session_state.get("token"):
        st.markdown("<style>.form-card{display:none !important;}</style>", unsafe_allow_html=True)
    
    # Query backend to see if the training environment is ready
    if 'trainer_available' not in st.session_state:
        get_train_env()

    tab1, tab2, tab3, tab4 = st.tabs(["🤗 Models & Datasets", "📊 Datasets", "🚀 Training", "📈 Experiments"])
    
    with tab1:
        # Helper to blur focused element after a click to remove focus rings/shadows
        def _blur_active_el():
            try:
                components.html("<script>try{window.parent.document.activeElement.blur();}catch(e){};</script>", height=0)
            except Exception:
                pass
        st.markdown("<div class='section-card'><h2>🤗 Models & Datasets</h2></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Browse Models")
            model_search = st.text_input("Search models", value="gpt2")
            if st.button("Search Models"):
                with st.spinner("Fetching models..."):
                    try:
                        response = requests.get(f"{API_URL}/models?search={model_search}&limit=100", headers=get_headers(), timeout=10)
                        if response.status_code == 200:
                            models = response.json()
                            st.session_state.models = models
                            st.session_state.model_page = 0
                        else:
                            st.error("Failed to fetch models")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Table view with pagination (10 rows per page)
            if "models" in st.session_state:
                models = st.session_state.models or []
                if models:
                    page_size = 10
                    page = st.session_state.get('model_page', 0)
                    total = len(models)
                    start = page * page_size
                    end = min(start + page_size, total)
                    page_items = models[start:end]

                    # Build a simple dataframe for the table view
                    df_models = []
                    for m in page_items:
                        df_models.append({
                            'Name': m.get('id'),
                            'Downloads': m.get('downloads') or 'N/A'
                        })
                    try:
                        st.table(df_models)
                    except Exception as e:
                        # Fallback if pyarrow/arrow-related import fails - render simple markdown list
                        st.warning("Table rendering failed due to a dependency issue; showing a simplified list.")
                        for r in df_models:
                            st.markdown(f"**{r['Name']}** — Downloads: {r['Downloads']}")

                    # Pagination controls
                    col_prev, col_info, col_next = st.columns([1, 6, 1])
                    # Dataset pagination handlers
                    def datasets_prev():
                        st.session_state.dataset_page = max(0, st.session_state.get('dataset_page', 0) - 1)
                        _blur_active_el()

                    def datasets_next():
                        st.session_state.dataset_page = st.session_state.get('dataset_page', 0) + 1
                        _blur_active_el()
                    # Use functions for pagination to avoid double-render artifacts
                    def models_prev():
                        st.session_state.model_page = max(0, st.session_state.get('model_page', 0) - 1)
                        _blur_active_el()

                    def models_next():
                        st.session_state.model_page = st.session_state.get('model_page', 0) + 1
                        _blur_active_el()
                    with col_prev:
                        st.button('◀ Prev', key='models_prev', disabled=(page == 0), on_click=models_prev)
                    with col_info:
                        st.caption(f"Showing {start + 1} - {end} of {total} models")
                    with col_next:
                        st.button('Next ▶', key='models_next', disabled=(end >= total), on_click=models_next)
                else:
                    st.info("No models found")
        
        with col2:
            st.subheader("Browse Datasets")
            dataset_search = st.text_input("Search datasets", value="imdb")
            if st.button("Search Datasets"):
                with st.spinner("Fetching datasets..."):
                    try:
                        response = requests.get(f"{API_URL}/datasets?search={dataset_search}&limit=100", headers=get_headers(), timeout=10)
                        if response.status_code == 200:
                            datasets = response.json()
                            st.session_state.datasets = datasets
                            st.session_state.dataset_page = 0
                        else:
                            st.error("Failed to fetch datasets")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Table view with pagination (10 rows per page)
            if "datasets" in st.session_state:
                datasets = st.session_state.datasets or []
                if datasets:
                    page_size = 10
                    page = st.session_state.get('dataset_page', 0)
                    total = len(datasets)
                    start = page * page_size
                    end = min(start + page_size, total)
                    page_items = datasets[start:end]

                    df_ds = []
                    for d in page_items:
                        df_ds.append({
                            'Name': d.get('id'),
                            'Downloads': d.get('downloads') or 'N/A'
                        })
                    try:
                        st.table(df_ds)
                    except Exception as e:
                        st.warning("Table rendering failed due to a dependency issue; showing a simplified list.")
                        for r in df_ds:
                            st.markdown(f"**{r['Name']}** — Downloads: {r['Downloads']}")

                    # Pagination controls
                    col_prev, col_info, col_next = st.columns([1, 6, 1])
                    with col_prev:
                        st.button('◀ Prev', key='datasets_prev', disabled=(page == 0), on_click=datasets_prev)
                    with col_info:
                        st.caption(f"Showing {start + 1} - {end} of {total} datasets")
                    with col_next:
                        st.button('Next ▶', key='datasets_next', disabled=(end >= total), on_click=datasets_next)
                else:
                    st.info("No datasets found")
    
    with tab2:
        st.markdown("<div class='section-card'><h2>📊 Datasets</h2></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load curated datasets
            if "curated_datasets" not in st.session_state:
                try:
                    with st.spinner("Loading curated datasets..."):
                        response = requests.get(f"{API_URL}/datasets/curated", headers=get_headers(), timeout=10)
                    if response.status_code == 200:
                        st.session_state.curated_datasets = response.json()
                    else:
                        st.session_state.curated_datasets = {}
                except Exception as e:
                    st.error(f"Error loading curated datasets: {str(e)}")
                    st.session_state.curated_datasets = {}
            
            st.subheader("Add Dataset")
            
            # Create tabs for different selection methods
            select_tab1, select_tab2 = st.tabs(["📚 Browse Popular Datasets", "✏️ Enter Custom Dataset"])
            
            with select_tab1:
                st.info("💡 Select from popular curated datasets - configuration is automatic!")
                
                curated_data = st.session_state.get("curated_datasets", {})
                
                if curated_data:
                    # Category selection
                    category = st.selectbox(
                        "Select Category",
                        options=list(curated_data.keys()),
                        key="dataset_category"
                    )
                    
                    # Get datasets in selected category
                    datasets_in_category = curated_data.get(category, [])
                    
                    # Create a readable list of options
                    dataset_options = [f"{ds['name']} - {ds['description']}" for ds in datasets_in_category]
                    
                    if dataset_options:
                        selected_option = st.selectbox(
                            "Select Dataset",
                            options=dataset_options,
                            key="selected_dataset"
                        )
                        
                        # Extract the dataset name from the selection
                        selected_dataset_name = selected_option.split(" - ")[0] if selected_option else None
                        
                        # Find the full dataset info
                        selected_dataset_info = next((ds for ds in datasets_in_category if ds['name'] == selected_dataset_name), None)
                        
                        if selected_dataset_info:
                            # Show dataset details
                            st.markdown("---")
                            st.markdown(f"**📊 Dataset:** `{selected_dataset_info['name']}`")
                            st.markdown(f"**📝 Description:** {selected_dataset_info['description']}")
                            st.markdown(f"**📈 Size:** {selected_dataset_info['size']}")
                            st.markdown(f"**🎯 Task:** {selected_dataset_info['task']}")
                            
                            col_x, col_y = st.columns(2)
                            
                            with col_x:
                                if st.button("🔍 Preview Dataset", key="preview_curated"):
                                    with st.spinner("Loading dataset preview..."):
                                        try:
                                            response = requests.post(
                                                f"{API_URL}/datasets/validate",
                                                params={"dataset_name": selected_dataset_name},
                                                headers=get_headers(),
                                                timeout=12
                                            )
                                            if response.status_code == 200:
                                                validation = response.json()
                                                if validation["valid"]:
                                                    st.success(f"✅ Dataset is valid and ready to use!")
                                                    st.write(f"**Total Samples:** {validation['num_samples']:,}")
                                                    st.write(f"**Text Column:** `{validation['text_column']}`")
                                                    st.write(f"**All Columns:** {', '.join([f'`{c}`' for c in validation['columns']])}")
                                                    if validation["sample_texts"]:
                                                        with st.expander("📄 Sample Texts from Dataset"):
                                                            for i, sample in enumerate(validation["sample_texts"], 1):
                                                                st.text(f"{i}. {sample}")
                                                    st.session_state.validated_dataset = selected_dataset_name
                                                else:
                                                    st.error(f"❌ Could not load dataset: {validation['error']}")
                                            else:
                                                st.error("Preview failed")
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                            
                            with col_y:
                                if st.button("➕ Add to My Datasets", key="add_curated"):
                                    with st.spinner("Adding dataset..."):
                                        try:
                                            response = requests.post(
                                                f"{API_URL}/datasets",
                                                json={
                                                    "name": selected_dataset_name,
                                                    "source": selected_dataset_info['description']
                                                },
                                                headers=get_headers(),
                                                timeout=10
                                            )
                                            if response.status_code == 200:
                                                st.success(f"✅ Added {selected_dataset_name} to your datasets!")
                                                st.rerun()
                                            else:
                                                st.error("Dataset already exists or failed to add")
                                        except requests.exceptions.Timeout:
                                            st.error("Request timed out. Please try again.")
                                        except Exception as e:
                                            st.error(f"Error: {str(e)}")
                else:
                    st.warning("Could not load curated datasets")
            
            with select_tab2:
                st.info("💡 Enter any dataset name from Hugging Face Hub")
                dataset_name = st.text_input("Dataset name (e.g., 'imdb', 'squad')", key="custom_dataset_name")
                dataset_source = st.text_input("Description (optional)", key="custom_dataset_source")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔍 Validate Custom Dataset"):
                        if dataset_name:
                            with st.spinner("Validating dataset..."):
                                try:
                                    response = requests.post(
                                        f"{API_URL}/datasets/validate",
                                        params={"dataset_name": dataset_name},
                                        headers=get_headers(),
                                        timeout=12
                                    )
                                    if response.status_code == 200:
                                        validation = response.json()
                                        if validation["valid"]:
                                            st.success(f"✅ Dataset is valid!")
                                            st.write(f"**Samples:** {validation['num_samples']:,}")
                                            st.write(f"**Text Column:** `{validation['text_column']}`")
                                            st.write(f"**Columns:** {', '.join([f'`{c}`' for c in validation['columns']])}")
                                            if validation["sample_texts"]:
                                                with st.expander("📄 Sample Texts"):
                                                    for i, sample in enumerate(validation["sample_texts"], 1):
                                                        st.text(f"{i}. {sample}")
                                            st.session_state.validated_dataset = dataset_name
                                        else:
                                            st.error(f"❌ Validation failed: {validation['error']}")
                                    else:
                                        st.error("Validation request failed")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Please enter a dataset name")
                
                with col_b:
                    if st.button("➕ Add Custom Dataset"):
                        if dataset_name:
                            with st.spinner("Adding dataset..."):
                                try:
                                    response = requests.post(
                                        f"{API_URL}/datasets",
                                        json={"name": dataset_name, "source": dataset_source or "Custom dataset"},
                                        headers=get_headers(),
                                        timeout=10
                                    )
                                    if response.status_code == 200:
                                        st.success("Dataset added successfully")
                                        st.rerun()
                                    else:
                                        st.error("Failed to add dataset")
                                except requests.exceptions.Timeout:
                                    st.error("Request timed out. Please try again")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.warning("Please enter a dataset name")
        
        with col2:
            st.subheader("Saved Datasets")
            try:
                with st.spinner("Loading saved datasets..."):
                    response = requests.get(f"{API_URL}/datasets/saved", headers=get_headers(), timeout=8)
                if response.status_code == 200:
                    saved_datasets = response.json()
                    for ds in saved_datasets:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**{ds['name']}**")
                            st.caption(ds['source'])
                        with col_b:
                            if st.button("🗑️", key=f"del_{ds['id']}"):
                                with st.spinner("Deleting dataset..."):
                                    try:
                                        requests.delete(f"{API_URL}/datasets/{ds['id']}", headers=get_headers(), timeout=6)
                                        st.rerun()
                                    except requests.exceptions.Timeout:
                                        st.error("Delete request timed out. Please try again.")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error loading saved datasets: {str(e)}")
    
    with tab3:
        st.markdown("<div class='section-card'><h2>🚀 Training</h2></div>", unsafe_allow_html=True)
        
        try:
            with st.spinner("Loading saved datasets..."):
                response = requests.get(f"{API_URL}/datasets/saved", headers=get_headers(), timeout=8)
            saved_datasets = response.json() if response.status_code == 200 else []
            dataset_names = [ds['name'] for ds in saved_datasets]
        except:
            dataset_names = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox("Select Model", ["gpt2", "distilgpt2", "facebook/opt-125m", "EleutherAI/pythia-70m"])
            dataset_name = st.selectbox("Select Dataset", dataset_names if dataset_names else ["No datasets available"])
            learning_rate = st.number_input("Learning Rate", value=2e-5, format="%.6f")
        
        with col2:
            num_epochs = st.number_input("Number of Epochs", value=3, min_value=1)
            batch_size = st.number_input("Batch Size", value=4, min_value=1)
            max_length = st.number_input("Max Length", value=512, min_value=64)
        
        if not st.session_state.get('trainer_available', True):
            st.error("⚠️ Training engine is not available on the backend. Please enable training by following the instructions below:")
            st.caption(st.session_state.get('trainer_message', 'Visit the backend logs for more information.'))
        else:
            if st.button("🚀 Start Training", type="primary"):
                if not dataset_names:
                    st.error("Please add a dataset first")
                else:
                    config = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "max_length": max_length
                    }
                    with st.spinner("Starting training..."):
                        try:
                            response = requests.post(f"{API_URL}/trainings", json=config, headers=get_headers(), timeout=20)
                            if response.status_code == 200:
                                st.success("Training started!")
                                st.json(response.json())
                            else:
                                try:
                                    data = response.json()
                                    detail = data.get('detail') if isinstance(data, dict) else None
                                except Exception:
                                    detail = None
                                st.error(detail or "Failed to start training")
                        except requests.exceptions.Timeout:
                            st.error("Training request timed out. Try again later.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        st.divider()
        st.subheader("Training Jobs")
        
        try:
            with st.spinner("Loading trainings..."):
                response = requests.get(f"{API_URL}/trainings", headers=get_headers(), timeout=12)
            if response.status_code == 200:
                trainings = response.json()
                if trainings:
                    # Show training cards
                    for t in trainings:
                        status = t.get('status', 'unknown')
                        mlflow_id = t.get('mlflow_run_id')
                        badge_color = 'orange' if 'running' in status else 'green' if 'completed' in status else 'red' if 'failed' in status else 'gray'
                        mlflow_link = f"http://localhost:5555/#/experiments/0/runs/{mlflow_id}" if mlflow_id else ''
                        details_html = f"<div class='section-card'><div style='display:flex;align-items:center;justify-content:space-between;'><div><strong>{t['model_name']}</strong><br/><small>{t['dataset_name']}</small></div><div style='text-align:right'><span style='color:{badge_color};font-weight:700'>{status}</span>" + (f"<br/><a href='{mlflow_link}' target=_blank style='color:#8ab4ff'>Open MLflow</a>" if mlflow_id else '') + "</div></div></div>"
                        st.markdown(details_html, unsafe_allow_html=True)
                else:
                    st.info("No training jobs yet")
        except Exception as e:
            st.error(f"Error loading trainings: {str(e)}")
    
    with tab4:
        st.markdown("<div class='section-card'><h2>📈 Experiments</h2></div>", unsafe_allow_html=True)
        st.info("Access MLflow UI at: http://localhost:5555")
        
        try:
            with st.spinner("Loading trainings..."):
                response = requests.get(f"{API_URL}/trainings", headers=get_headers(), timeout=12)
            if response.status_code == 200:
                trainings = response.json()
                for training in trainings:
                    with st.expander(f"Training #{training['id']} - {training['model_name']}"):
                        st.write(f"**Status:** {training['status']}")
                        st.write(f"**Dataset:** {training['dataset_name']}")
                        st.write(f"**MLflow Run ID:** {training.get('mlflow_run_id', 'N/A')}")
                        st.write(f"**Created:** {training['created_at']}")
        except Exception as e:
            st.error(f"Error loading experiments: {str(e)}")

# Check for token and backend connection
if not check_backend_connection():
    st.error("⚠️ Backend server is not running!")
    st.info("Please start the backend server first:")
    st.code("cd backend\npy -m uvicorn main:app --reload", language="bash")
    st.stop()

if st.session_state.get("token"):
    # Attempt to verify the token and refresh username; clear token on failure
    username = get_user_info()
    if username is None:
        st.session_state.token = None
        delete_token_file()
        login_page()
    else:
        main_page()
else:
    # No token present – render the login page so users can sign in
    login_page()