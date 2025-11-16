import streamlit as st
import requests
import pandas as pd
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mini Cloud Training", page_icon="☁️", layout="wide")

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}

def login_page():
    st.title("☁️ Mini Cloud Training Platform")
    st.subheader("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            response = requests.post(
                f"{API_URL}/token",
                data={"username": username, "password": password}
            )
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col2:
        if st.button("Register", use_container_width=True):
            response = requests.post(
                f"{API_URL}/register",
                json={"username": username, "password": password}
            )
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.rerun()
            else:
                st.error("Registration failed")

def main_page():
    st.title("☁️ Mini Cloud Training Platform")
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤗 Models & Datasets", "📊 Datasets", "🚀 Training", "📈 Experiments"])
    
    with tab1:
        st.header("Hugging Face Models & Datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Browse Models")
            model_search = st.text_input("Search models", value="gpt2")
            if st.button("Search Models"):
                response = requests.get(f"{API_URL}/models?search={model_search}&limit=20", headers=get_headers())
                if response.status_code == 200:
                    models = response.json()
                    st.session_state.models = models
            
            if "models" in st.session_state:
                models_df = pd.DataFrame(st.session_state.models)
                st.dataframe(models_df, use_container_width=True)
        
        with col2:
            st.subheader("Browse Datasets")
            dataset_search = st.text_input("Search datasets", value="imdb")
            if st.button("Search Datasets"):
                response = requests.get(f"{API_URL}/datasets?search={dataset_search}&limit=20", headers=get_headers())
                if response.status_code == 200:
                    datasets = response.json()
                    st.session_state.datasets = datasets
            
            if "datasets" in st.session_state:
                datasets_df = pd.DataFrame(st.session_state.datasets)
                st.dataframe(datasets_df, use_container_width=True)
    
    with tab2:
        st.header("Manage Datasets")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load curated datasets
            if "curated_datasets" not in st.session_state:
                response = requests.get(f"{API_URL}/datasets/curated", headers=get_headers())
                if response.status_code == 200:
                    st.session_state.curated_datasets = response.json()
                else:
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
                                if st.button("🔍 Preview Dataset", use_container_width=True, key="preview_curated"):
                                    with st.spinner("Loading dataset preview..."):
                                        response = requests.post(
                                            f"{API_URL}/datasets/validate",
                                            params={"dataset_name": selected_dataset_name},
                                            headers=get_headers()
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
                            
                            with col_y:
                                if st.button("➕ Add to My Datasets", use_container_width=True, key="add_curated"):
                                    response = requests.post(
                                        f"{API_URL}/datasets",
                                        json={
                                            "name": selected_dataset_name,
                                            "source": selected_dataset_info['description']
                                        },
                                        headers=get_headers()
                                    )
                                    if response.status_code == 200:
                                        st.success(f"✅ Added {selected_dataset_name} to your datasets!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Dataset already exists or failed to add")
                else:
                    st.warning("Could not load curated datasets")
            
            with select_tab2:
                st.info("💡 Enter any dataset name from Hugging Face Hub")
                dataset_name = st.text_input("Dataset name (e.g., 'imdb', 'squad')", key="custom_dataset_name")
                dataset_source = st.text_input("Description (optional)", key="custom_dataset_source")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔍 Validate Custom Dataset", use_container_width=True):
                        if dataset_name:
                            with st.spinner("Validating dataset..."):
                                response = requests.post(
                                    f"{API_URL}/datasets/validate",
                                    params={"dataset_name": dataset_name},
                                    headers=get_headers()
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
                        else:
                            st.warning("Please enter a dataset name")
                
                with col_b:
                    if st.button("➕ Add Custom Dataset", use_container_width=True):
                        if dataset_name:
                            response = requests.post(
                                f"{API_URL}/datasets",
                                json={"name": dataset_name, "source": dataset_source or "Custom dataset"},
                                headers=get_headers()
                            )
                            if response.status_code == 200:
                                st.success("Dataset added successfully")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to add dataset")
                        else:
                            st.warning("Please enter a dataset name")
        
        with col2:
            st.subheader("Saved Datasets")
            response = requests.get(f"{API_URL}/datasets/saved", headers=get_headers())
            if response.status_code == 200:
                saved_datasets = response.json()
                for ds in saved_datasets:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{ds['name']}**")
                        st.caption(ds['source'])
                    with col_b:
                        if st.button("🗑️", key=f"del_{ds['id']}"):
                            requests.delete(f"{API_URL}/datasets/{ds['id']}", headers=get_headers())
                            st.rerun()
    
    with tab3:
        st.header("Fine-Tuning Configuration")
        
        response = requests.get(f"{API_URL}/datasets/saved", headers=get_headers())
        saved_datasets = response.json() if response.status_code == 200 else []
        dataset_names = [ds['name'] for ds in saved_datasets]
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox("Select Model", ["gpt2", "distilgpt2", "facebook/opt-125m", "EleutherAI/pythia-70m"])
            dataset_name = st.selectbox("Select Dataset", dataset_names if dataset_names else ["No datasets available"])
            learning_rate = st.number_input("Learning Rate", value=2e-5, format="%.6f")
        
        with col2:
            num_epochs = st.number_input("Number of Epochs", value=3, min_value=1)
            batch_size = st.number_input("Batch Size", value=4, min_value=1)
            max_length = st.number_input("Max Length", value=512, min_value=64)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
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
                response = requests.post(f"{API_URL}/trainings", json=config, headers=get_headers())
                if response.status_code == 200:
                    st.success("Training started!")
                    st.json(response.json())
                else:
                    st.error("Failed to start training")
        
        st.divider()
        st.subheader("Training Jobs")
        
        response = requests.get(f"{API_URL}/trainings", headers=get_headers())
        if response.status_code == 200:
            trainings = response.json()
            if trainings:
                df = pd.DataFrame(trainings)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No training jobs yet")
    
    with tab4:
        st.header("MLflow Experiments")
        st.info("Access MLflow UI at: http://localhost:5000")
        
        response = requests.get(f"{API_URL}/trainings", headers=get_headers())
        if response.status_code == 200:
            trainings = response.json()
            for training in trainings:
                with st.expander(f"Training #{training['id']} - {training['model_name']}"):
                    st.write(f"**Status:** {training['status']}")
                    st.write(f"**Dataset:** {training['dataset_name']}")
                    st.write(f"**MLflow Run ID:** {training.get('mlflow_run_id', 'N/A')}")
                    st.write(f"**Created:** {training['created_at']}")

if "token" not in st.session_state:
    login_page()
else:
    main_page()
