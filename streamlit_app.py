"""
California Housing Price Predictor - Streamlit Application
Author: Data Science Team
Last Modified: 2025-11-17

A production-grade ML application for predicting California housing prices
using a trained K-Nearest Neighbors pipeline.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Configuration
MODEL_PATH = Path("california_knn_pipeline.pkl")
FEATURE_DESCRIPTIONS = {
    "MedInc": "Median income in block group (in tens of thousands)",
    "HouseAge": "Median age of houses in block group",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average number of household members",
    "Latitude": "Block group latitude",
    "Longitude": "Block group longitude"
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from disk with error handling."""
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please ensure the model file is in the application directory."
        )
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def get_feature_names(model) -> List[str]:
    """
    Extract feature names from the model pipeline.
    
    Falls back to canonical California Housing dataset features if extraction fails.
    """
    try:
        # Attempt to extract from preprocessor
        return list(model.named_steps['preprocessor'].transformers_[0][2])
    except (AttributeError, KeyError, IndexError) as e:
        logger.warning(f"Could not extract features from model: {e}. Using defaults.")
        return [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]


def validate_input_data(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate that input data contains all required columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        null_features = null_counts[null_counts > 0].to_dict()
        return False, f"Null values detected: {null_features}"
    
    return True, None


def predict_single(model, features: Dict[str, float]) -> float:
    """Generate a single prediction from input features."""
    cols = get_feature_names(model)
    df = pd.DataFrame([features], columns=cols)
    prediction = model.predict(df)[0]
    return float(prediction)


def batch_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a batch of samples."""
    cols = get_feature_names(model)
    
    # Validate input
    is_valid, error_msg = validate_input_data(df, cols)
    if not is_valid:
        raise ValueError(error_msg)
    
    # Create output dataframe with ordered columns
    result = df[cols].copy()
    result['predicted_value'] = model.predict(result)
    
    return result


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    residuals = y_true - y_pred
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std()
    }


def plot_predictions_vs_actual(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Create an interactive scatter plot of predictions vs actual values."""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(size=6, opacity=0.6, color='steelblue'),
        name='Predictions',
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Values',
        xaxis_title='Actual Median House Value ($100k)',
        yaxis_title='Predicted Median House Value ($100k)',
        hovermode='closest',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_residuals(y_true: pd.Series, y_pred: pd.Series) -> go.Figure:
    """Create residual plot for error analysis."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(size=6, opacity=0.6, color='steelblue'),
        hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    
    fig.update_layout(
        title='Residual Plot',
        xaxis_title='Predicted Value ($100k)',
        yaxis_title='Residual (Actual - Predicted)',
        height=400
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Create an interactive bar chart for feature importance."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['importance_mean'],
        y=importance_df['feature'],
        orientation='h',
        error_x=dict(type='data', array=importance_df['importance_std']),
        marker=dict(color='steelblue'),
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Permutation Feature Importance',
        xaxis_title='Mean Importance (± std)',
        yaxis_title='Feature',
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def compute_feature_importance(
    model, 
    X: pd.DataFrame, 
    y: pd.Series, 
    n_repeats: int = 10
) -> Optional[pd.DataFrame]:
    """
    Calculate permutation feature importance.
    
    Uses multiple repeats to get stable estimates with standard deviations.
    """
    try:
        result = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Feature importance calculation failed: {e}")
        return None


def render_metrics_dashboard(metrics: Dict[str, float]):
    """Display model performance metrics in a clean layout."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MAE", 
            f"${metrics['mae']:.3f}",
            help="Mean Absolute Error (in $100k)"
        )
    
    with col2:
        st.metric(
            "RMSE", 
            f"${metrics['rmse']:.3f}",
            help="Root Mean Squared Error (in $100k)"
        )
    
    with col3:
        st.metric(
            "R² Score", 
            f"{metrics['r2']:.3f}",
            help="Coefficient of Determination (0-1, higher is better)"
        )
    
    with col4:
        st.metric(
            "MAPE", 
            f"{metrics['mape']:.2f}%",
            help="Mean Absolute Percentage Error"
        )


def main():
    st.set_page_config(
        page_title="California Housing Predictor",
        page_icon="🏘️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("California Housing Price Predictor")
    st.markdown(
        """
        **Interactive ML Application** | K-Nearest Neighbors Pipeline  
        Predict median house values based on California Housing dataset features.
        """
    )
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model()
        features = get_feature_names(model)
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.stop()
    
    # Sidebar for single prediction
    st.sidebar.header("🎯 Single Prediction")
    st.sidebar.markdown("Adjust features to estimate median house value")
    
    single_input = {}
    for feature in features:
        if feature == "MedInc":
            single_input[feature] = st.sidebar.slider(
                "Median Income ($10k)", 0.5, 15.0, 3.5, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "HouseAge":
            single_input[feature] = st.sidebar.slider(
                "House Age (years)", 1.0, 52.0, 28.0, 1.0,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "AveRooms":
            single_input[feature] = st.sidebar.slider(
                "Avg Rooms", 1.0, 10.0, 5.4, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "AveBedrms":
            single_input[feature] = st.sidebar.slider(
                "Avg Bedrooms", 0.5, 5.0, 1.0, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "Population":
            single_input[feature] = st.sidebar.slider(
                "Population", 100.0, 5000.0, 1425.0, 50.0,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "AveOccup":
            single_input[feature] = st.sidebar.slider(
                "Avg Occupancy", 1.0, 6.0, 3.0, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "Latitude":
            single_input[feature] = st.sidebar.slider(
                "Latitude", 32.5, 42.0, 35.6, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
        elif feature == "Longitude":
            single_input[feature] = st.sidebar.slider(
                "Longitude", -124.3, -114.3, -119.5, 0.1,
                help=FEATURE_DESCRIPTIONS[feature]
            )
    
    if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            prediction = predict_single(model, single_input)
            st.sidebar.success(
                f"**Estimated Median House Value**\n\n"
                f"## ${prediction * 100:,.0f}"
            )
            st.sidebar.caption(f"(${prediction:.2f} in $100k units)")
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Batch Predictions", "📈 Model Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Batch Prediction Interface")
        st.markdown(
            """
            Upload a CSV file containing housing features to generate predictions.  
            Optionally include a target column for performance evaluation.
            """
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV must contain all required feature columns"
            )
        
        with col2:
            target_column = st.text_input(
                "Target column name (optional)",
                value="MedHouseVal",
                help="Column name containing actual values for metric calculation"
            )
            
            # Template download button
            st.markdown("---")
            st.markdown("**Need help?**")
            template_df = pd.DataFrame(columns=features + [target_column])
            # Add sample rows
            sample_data = {
                "MedInc": [3.5, 4.2, 2.8],
                "HouseAge": [28.0, 35.0, 15.0],
                "AveRooms": [5.4, 6.1, 4.8],
                "AveBedrms": [1.0, 1.2, 0.9],
                "Population": [1425.0, 2100.0, 950.0],
                "AveOccup": [3.0, 2.8, 3.5],
                "Latitude": [35.6, 37.8, 34.2],
                "Longitude": [-119.5, -122.4, -118.3],
                target_column: [2.5, 3.8, 2.1]
            }
            template_df = pd.DataFrame(sample_data)
            template_csv = template_df.to_csv(index=False)
            
            st.download_button(
                "📥 Download CSV Template",
                template_csv,
                "housing_data_template.csv",
                "text/csv",
                help="Download a template CSV with sample data",
                use_container_width=True
            )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                
                if st.button("▶️ Run Predictions", type="primary"):
                    with st.spinner("Generating predictions..."):
                        try:
                            predictions_df = batch_predict(model, df)
                            
                            st.success(f"✅ Generated {len(predictions_df):,} predictions")
                            
                            # Display predictions
                            st.subheader("Predictions")
                            st.dataframe(predictions_df, use_container_width=True)
                            
                            # Download button
                            csv = predictions_df.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Predictions",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                            # Compute metrics if target column exists
                            if target_column in df.columns:
                                st.subheader("Performance Metrics")
                                
                                y_true = df[target_column]
                                y_pred = predictions_df['predicted_value']
                                
                                metrics = compute_regression_metrics(y_true, y_pred)
                                render_metrics_dashboard(metrics)
                                
                                # Visualization tabs
                                viz_tab1, viz_tab2 = st.tabs(["Predictions vs Actual", "Residual Analysis"])
                                
                                with viz_tab1:
                                    fig_scatter = plot_predictions_vs_actual(y_true, y_pred)
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                
                                with viz_tab2:
                                    fig_residuals = plot_residuals(y_true, y_pred)
                                    st.plotly_chart(fig_residuals, use_container_width=True)
                                    
                                    st.markdown("**Residual Statistics**")
                                    res_col1, res_col2 = st.columns(2)
                                    with res_col1:
                                        st.metric("Mean Residual", f"{metrics['mean_residual']:.4f}")
                                    with res_col2:
                                        st.metric("Std Residual", f"{metrics['std_residual']:.4f}")
                            
                        except ValueError as e:
                            st.error(f"❌ Validation Error: {e}")
                        except Exception as e:
                            st.error(f"❌ Prediction Error: {e}")
                            logger.exception("Batch prediction failed")
                
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
    
    with tab2:
        st.header("Model Analysis & Interpretability")
        
        if uploaded_file is not None and target_column in df.columns:
            st.markdown("Analyze feature importance using permutation-based methods.")
            
            compute_importance = st.checkbox(
                "🧮 Compute Feature Importance",
                help="May take 30-60 seconds for large datasets"
            )
            
            if compute_importance:
                n_repeats = st.slider("Number of permutations", 5, 30, 10)
                
                with st.spinner("Computing feature importance..."):
                    cols = get_feature_names(model)
                    X = df[cols]
                    y = df[target_column]
                    
                    importance_df = compute_feature_importance(model, X, y, n_repeats)
                    
                    if importance_df is not None:
                        fig_importance = plot_feature_importance(importance_df)
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        with st.expander("📋 Detailed Importance Values"):
                            st.dataframe(importance_df, use_container_width=True)
                    else:
                        st.error("Failed to compute feature importance")
        else:
            st.info("Upload a CSV file with a target column to enable model analysis.")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### Model Details
        - **Algorithm:** K-Nearest Neighbors Regression
        - **Dataset:** California Housing (1990 Census)
        - **Target Variable:** Median house value (in $100,000)
        
        ### Features Used
        """)
        
        for feature, description in FEATURE_DESCRIPTIONS.items():
            st.markdown(f"- **{feature}:** {description}")
        
        st.markdown("""
        ### Interpretation Notes
        - All predictions are in units of $100,000
        - Model performance metrics:
            - **MAE:** Average absolute error in predictions
            - **RMSE:** Root mean squared error (penalizes large errors)
            - **R²:** Proportion of variance explained (0-1 scale)
            - **MAPE:** Mean absolute percentage error
        
        ### Usage Guidelines
        1. Use the sidebar for quick single predictions
        2. Upload CSV files for batch processing
        3. Include target column to evaluate model performance
        4. Analyze feature importance to understand model behavior
        
        ---
        *Built with Streamlit, scikit-learn, and Plotly*
        """)


if __name__ == "__main__":
    main()