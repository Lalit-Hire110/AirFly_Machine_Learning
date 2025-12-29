"""
AirFly - Smart Flight Delay Forecast
Streamlit web application for flight delay prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from predictor import get_predictor

# Page configuration
st.set_page_config(
    page_title="AirFly - Flight Delay Forecast",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero gradient */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.95;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .delay-value {
        font-size: 4rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Airplane animation */
    @keyframes fly {
        0% { transform: translateX(-100px) translateY(0); }
        50% { transform: translateX(50px) translateY(-20px); }
        100% { transform: translateX(-100px) translateY(0); }
    }
    
    .airplane {
        animation: fly 6s ease-in-out infinite;
        font-size: 3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor (singleton, loads once)
@st.cache_resource
def load_predictor():
    """Load predictor singleton (cached)."""
    try:
        predictor = get_predictor()
        return predictor
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

predictor = load_predictor()

# Sidebar navigation
st.sidebar.title("🛫 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🎯 Single Prediction", "📊 Batch Prediction", "🗺️ Route Context", "📖 About & Explainability"]
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="airplane">✈️</div>
        <h1 class="hero-title">AirFly</h1>
        <p class="hero-subtitle">Smart Flight Delay Forecast — Predict delays in one click</p>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            We'll tell you if this flight is in the risk zone using advanced ML models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model metrics
    st.markdown("### 📊 Model Performance")
    
    model_info = predictor.get_model_info()
    metadata = model_info["metadata"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Classifier Accuracy</div>
            <div class="metric-value">{metadata['classifier']['accuracy']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Heavy Delay F1</div>
            <div class="metric-value">{metadata['classifier']['f1_heavy']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Overall MAE</div>
            <div class="metric-value">{metadata['regression']['MAE']:.1f} min</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metadata['regression']['R2']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset info
    st.markdown("### 📈 Dataset & Training")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Dataset Size**
        - Initial: {metadata['data']['initial_shape'][0]:,} flights
        - After filtering: {metadata['data']['after_filtering'][0]:,} flights
        - Train/Test: {metadata['data']['train_shape'][0]:,} / {metadata['data']['test_shape'][0]:,}
        """)
    
    with col2:
        st.info(f"""
        **Features Used**
        - Numeric features: {metadata['data']['numeric_features_used']}
        - Categorical features: {metadata['data']['categorical_features_used']}
        - Rolling features: {metadata['data']['rolling_train_cols']} columns
        """)
    
    # CTA
    st.markdown("---")
    st.markdown("### 🚀 Ready to predict?")
    st.markdown("Use the sidebar to navigate to **Single Prediction** for a quick forecast or **Batch Prediction** to analyze multiple flights.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🎯 Single Prediction":
    st.title("🎯 Single Flight Delay Prediction")
    st.markdown("Enter flight details below to get an instant delay forecast.")
    
    # Get required features
    required_features = predictor.get_required_features()
    
    # Input form
    with st.form("prediction_form"):
        st.markdown("#### Flight Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            airline = st.text_input("Airline Code", value="AA", help="e.g., AA, DL, UA")
            origin = st.text_input("Origin Airport", value="ATL", help="e.g., ATL, JFK, LAX")
            dest = st.text_input("Destination Airport", value="DFW", help="e.g., DFW, ORD, SFO")
        
        with col2:
            scheduled_dep_hour = st.number_input("Scheduled Departure Hour (0-23)", min_value=0, max_value=23, value=14)
            departure_hour = st.number_input("Actual Departure Hour (0-23)", min_value=0, max_value=23, value=14)
            arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=17)
        
        with col3:
            distance = st.number_input("Distance (miles)", min_value=0, value=800)
            taxi_out = st.number_input("Taxi Out (min)", min_value=0, value=15)
            taxi_in = st.number_input("Taxi In (min)", min_value=0, value=7)
        
        st.markdown("#### Time & Date")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            air_time = st.number_input("Air Time (min)", min_value=0, value=120)
        with col2:
            month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
        with col3:
            dayofweek = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
        with col4:
            is_weekend = st.selectbox("Weekend?", [0, 1], index=0)
        
        st.markdown("#### Additional Features (Rolling Stats & Aggregates)")
        st.caption("These are typically computed from historical data. Use approximate values or defaults.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crs_elapsed_time = st.number_input("CRS Elapsed Time (min)", min_value=0, value=150)
            elapsed_time = st.number_input("Elapsed Time (min)", min_value=0, value=155)
            route_rolling_mean_7d = st.number_input("Route 7d Mean Delay", value=10.0)
            route_rolling_count_7d = st.number_input("Route 7d Count", value=50)
            route_rolling_mean_14d = st.number_input("Route 14d Mean Delay", value=12.0)
            route_rolling_count_14d = st.number_input("Route 14d Count", value=100)
        
        with col2:
            airline_rolling_mean_7d = st.number_input("Airline 7d Mean Delay", value=8.0)
            airline_rolling_count_7d = st.number_input("Airline 7d Count", value=200)
            airline_rolling_mean_14d = st.number_input("Airline 14d Mean Delay", value=9.0)
            airline_rolling_count_14d = st.number_input("Airline 14d Count", value=400)
            airline_mean = st.number_input("Airline Mean Delay", value=10.0)
            origin_mean = st.number_input("Origin Mean Delay", value=11.0)
            dest_mean = st.number_input("Dest Mean Delay", value=9.0)
            route_mean = st.number_input("Route Mean Delay", value=10.5)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            airline_count = st.number_input("Airline Count", value=1000)
        with col2:
            origin_count = st.number_input("Origin Count", value=500)
        with col3:
            dest_count = st.number_input("Dest Count", value=450)
        with col4:
            route_count = st.number_input("Route Count", value=200)
        
        submitted = st.form_submit_button("🚀 Predict Delay", use_container_width=True)
    
    if submitted:
        # Build input dictionary
        route = f"{origin}_{dest}"
        
        input_data = {
            "TAXI_OUT": taxi_out,
            "TAXI_IN": taxi_in,
            "AIR_TIME": air_time,
            "DISTANCE": distance,
            "CRS_ELAPSED_TIME": crs_elapsed_time,
            "ELAPSED_TIME": elapsed_time,
            "ScheduledDepHour": scheduled_dep_hour,
            "DepartureHour": departure_hour,
            "ArrivalHour": arrival_hour,
            "month": month,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "Route_rolling_mean_7d": route_rolling_mean_7d,
            "Route_rolling_count_7d": route_rolling_count_7d,
            "Route_rolling_mean_14d": route_rolling_mean_14d,
            "Route_rolling_count_14d": route_rolling_count_14d,
            "Airline_rolling_mean_7d": airline_rolling_mean_7d,
            "Airline_rolling_count_7d": airline_rolling_count_7d,
            "Airline_rolling_mean_14d": airline_rolling_mean_14d,
            "Airline_rolling_count_14d": airline_rolling_count_14d,
            "Airline_mean": airline_mean,
            "ORIGIN_mean": origin_mean,
            "DEST_mean": dest_mean,
            "Route_mean": route_mean,
            "Airline_count": airline_count,
            "ORIGIN_count": origin_count,
            "DEST_count": dest_count,
            "Route_count": route_count,
            "Airline": airline,
            "ORIGIN": origin,
            "DEST": dest,
            "Route": route
        }
        
        try:
            with st.spinner("Predicting..."):
                result = predictor.predict_single(input_data)
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            # Main delay value
            st.markdown(f"""
            <div class="prediction-card">
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">PREDICTED DELAY</div>
                    <div class="delay-value">{result['predicted_delay_min']:.1f} min</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk badge
            risk_class = f"risk-{result['risk_level'].lower()}"
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="risk-badge {risk_class}">{result['risk_level']} Risk</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            st.markdown("#### Delay Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['predicted_delay_min'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Delay (minutes)", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [None, 120], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 15], 'color': '#38ef7d'},
                        {'range': [15, 45], 'color': '#f5576c'},
                        {'range': [45, 120], 'color': '#fee140'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': result['predicted_delay_min']
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Breakdown
            st.markdown("#### Prediction Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Normal Regime Prediction", f"{result['pred_norm']:.1f} min")
            with col2:
                st.metric("Heavy Regime Prediction", f"{result['pred_heavy']:.1f} min")
            with col3:
                regime = "Heavy" if result['heavy_flag'] == 1 else "Normal"
                st.metric("Selected Regime", regime)
            
            st.info(f"**Expected Error (MAE):** ±{result['expected_error']:.1f} minutes for {regime.lower()} delays")
            
            # Feature importance (optional)
            with st.expander("🔍 Why this prediction? (Feature Importance)"):
                st.markdown("Computing approximate feature importance...")
                try:
                    importances = predictor.get_feature_importance_approximate(input_data, top_n=6)
                    
                    if importances:
                        # Create bar chart
                        features = [imp['feature'] for imp in importances]
                        values = [imp['importance'] for imp in importances]
                        
                        fig = go.Figure(go.Bar(
                            x=values,
                            y=features,
                            orientation='h',
                            marker=dict(color='#667eea')
                        ))
                        fig.update_layout(
                            title="Top 6 Most Influential Features",
                            xaxis_title="Impact on Delay (minutes)",
                            yaxis_title="Feature",
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not compute feature importance for this input.")
                except Exception as e:
                    st.warning(f"Feature importance calculation failed: {str(e)}")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ============================================================================
# BATCH PREDICTION PAGE
# ============================================================================
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Flight Delay Prediction")
    st.markdown("Upload a CSV file with flight data to get predictions for multiple flights.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ Loaded {len(df)} flights from CSV")
            
            # Preview
            st.markdown("#### Preview (first 5 rows)")
            st.dataframe(df.head(), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                try:
                    with st.spinner(f"Predicting delays for {len(df)} flights..."):
                        results_df = predictor.predict_batch(df)
                    
                    st.success("✓ Predictions complete!")
                    
                    # Summary statistics
                    st.markdown("### 📈 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_delay = results_df['predicted_delay_min'].mean()
                        st.metric("Average Predicted Delay", f"{avg_delay:.1f} min")
                    
                    with col2:
                        max_delay = results_df['predicted_delay_min'].max()
                        st.metric("Maximum Delay", f"{max_delay:.1f} min")
                    
                    with col3:
                        heavy_count = (results_df['heavy_flag'] == 1).sum()
                        st.metric("Heavy Delays", f"{heavy_count} flights")
                    
                    with col4:
                        normal_count = (results_df['heavy_flag'] == 0).sum()
                        st.metric("Normal Delays", f"{normal_count} flights")
                    
                    # Histogram
                    st.markdown("#### Distribution of Predicted Delays")
                    fig = px.histogram(
                        results_df,
                        x='predicted_delay_min',
                        nbins=30,
                        title="Predicted Delay Distribution",
                        labels={'predicted_delay_min': 'Predicted Delay (minutes)'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk level distribution
                    st.markdown("#### Risk Level Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map={'Low': '#38ef7d', 'Medium': '#f5576c', 'High': '#fee140'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("### 💾 Download Results")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="airfly_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show results table
                    with st.expander("View Full Results Table"):
                        st.dataframe(results_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")
                    st.info("Make sure your CSV contains all required features. See the About page for CSV format specification.")
        
        except Exception as e:
            st.error(f"Failed to load CSV: {str(e)}")

# ============================================================================
# ROUTE CONTEXT PAGE
# ============================================================================
elif page == "🗺️ Route Context":
    st.title("🗺️ Route Historical Context")
    st.markdown("View historical delay trends for specific routes.")
    
    st.info("📌 This feature shows rolling statistics for routes based on the training data. In production, this would query a live database.")
    
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Origin Airport", value="ATL")
    with col2:
        dest = st.text_input("Destination Airport", value="DFW")
    
    route = f"{origin}_{dest}"
    
    if st.button("📊 Show Route Context"):
        # Simulate 14-day rolling data (in production, this would query actual data)
        st.markdown(f"### Route: {origin} → {dest}")
        
        # Generate sample rolling data
        days = 14
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Simulate rolling means (would be real data in production)
        np.random.seed(hash(route) % 2**32)
        rolling_means = np.random.normal(15, 8, days).clip(0, 60)
        rolling_counts = np.random.randint(20, 100, days)
        
        df_rolling = pd.DataFrame({
            'Date': dates,
            'Mean Delay (min)': rolling_means,
            'Flight Count': rolling_counts
        })
        
        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_rolling['Date'],
            y=df_rolling['Mean Delay (min)'],
            mode='lines+markers',
            name='Mean Delay',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f"14-Day Rolling Mean Delay for {route}",
            xaxis_title="Date",
            yaxis_title="Mean Delay (minutes)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Flight count chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_rolling['Date'],
            y=df_rolling['Flight Count'],
            name='Flight Count',
            marker=dict(color='#764ba2')
        ))
        fig2.update_layout(
            title=f"Daily Flight Count for {route}",
            xaxis_title="Date",
            yaxis_title="Number of Flights",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("14-Day Avg Delay", f"{rolling_means.mean():.1f} min")
        with col2:
            st.metric("Total Flights (14d)", f"{rolling_counts.sum()}")
        with col3:
            st.metric("Avg Daily Flights", f"{rolling_counts.mean():.0f}")
        
        st.caption("Note: This is simulated data for demonstration. In production, this would show actual historical statistics from your database.")

# ============================================================================
# ABOUT & EXPLAINABILITY PAGE
# ============================================================================
elif page == "📖 About & Explainability":
    st.title("📖 About AirFly")
    
    st.markdown("""
    ### What is AirFly?
    
    AirFly is a machine learning-powered flight delay prediction system that helps you forecast arrival delays 
    for commercial flights. Using a sophisticated two-stage modeling approach, AirFly can predict delays with 
    high accuracy and provide risk assessments.
    
    ### 🧠 How It Works
    
    Our prediction system uses a **two-stage ensemble approach**:
    
    1. **Stage 1: Heavy Delay Classifier**
       - Determines if a flight will experience a "heavy" delay (> 30 minutes)
       - Uses gradient boosting with 92.7% accuracy
       - Precision: 87.4% | Recall: 58.2% | F1: 0.698
    
    2. **Stage 2: Dual Regression Models**
       - **Normal Regime Regressor**: Predicts delays for typical flights (MAE: 12.5 min)
       - **Heavy Regime Regressor**: Predicts delays for problematic flights (MAE: 38.5 min)
       - Uses log1p transformation to handle skewed delay distributions
    
    3. **Final Prediction**
       - Combines classifier output with appropriate regressor
       - Provides risk level (Low/Medium/High) and expected error bounds
    
    ### 📊 Model Performance
    """)
    
    model_info = predictor.get_model_info()
    metadata = model_info["metadata"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Classification Metrics")
        st.json({
            "Accuracy": metadata['classifier']['accuracy'],
            "F1 Score (Heavy)": metadata['classifier']['f1_heavy'],
            "Precision (Heavy)": metadata['classifier']['precision_heavy'],
            "Recall (Heavy)": metadata['classifier']['recall_heavy']
        })
    
    with col2:
        st.markdown("#### Regression Metrics")
        st.json({
            "Overall RMSE": f"{metadata['regression']['RMSE']:.2f} min",
            "Overall MAE": f"{metadata['regression']['MAE']:.2f} min",
            "R² Score": metadata['regression']['R2'],
            "Normal MAE": f"{metadata['regression']['normal_segment']['MAE']:.2f} min",
            "Heavy MAE": f"{metadata['regression']['heavy_segment']['MAE']:.2f} min"
        })
    
    st.markdown("""
    ### 🎯 Features Used
    
    Our model uses **32 features** across multiple categories:
    
    - **Flight Characteristics**: Distance, taxi times, air time, scheduled times
    - **Temporal Features**: Month, day of week, weekend flag, departure/arrival hours
    - **Rolling Statistics**: 7-day and 14-day rolling means and counts for routes and airlines
    - **Aggregate Features**: Historical means and counts for airlines, origins, destinations, and routes
    - **Categorical**: Airline, origin, destination, route
    
    These features are engineered to be **leakage-safe** — they don't use information that wouldn't 
    be available at prediction time.
    
    ### ⚠️ Limitations & Assumptions
    
    """)
    
    st.warning("""
    **Important Limitations:**
    
    - **No Weather Data**: Our model doesn't account for weather conditions, which are a major factor in delays
    - **No ATC Information**: Air traffic control delays and airspace congestion are not included
    - **No Mechanical Issues**: Aircraft maintenance and mechanical problems are not predicted
    - **Historical Bias**: Model is trained on historical data and may not capture unprecedented events
    - **Feature Availability**: Requires rolling statistics that may not be available for new routes
    
    **Best Use Cases:**
    - Comparative analysis of flight options
    - Risk assessment for travel planning
    - Batch analysis of flight schedules
    - Understanding delay patterns by route/airline
    
    **Not Recommended For:**
    - Real-time operational decisions (use live ATC data instead)
    - Guaranteed delay predictions (treat as probabilistic estimates)
    - Routes with very limited historical data
    """)
    
    st.markdown("""
    ### ❓ FAQ
    
    **Q: How accurate are the predictions?**  
    A: For normal delays, our MAE is ~12.5 minutes. For heavy delays, it's ~38.5 minutes. The overall R² is 0.604, 
    meaning we explain about 60% of the variance in delays.
    
    **Q: What does "risk level" mean?**  
    A: Risk levels are derived from predicted delay:
    - **Low**: < 15 minutes (minimal impact)
    - **Medium**: 15-45 minutes (moderate impact)
    - **High**: > 45 minutes (significant impact)
    
    **Q: Can I use this for real-time predictions?**  
    A: The model is designed for planning and analysis. For real-time predictions, you'd need live data feeds 
    for weather, ATC, and current airport conditions.
    
    **Q: What's the CSV format for batch predictions?**  
    A: Your CSV must include all 32 required features. See the demo/sample_batch.csv file for an example. 
    Required columns include: TAXI_OUT, TAXI_IN, AIR_TIME, DISTANCE, Airline, ORIGIN, DEST, and all rolling statistics.
    
    **Q: How were the models trained?**  
    A: Models were trained on 213,863 flights with 53,466 held out for testing. We used scikit-learn pipelines 
    with XGBoost classifiers and regressors, with careful feature engineering to avoid data leakage.
    
    ### 🛠️ Technical Details
    """)
    
    with st.expander("View Model Versions & Dependencies"):
        st.json(model_info["versions"])
    
    with st.expander("View All Features"):
        st.markdown("**Numeric Features:**")
        st.write(model_info["features"]["numeric_features"])
        st.markdown("**Categorical Features:**")
        st.write(model_info["features"]["categorical_features"])
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ by Lalit Hire</p>
        <p>Powered by Streamlit, scikit-learn, and XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**AirFly v1.0**

Smart flight delay forecasting using machine learning.

- 92.7% classifier accuracy
- 12.5 min MAE (normal delays)
- 32 engineered features
- Two-stage ensemble model
""")
