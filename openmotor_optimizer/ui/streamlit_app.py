import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="OpenMotor-Optimizer Dashboard", layout="wide")

st.title("🚀 OpenMotor-Optimizer")
st.markdown("Physics-guided optimization of solid rocket motors with GPU-accelerated search.")

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../experiments/results"))

st.sidebar.header("Experiment Tracking")
if os.path.exists(RESULTS_DIR):
    experiments = sorted(os.listdir(RESULTS_DIR), reverse=True)
    selected_exp = st.sidebar.selectbox("Select Experiment Run", experiments)
    
    if selected_exp:
        exp_path = os.path.join(RESULTS_DIR, selected_exp)
        csv_file = os.path.join(exp_path, "motors.csv")
        ric_file = os.path.join(exp_path, "best_motor.ric")
        
        st.subheader(f"Run: {selected_exp}")
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            st.write(f"**Total Valid Motors Evaluated**: {len(df)}")
            
            # Metrics columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Max Peak Thrust (N)", f"{df['peak_thrust_N'].max():.1f}")
            col2.metric("Max Total Impulse (Ns)", f"{df['total_impulse_Ns'].max():.1f}")
            col3.metric("Max Burn Time (s)", f"{df['burn_time_s'].max():.2f}")
            col4.metric("Avg Specific Impulse (s)", f"{df['specific_impulse_s'].mean():.1f}")
            
            # Plot Pareto Frontier
            st.subheader("Pareto Frontier: Internal Ballistics")
            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(df['total_impulse_Ns'], df['peak_thrust_N'], 
                            c=df['peak_pressure_Pa'], cmap='viridis', alpha=0.6)
            plt.colorbar(sc, label="Peak Pressure (Pa)")
            ax.set_xlabel("Total Impulse (N·s)")
            ax.set_ylabel("Peak Thrust (N)")
            ax.set_title("Total Impulse vs. Peak Thrust by Chamber Pressure")
            st.pyplot(fig)
            
            st.dataframe(df.head(50))
            
        if os.path.exists(ric_file):
            with open(ric_file, "r") as f:
                ric_data = f.read()
            st.sidebar.download_button("Download Best .ric", ric_data, file_name=f"{selected_exp}_best.ric")
else:
    st.info("No experiments found. Run `run_experiment.py` first.")
