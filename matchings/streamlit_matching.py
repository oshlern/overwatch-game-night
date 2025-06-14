import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
import cvxpy as cp
from team_matching import *
import team_matching as tm

st.set_page_config(layout="wide")

# st.title("Team Matching App")

uploaded_file = st.file_uploader("upload RanksData", type="csv")
# Create sliders for setting the constants
tm.WEIGHTS["AVG_SR_WEIGHT"] = st.slider("Average SR Weight", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
tm.WEIGHTS["SR_DIFF_PENALTY_WEIGHT"] = st.slider("SR Difference Penalty Weight", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
tm.WEIGHTS["AVG_COMF_WEIGHT"] = st.slider("Average Comfort Weight", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
tm.WEIGHTS["MAX_SR_DIFF"] = st.slider("Max SR Difference", min_value=0.0, max_value=10.0, value=5.0, step=0.01)

if uploaded_file is None:
    uploaded_file = "5v5 - RanksData (10).csv"


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # df['Playing'] = False

    # Configure AgGrid options
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True, pre_selected_rows=list(range(len(df))))
    gridOptions = gb.build()

    # Display the dataframe with AgGrid
    grid_response = AgGrid(df, gridOptions=gridOptions, update_mode='MODEL_CHANGED', height=400, width='100%')
    selected_rows = grid_response['selected_rows']
    selected_df = pd.DataFrame(selected_rows)

    # st.write("### Selected Rows")
    # st.dataframe(selected_df)

    # st.write("### All Players")
    # st.dataframe(df)

    if st.button("Generate Teams"):
        result_df, avg_srs, devs, comfs, identity, id_n, n_roles, n_players, role_comfs, role_srs = team_matching(selected_df)

        st.text_area("Teams", get_team_txt(result_df), height=60)

        avg_comf, sr_diff, overall_avg_sr = compute_team_metrics(result_df)

        # st.write("### Team Metrics")
        st.write(f"Average Comfort Score: {avg_comf:.2f}")
        st.write(f"SR Difference: {sr_diff:.2f}")
        st.write(f"Overall Average SR: {overall_avg_sr:.2f}")

                # st.write("### Matchings")
        st.dataframe(result_df)

        # Plot the summary data
        # fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # plot_summary_data(n_players, identity, role_comfs, role_srs, 0.005, 0.1, id_n, n_roles, 5, 2)

        # st.pyplot(fig)

        # # Print summary of cost factors
        # st.write("\n### Summary of Cost Factors:")
        # st.write(f"Total comfort scores: {total_comfort_scores}")
        # st.write(f"Average SRs: {average_srs}")
        # st.write(f"SR deviations: {sr_deviations}")
        # st.write(f"Total costs: {costs}")
        # st.write(f"Comfort components: {comfort_components}")
        # st.write(f"Average SR components: {avg_sr_components}")
        # st.write(f"SR deviation components: {sr_deviation_components}")
