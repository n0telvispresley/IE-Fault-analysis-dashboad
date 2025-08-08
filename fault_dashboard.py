import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# Streamlit page configuration
st.set_page_config(page_title="Ikeja Electric Fault Clearance Dashboard", layout="wide")

# Custom function for comma-separated number formatting
def format_number(value, decimals=2):
    if pd.isna(value):
        return "NaN"
    try:
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return "NaN"

# Title and description
st.title("Ikeja Electric Monthly Fault Clearance Dashboard")
st.markdown("Upload an Excel file to analyze fault clearance operations, including fault classification, energy/monetary losses, downtime, and more.")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read Excel file
    df = pd.read_excel(uploaded_file)

    # Ensure date and time columns are in datetime format
    date_time_cols = ['DATE REPORTED', 'TIME REPORTED', 'DATE CLEARED', 'TIME CLEARED', 'DATE RESTORED', 'TIME RESTORED']
    for col in date_time_cols:
        if col in df.columns:
            if 'DATE' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')

    # Combine date and time for full timestamps
    df['REPORTED_TIMESTAMP'] = df['DATE REPORTED'] + pd.to_timedelta(df['TIME REPORTED'].dt.strftime('%H:%M:%S'))
    df['RESTORED_TIMESTAMP'] = df['DATE RESTORED'] + pd.to_timedelta(df['TIME RESTORED'].dt.strftime('%H:%M:%S'))
    df['CLEARED_TIMESTAMP'] = df['DATE CLEARED'] + pd.to_timedelta(df['TIME CLEARED'].dt.strftime('%H:%M:%S'))

    # Calculate downtime (in hours) and ensure positive
    df['DOWNTIME_HOURS'] = (df['RESTORED_TIMESTAMP'] - df['REPORTED_TIMESTAMP']).dt.total_seconds() / 3600
    df['DOWNTIME_HOURS'] = df['DOWNTIME_HOURS'].abs()

    # Calculate clearance time (in hours) and ensure positive
    df['CLEARANCE_TIME_HOURS'] = (df['CLEARED_TIMESTAMP'] - df['REPORTED_TIMESTAMP']).dt.total_seconds() / 3600
    df['CLEARANCE_TIME_HOURS'] = df['CLEARANCE_TIME_HOURS'].abs()

    # Filter out faults labeled "Opened on Emergency", "OE", or "emergency"
    df['FAULT/OPERATION'] = df['FAULT/OPERATION'].astype(str).replace('nan', 'Unknown')
    initial_count = len(df)
    df = df[~df['FAULT/OPERATION'].str.lower().str.contains('opened on emergency|oe|emergency', na=False)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        st.info(f"Excluded {filtered_count} faults labeled 'Opened on Emergency', 'OE', or 'emergency' as they are handled by CHQ.")

    # Debug outputs in expander
    with st.expander("Debug Data (For Validation)"):
        st.write("**LOAD LOSS (Current in Amps) Column**")
        st.write("Sample values:", df['LOAD LOSS'].head().to_list())
        st.write("Data type:", df['LOAD LOSS'].dtype)
        st.write("Any non-numeric values:", df['LOAD LOSS'].apply(lambda x: not isinstance(x, (int, float))).sum())
        st.write("**FAULT/OPERATION Column**")
        st.write("Sample values:", df['FAULT/OPERATION'].head().to_list())
        st.write("Data type:", df['FAULT/OPERATION'].dtype)
        st.write("Any non-string values:", df['FAULT/OPERATION'].apply(lambda x: not isinstance(x, str) and pd.notna(x)).sum())
        st.write("**PHASE AFFECTED Column**")
        st.write("Sample values:", df['PHASE AFFECTED'].head().to_list())
        st.write("Data type:", df['PHASE AFFECTED'].dtype)
        st.write("**11kV FEEDER Column**")
        st.write("Sample values:", df['11kV FEEDER'].head().to_list())

    # Convert LOAD LOSS (current in amps) to numeric, coercing invalid values to NaN
    df['LOAD LOSS'] = pd.to_numeric(df['LOAD LOSS'], errors='coerce')

    # Extract voltage (11 kV or 33 kV) from the first part of 11kV FEEDER
    def extract_voltage(feeder_name):
        if not isinstance(feeder_name, str) or pd.isna(feeder_name):
            return None
        try:
            # Get part before first hyphen and convert to float (expecting 11 or 33)
            voltage_kv = float(feeder_name.split('-')[0].strip())
            # Convert to volts (kV to V)
            return voltage_kv * 1000
        except (ValueError, IndexError):
            return None

    df['VOLTAGE_V'] = df['11kV FEEDER'].apply(extract_voltage)

    # Check for NaN values in LOAD LOSS and VOLTAGE_V
    if df['LOAD LOSS'].isna().sum() > 0:
        st.warning(f"Found {df['LOAD LOSS'].isna().sum()} non-numeric or missing values in LOAD LOSS (current). These have been converted to NaN.")
    if df['VOLTAGE_V'].isna().sum() > 0:
        st.warning(f"Found {df['VOLTAGE_V'].isna().sum()} invalid voltage values derived from 11kV FEEDER. Ensure feeder names start with '11-' or '33-'. These have been converted to NaN.")
    if df['CLEARANCE_TIME_HOURS'].isna().sum() > 0:
        st.warning(f"Found {df['CLEARANCE_TIME_HOURS'].isna().sum()} invalid clearance times due to missing or incorrect date/time data.")

    # Calculate energy loss using E = I * V * t (in watt-hours, then convert to MWh)
    # E (Wh) = Current (A) * Voltage (V) * Time (h)
    df['ENERGY_LOSS_WH'] = df['LOAD LOSS'] * df['VOLTAGE_V'] * df['DOWNTIME_HOURS']
    # Convert watt-hours to megawatt-hours (1 MWh = 1,000,000 Wh)
    df['ENERGY_LOSS_MWH'] = df['ENERGY_LOSS_WH'] / 1_000_000

    # Monetary loss (using 209.5 NGN/kWh for Band A feeders, convert to millions)
    df['MONETARY_LOSS_NGN_MILLIONS'] = (df['ENERGY_LOSS_MWH'] * 1000 * 209.5) / 1_000_000

    # Fault classification
    def classify_fault(fault_str):
        if not isinstance(fault_str, str) or fault_str == 'Unknown':
            return 'Unknown'
        fault_str = fault_str.lower()
        if 'e/f' in fault_str or 'earth fault' in fault_str:
            return 'Earth Fault'
        elif 'o/c' in fault_str or 'over current' in fault_str:
            return 'Over Current'
        elif 'b/c' in fault_str or 'broken conductor' in fault_str:
            return 'Broken Conductor'
        elif 'cable' in fault_str:
            return 'Cable Fault'
        elif 'transformer' in fault_str:
            return 'Transformer Fault'
        elif 'breaker' in fault_str:
            return 'Breaker Fault'
        else:
            return 'Other'

    df['FAULT_TYPE'] = df['FAULT/OPERATION'].apply(classify_fault)

    # Feeder ratings (based on average downtime, lower is better)
    def calculate_feeder_ratings(df_grouped):
        if len(df_grouped) < 2 or df_grouped['DOWNTIME_HOURS'].nunique() < 2:
            df_grouped['RATING'] = 'Unknown'
        else:
            try:
                n_bins = min(4, df_grouped['DOWNTIME_HOURS'].nunique())
                df_grouped['RATING'] = pd.qcut(df_grouped['DOWNTIME_HOURS'], q=n_bins, labels=['Excellent', 'Good', 'Fair', 'Poor'][:n_bins], duplicates='drop')
            except ValueError:
                df_grouped['RATING'] = 'Unknown'
        return df_grouped

    feeder_downtime = df.groupby('11kV FEEDER')['DOWNTIME_HOURS'].mean().reset_index()
    feeder_downtime = calculate_feeder_ratings(feeder_downtime)

    # Frequent tripping feeders (more than 2 trips)
    feeder_trips = df['11kV FEEDER'].value_counts().reset_index()
    feeder_trips.columns = ['11kV FEEDER', 'TRIP_COUNT']
    frequent_trippers = feeder_trips[feeder_trips['TRIP_COUNT'] > 2]

    # Maintenance suggestions
    def suggest_maintenance(fault_type):
        if not isinstance(fault_type, str) or fault_type == 'Unknown':
            return "Conduct root cause analysis for unspecified fault."
        fault_type = fault_type.lower()
        if 'e/f' in fault_type or 'earth fault' in fault_type:
            return "Inspect grounding systems and insulation; repair earth faults."
        elif 'o/c' in fault_type or 'over current' in fault_type:
            return "Check for overloads and short circuits; calibrate protective devices."
        elif 'b/c' in fault_type or 'broken conductor' in fault_type:
            return "Replace broken conductors; inspect for physical damage."
        elif 'cable' in fault_type:
            return "Inspect and replace damaged cables; check for water ingress."
        elif 'transformer' in fault_type:
            return "Schedule transformer maintenance; check oil levels and insulation."
        elif 'breaker' in fault_type:
            return "Test and calibrate circuit breakers; inspect for wear."
        else:
            return "Conduct root cause analysis and regular preventive maintenance."

    df['MAINTENANCE_SUGGESTION'] = df['FAULT/OPERATION'].apply(suggest_maintenance)

    # Phase-specific fault analysis
    def parse_phases(phase_str):
        if not isinstance(phase_str, str) or pd.isna(phase_str) or phase_str.lower() == 'nan':
            return []
        phase_str = re.sub(r'\s*AND\s*|\s*&\s*|,\s*', ',', phase_str, flags=re.IGNORECASE)
        phases = [p.strip().lower() for p in phase_str.split(',') if p.strip()]
        return phases

    df['PHASE_AFFECTED_LIST'] = df['PHASE AFFECTED'].apply(parse_phases)
    phase_types = ['red', 'yellow', 'blue', 'earth fault', 'neutral']
    phase_counts = {phase: 0 for phase in phase_types}
    for phases in df['PHASE_AFFECTED_LIST']:
        for phase in phases:
            if phase in phase_counts:
                phase_counts[phase] += 1
    phase_faults = pd.DataFrame({
        'Phase Affected': phase_counts.keys(),
        'Fault Count': phase_counts.values()
    })
    phase_faults = phase_faults[phase_faults['Fault Count'] > 0]

    # Maintenance priority score
    df['PRIORITY_SCORE'] = (
        0.4 * (df['CLEARANCE_TIME_HOURS'] / df['CLEARANCE_TIME_HOURS'].max()) +
        0.4 * (df['ENERGY_LOSS_MWH'] / df['ENERGY_LOSS_MWH'].max()) +
        0.2 * df['11kV FEEDER'].map(feeder_trips.set_index('11kV FEEDER')['TRIP_COUNT']) / feeder_trips['TRIP_COUNT'].max()
    ).fillna(0)

    # Identify clearance time outliers (>48 hours)
    clearance_outliers = df[df['CLEARANCE_TIME_HOURS'] > 48][['11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS']]
    clearance_outliers['CLEARANCE_TIME_HOURS'] = clearance_outliers['CLEARANCE_TIME_HOURS'].round(2)

    # Dynamic filters
    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        feeder_filter = st.selectbox("Select Feeder", ['All'] + list(df['11kV FEEDER'].unique()))
    with col2:
        df['YEAR'] = df['DATE REPORTED'].dt.year
        year_filter = st.selectbox("Select Year", ['All'] + sorted(df['YEAR'].unique().tolist()))
    with col3:
        df['MONTH'] = df['DATE REPORTED'].dt.month
        month_filter = st.selectbox("Select Month", ['All'] + sorted(df['MONTH'].unique().tolist()))

    # Apply filters
    filtered_df = df
    if feeder_filter != 'All':
        filtered_df = filtered_df[filtered_df['11kV FEEDER'] == feeder_filter]
    if year_filter != 'All':
        filtered_df = filtered_df[filtered_df['YEAR'] == year_filter]
    if month_filter != 'All':
        filtered_df = filtered_df[filtered_df['MONTH'] == month_filter]

    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Try different options.")
        st.stop()

    # Update metrics and visuals based on filtered data
    fault_counts_filtered = filtered_df['FAULT_TYPE'].value_counts().reset_index()
    fault_counts_filtered.columns = ['Fault Type', 'Count']
    feeder_downtime_filtered = filtered_df.groupby('11kV FEEDER')['DOWNTIME_HOURS'].mean().reset_index()
    feeder_downtime_filtered = calculate_feeder_ratings(feeder_downtime_filtered)
    # Add short feeder name for display
    def get_short_feeder_name(feeder_name):
        if not isinstance(feeder_name, str) or pd.isna(feeder_name):
            return "Unknown"
        return feeder_name.split('-')[-1].strip()
    feeder_downtime_filtered['SHORT_FEEDER_NAME'] = feeder_downtime_filtered['11kV FEEDER'].apply(get_short_feeder_name)
    frequent_trippers_filtered = filtered_df['11kV FEEDER'].value_counts().reset_index()
    frequent_trippers_filtered.columns = ['11kV FEEDER', 'TRIP_COUNT']
    frequent_trippers_filtered = frequent_trippers_filtered[frequent_trippers_filtered['TRIP_COUNT'] > 2]
    phase_counts_filtered = {phase: 0 for phase in phase_types}
    for phases in filtered_df['PHASE_AFFECTED_LIST']:
        for phase in phases:
            if phase in phase_counts_filtered:
                phase_counts_filtered[phase] += 1
    phase_faults_filtered = pd.DataFrame({
        'Phase Affected': phase_counts_filtered.keys(),
        'Fault Count': phase_counts_filtered.values()
    })
    phase_faults_filtered = phase_faults_filtered[phase_faults_filtered['Fault Count'] > 0]
    clearance_outliers_filtered = filtered_df[filtered_df['CLEARANCE_TIME_HOURS'] > 48][['11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS']]
    clearance_outliers_filtered['CLEARANCE_TIME_HOURS'] = clearance_outliers_filtered['CLEARANCE_TIME_HOURS'].round(2)

    # Dashboard layout
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Faults", format_number(len(filtered_df), decimals=0))
    with col2:
        st.metric("Total Energy Loss (MWh)", format_number(filtered_df['ENERGY_LOSS_MWH'].sum(), decimals=2))
    with col3:
        st.metric("Total Monetary Loss (M NGN)", format_number(filtered_df['MONETARY_LOSS_NGN_MILLIONS'].sum(), decimals=2))

    st.subheader("Total Downtime")
    st.metric("Total Downtime (Hours)", format_number(filtered_df['DOWNTIME_HOURS'].sum(), decimals=2))

    # Alerts for outliers
    if not clearance_outliers_filtered.empty:
        st.warning(f"Critical: {len(clearance_outliers_filtered)} faults took over 48 hours to clear. Review outliers table.")
    if not frequent_trippers_filtered.empty:
        st.warning(f"Critical: {len(frequent_trippers_filtered)} feeders tripped more than twice. Review frequent trippers table.")

    st.subheader("Fault Classification")
    fig_faults = px.bar(fault_counts_filtered, x='Fault Type', y='Count', title="Fault Types Distribution")
    st.plotly_chart(fig_faults, use_container_width=True)

    st.subheader("Daily Fault Trend")
    daily_faults = filtered_df.groupby(filtered_df['DATE REPORTED'].dt.date)['FAULT_TYPE'].count().reset_index()
    daily_faults.columns = ['Date', 'Fault Count']
    fig_trend = px.line(daily_faults, x='Date', y='Fault Count', title="Daily Fault Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Average Downtime by Feeder")
    # Define custom colors for RATING categories
    rating_colors = {
        'Excellent': '#006400',  # Dark Green
        'Good': '#32CD32',      # Lemon Green
        'Fair': '#FFFF00',       # Yellow
        'Poor': '#FF0000'        # Red
    }
    fig_downtime = px.bar(
        feeder_downtime_filtered,
        x='SHORT_FEEDER_NAME',
        y='DOWNTIME_HOURS',
        color='RATING',
        title="Average Downtime by Feeder",
        color_discrete_map=rating_colors
    )
    st.plotly_chart(fig_downtime, use_container_width=True)

    st.subheader("Frequent Tripping Feeders (>2 Trips)")
    st.dataframe(frequent_trippers_filtered)

    st.subheader("Maintenance Suggestions")
    st.dataframe(filtered_df[['11kV FEEDER', 'FAULT/OPERATION', 'FAULT_TYPE', 'MAINTENANCE_SUGGESTION']].drop_duplicates())

    st.subheader("High-Priority Faults")
    st.dataframe(filtered_df[['11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS', 'ENERGY_LOSS_MWH', 'PRIORITY_SCORE']].sort_values('PRIORITY_SCORE', ascending=False).head(10))

    st.subheader("Additional Insights")
    st.write("1. **Fault Clearance Time Distribution (0-48 Hours)**: Distribution of clearance times up to 48 hours.")
    filtered_clearance = filtered_df[(filtered_df['CLEARANCE_TIME_HOURS'] >= 0) & (filtered_df['CLEARANCE_TIME_HOURS'] <= 48)]
    fig_clearance = px.histogram(filtered_clearance, x='CLEARANCE_TIME_HOURS', nbins=24, title="Fault Clearance Time Distribution (0-48 Hours)")
    st.plotly_chart(fig_clearance, use_container_width=True)

    st.write("2. **Clearance Time Outliers (>48 Hours)**: Faults taking more than 48 hours to clear.")
    st.dataframe(clearance_outliers_filtered)

    st.write("3. **Phase-Specific Faults**: Fault counts by affected phase.")
    fig_phase = px.bar(phase_faults_filtered, x='Phase Affected', y='Fault Count', title="Faults by Phase Affected")
    st.plotly_chart(fig_phase, use_container_width=True)

    # Export report as CSV
    st.subheader("Download Report")
    report_df = filtered_df[['BUSINESS UNIT', '11kV FEEDER', 'FAULT/OPERATION', 'FAULT_TYPE', 'ENERGY_LOSS_MWH', 'MONETARY_LOSS_NGN_MILLIONS', 'DOWNTIME_HOURS', 'CLEARANCE_TIME_HOURS', 'MAINTENANCE_SUGGESTION', 'PRIORITY_SCORE']]
    report_df = report_df.merge(feeder_downtime_filtered[['11kV FEEDER', 'RATING']], on='11kV FEEDER', how='left')
    report_df['ENERGY_LOSS_MWH'] = report_df['ENERGY_LOSS_MWH'].apply(lambda x: format_number(x, decimals=2) if pd.notnull(x) else "NaN")
    report_df['MONETARY_LOSS_NGN_MILLIONS'] = report_df['MONETARY_LOSS_NGN_MILLIONS'].apply(lambda x: format_number(x, decimals=2) if pd.notnull(x) else "NaN")
    report_df['DOWNTIME_HOURS'] = report_df['DOWNTIME_HOURS'].apply(lambda x: format_number(x, decimals=2) if pd.notnull(x) else "NaN")
    report_df['CLEARANCE_TIME_HOURS'] = report_df['CLEARANCE_TIME_HOURS'].apply(lambda x: format_number(x, decimals=2) if pd.notnull(x) else "NaN")
    report_df['PRIORITY_SCORE'] = report_df['PRIORITY_SCORE'].apply(lambda x: format_number(x, decimals=2) if pd.notnull(x) else "NaN")
    csv = report_df.to_csv(index=False)
    st.download_button("Download CSV Report", csv, "fault_clearance_report.csv", "text/csv")

else:
    st.warning("Please upload an Excel file to proceed.")
