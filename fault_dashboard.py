import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from datetime import datetime
import re
from streamlit_plotly_events import plotly_events

# Inject JavaScript to handle module fetch errors
components.html(
    """
    <script>
    window.addEventListener('error', function(e) {
        if (e.message.includes('Failed to fetch dynamically imported module')) {
            alert('Failed to load the app. Reloading the page...');
            window.location.reload();
        }
    });
    </script>
    """,
    height=0,
)

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

# Function to clean LOAD LOSS values
def clean_load_loss(value):
    if not isinstance(value, str) or pd.isna(value):
        return value
    value = re.sub(r'[oO]', '0', value)
    match = re.match(r'^\s*(\d*\.?\d*)\s*(?:amps?|a|A)?\s*$', value, re.IGNORECASE)
    if match:
        return match.group(1)
    return value

# Title and description
st.title("Ikeja Electric Monthly Fault Clearance Dashboard")
st.markdown("Upload an Excel file (sheet: '11kV Tripping Log') to analyze fault clearance operations across Business Units and Undertakings.")

# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="11kV Tripping Log", header=10)
    except ValueError as e:
        st.error(f"Error: Could not find sheet '11kV Tripping Log' in the uploaded file. ({str(e)})")
        st.stop()

    required_columns = [
        'BUSINESS UNIT', '11kV FEEDER', 'LOAD LOSS', 'DATE REPORTED', 'TIME REPORTED',
        'DATE CLEARED', 'TIME CLEARED', 'DATE RESTORED', 'TIME RESTORED', 'FAULT/OPERATION', 'PHASE AFFECTED'
    ]
    optional_columns = ['RESPONSIBLE UNDERTAKINGS', 'FINDINGS/ACTION TAKEN']
    missing_required = [col for col in required_columns if col not in df.columns]
    missing_optional = [col for col in optional_columns if col not in df.columns]
    if missing_required:
        st.error(f"Error: Missing required columns: {', '.join(missing_required)}.")
        st.stop()
    if missing_optional:
        st.warning(f"Missing optional columns: {', '.join(missing_optional)}.")

    df['LOAD LOSS'] = df['LOAD LOSS'].apply(clean_load_loss)
    date_time_cols = ['DATE REPORTED', 'TIME REPORTED', 'DATE CLEARED', 'TIME CLEARED', 'DATE RESTORED', 'TIME RESTORED']
    for col in date_time_cols:
        if col in df.columns:
            if 'DATE' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')

    df['REPORTED_TIMESTAMP'] = df['DATE REPORTED'] + pd.to_timedelta(df['TIME REPORTED'].dt.strftime('%H:%M:%S'))
    df['RESTORED_TIMESTAMP'] = df['DATE RESTORED'] + pd.to_timedelta(df['TIME RESTORED'].dt.strftime('%H:%M:%S'))
    df['CLEARED_TIMESTAMP'] = df['DATE CLEARED'] + pd.to_timedelta(df['TIME CLEARED'].dt.strftime('%H:%M:%S'))
    df['DOWNTIME_HOURS'] = (df['RESTORED_TIMESTAMP'] - df['REPORTED_TIMESTAMP']).dt.total_seconds() / 3600
    df['DOWNTIME_HOURS'] = df['DOWNTIME_HOURS'].abs()
    df['CLEARANCE_TIME_HOURS'] = (df['CLEARED_TIMESTAMP'] - df['REPORTED_TIMESTAMP']).dt.total_seconds() / 3600
    df['CLEARANCE_TIME_HOURS'] = df['CLEARANCE_TIME_HOURS'].abs()

    df['FAULT/OPERATION'] = df['FAULT/OPERATION'].astype(str).replace('nan', 'Unknown')
    initial_count = len(df)
    df = df[~df['FAULT/OPERATION'].str.lower().str.contains('opened on emergency|oe|emergency', na=False)]
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        st.info(f"Excluded {filtered_count} faults labeled 'Opened on Emergency', 'OE', or 'emergency'.")

    with st.expander("Debug Data (For Validation)"):
        st.write("**Columns in DataFrame**", df.columns.tolist())
        st.write("**LOAD LOSS**", df['LOAD LOSS'].head().to_list())
        st.write("Data type:", df['LOAD LOSS'].dtype)
        st.write("Non-numeric values:", df['LOAD LOSS'].apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x)).sum())
        st.write("**FAULT/OPERATION**", df['FAULT/OPERATION'].head().to_list())
        st.write("**PHASE AFFECTED**", df['PHASE AFFECTED'].head().to_list())
        st.write("**11kV FEEDER**", df['11kV FEEDER'].head().to_list())
        st.write("**BUSINESS UNIT**", df['BUSINESS UNIT'].head().to_list())
        if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
            st.write("**RESPONSIBLE UNDERTAKINGS**", df['RESPONSIBLE UNDERTAKINGS'].head().to_list())
        if 'FINDINGS/ACTION TAKEN' in df.columns:
            st.write("**FINDINGS/ACTION TAKEN**", df['FINDINGS/ACTION TAKEN'].head().to_list())

    df['LOAD LOSS'] = pd.to_numeric(df['LOAD LOSS'], errors='coerce')

    def extract_voltage(feeder_name):
        if not isinstance(feeder_name, str) or pd.isna(feeder_name):
            return None
        try:
            voltage_kv = float(feeder_name.split('-')[0].strip())
            return voltage_kv * 1000
        except (ValueError, IndexError):
            return None

    df['VOLTAGE_V'] = df['11kV FEEDER'].apply(extract_voltage)

    if df['LOAD LOSS'].isna().sum() > 0:
        st.warning(f"Found {df['LOAD LOSS'].isna().sum()} non-numeric or missing values in LOAD LOSS.")
    if df['VOLTAGE_V'].isna().sum() > 0:
        st.warning(f"Found {df['VOLTAGE_V'].isna().sum()} invalid voltage values.")
    if df['CLEARANCE_TIME_HOURS'].isna().sum() > 0:
        st.warning(f"Found {df['CLEARANCE_TIME_HOURS'].isna().sum()} invalid clearance times.")
    if df['BUSINESS UNIT'].isna().sum() > 0:
        st.warning(f"Found {df['BUSINESS UNIT'].isna().sum()} missing values in BUSINESS UNIT.")
    if 'RESPONSIBLE UNDERTAKINGS' in df.columns and df['RESPONSIBLE UNDERTAKINGS'].isna().sum() > 0:
        st.warning(f"Found {df['RESPONSIBLE UNDERTAKINGS'].isna().sum()} missing values in RESPONSIBLE UNDERTAKINGS.")
    if 'FINDINGS/ACTION TAKEN' in df.columns and df['FINDINGS/ACTION TAKEN'].isna().sum() > 0:
        st.warning(f"Found {df['FINDINGS/ACTION TAKEN'].isna().sum()} missing values in FINDINGS/ACTION TAKEN.")

    if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        def split_undertakings(undertakings):
            if not isinstance(undertakings, str) or pd.isna(undertakings):
                return []
            undertakings = re.sub(r'\s*(?:AND|&|and)\s*', '/', undertakings, flags=re.IGNORECASE)
            return [u.strip().lower() for u in undertakings.split('/') if u.strip()]
        df['UNDERTAKINGS_LIST'] = df['RESPONSIBLE UNDERTAKINGS'].apply(split_undertakings)

    df['ENERGY_LOSS_WH'] = df['LOAD LOSS'] * df['VOLTAGE_V'] * df['DOWNTIME_HOURS']
    df['ENERGY_LOSS_MWH'] = df['ENERGY_LOSS_WH'] / 1_000_000
    df['MONETARY_LOSS_NGN_MILLIONS'] = (df['ENERGY_LOSS_MWH'] * 1000 * 209.5) / 1_000_000

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

    feeder_trips = df['11kV FEEDER'].value_counts().reset_index()
    feeder_trips.columns = ['11kV FEEDER', 'TRIP_COUNT']
    frequent_trippers = feeder_trips[feeder_trips['TRIP_COUNT'] > 2]

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

    df['PRIORITY_SCORE'] = (
        0.4 * (df['CLEARANCE_TIME_HOURS'] / df['CLEARANCE_TIME_HOURS'].max()) +
        0.4 * (df['ENERGY_LOSS_MWH'] / df['ENERGY_LOSS_MWH'].max()) +
        0.2 * df['11kV FEEDER'].map(feeder_trips.set_index('11kV FEEDER')['TRIP_COUNT']) / feeder_trips['TRIP_COUNT'].max()
    ).fillna(0)

    outlier_columns = ['11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS']
    if 'FINDINGS/ACTION TAKEN' in df.columns:
        outlier_columns.append('FINDINGS/ACTION TAKEN')
    clearance_outliers = df[df['CLEARANCE_TIME_HOURS'] > 48][outlier_columns]
    clearance_outliers['CLEARANCE_TIME_HOURS'] = clearance_outliers['CLEARANCE_TIME_HOURS'].round(2)

    st.subheader("Filter Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bu_filter = st.selectbox("Select Business Unit", ['All'] + sorted(df['BUSINESS UNIT'].unique().astype(str).tolist()))
    with col2:
        if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
            all_undertakings = set()
            for undertakings in df['UNDERTAKINGS_LIST']:
                all_undertakings.update(undertakings)
            undertaking_options = sorted(list(all_undertakings))
            undertaking_filter = st.multiselect("Select Responsible Undertakings", undertaking_options, default=undertaking_options)
        else:
            undertaking_filter = []
            st.write("Responsible Undertakings filter disabled.")
    with col3:
        feeder_filter = st.selectbox("Select Feeder", ['All'] + sorted(df['11kV FEEDER'].unique().astype(str).tolist()))
    with col4:
        df['YEAR'] = df['DATE REPORTED'].dt.year
        year_filter = st.selectbox("Select Year", ['All'] + sorted(df['YEAR'].unique().tolist()))
        df['MONTH'] = df['DATE REPORTED'].dt.month
        month_filter = st.selectbox("Select Month", ['All'] + sorted(df['MONTH'].unique().tolist()))

    filtered_df = df
    if bu_filter != 'All':
        filtered_df = filtered_df[filtered_df['BUSINESS UNIT'].astype(str) == bu_filter]
    if undertaking_filter and 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        filtered_df = filtered_df[filtered_df['UNDERTAKINGS_LIST'].apply(lambda x: any(u in undertaking_filter for u in x))]
    if feeder_filter != 'All':
        filtered_df = filtered_df[filtered_df['11kV FEEDER'].astype(str) == feeder_filter]
    if year_filter != 'All':
        filtered_df = filtered_df[filtered_df['YEAR'] == year_filter]
    if month_filter != 'All':
        filtered_df = filtered_df[filtered_df['MONTH'] == month_filter]

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    fault_counts_filtered = filtered_df['FAULT_TYPE'].value_counts().reset_index()
    fault_counts_filtered.columns = ['Fault Type', 'Count']
    feeder_downtime_filtered = filtered_df.groupby('11kV FEEDER')['DOWNTIME_HOURS'].mean().reset_index()
    feeder_downtime_filtered = calculate_feeder_ratings(feeder_downtime_filtered)
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
    outlier_columns_filtered = ['11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS']
    if 'FINDINGS/ACTION TAKEN' in df.columns:
        outlier_columns_filtered.append('FINDINGS/ACTION TAKEN')
    clearance_outliers_filtered = filtered_df[filtered_df['CLEARANCE_TIME_HOURS'] > 48][outlier_columns_filtered]
    clearance_outliers_filtered['CLEARANCE_TIME_HOURS'] = clearance_outliers_filtered['CLEARANCE_TIME_HOURS'].round(2)

    detail_cols = ['DATE REPORTED', '11kV FEEDER', 'FAULT/OPERATION', 'FAULT_TYPE', 'DOWNTIME_HOURS', 'CLEARANCE_TIME_HOURS', 'ENERGY_LOSS_MWH']
    if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        detail_cols.insert(2, 'RESPONSIBLE UNDERTAKINGS')
    if 'FINDINGS/ACTION TAKEN' in df.columns:
        detail_cols.append('FINDINGS/ACTION TAKEN')

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

    if not clearance_outliers_filtered.empty:
        st.warning(f"Critical: {len(clearance_outliers_filtered)} faults took over 48 hours to clear.")
    if not frequent_trippers_filtered.empty:
        st.warning(f"Critical: {len(frequent_trippers_filtered)} feeders tripped more than twice.")

    st.subheader("Fault Classification")
    fig_faults = px.bar(fault_counts_filtered, x='Fault Type', y='Count', title="Fault Types Distribution")
    selected_fault = plotly_events(fig_faults, key="fault_click", click_event=True)
    st.plotly_chart(fig_faults, use_container_width=True)
    if selected_fault:
        fault_type = selected_fault[0]['x']
        with st.expander(f"Details for Fault Type: {fault_type}"):
            filtered_details = filtered_df[filtered_df['FAULT_TYPE'] == fault_type][detail_cols].copy()
            filtered_details['DOWNTIME_HOURS'] = filtered_details['DOWNTIME_HOURS'].round(2)
            filtered_details['CLEARANCE_TIME_HOURS'] = filtered_details['CLEARANCE_TIME_HOURS'].round(2)
            filtered_details['ENERGY_LOSS_MWH'] = filtered_details['ENERGY_LOSS_MWH'].round(2)
            st.dataframe(filtered_details)

    st.subheader("Daily Fault Trend")
    daily_faults = filtered_df.groupby(filtered_df['DATE REPORTED'].dt.date).size().reset_index(name='Fault Count')
    daily_faults['Date'] = pd.to_datetime(daily_faults['Date'])
    fig_trend = px.line(daily_faults, x='Date', y='Fault Count', title="Daily Fault Trend")
    selected_date = plotly_events(fig_trend, key="trend_click", click_event=True)
    st.plotly_chart(fig_trend, use_container_width=True)
    if selected_date:
        date_str = selected_date[0]['x']
        with st.expander(f"Details for Date: {date_str}"):
            filtered_details = filtered_df[filtered_df['DATE REPORTED'].dt.date == pd.to_datetime(date_str).date()][detail_cols].copy()
            filtered_details['DOWNTIME_HOURS'] = filtered_details['DOWNTIME_HOURS'].round(2)
            filtered_details['CLEARANCE_TIME_HOURS'] = filtered_details['CLEARANCE_TIME_HOURS'].round(2)
            filtered_details['ENERGY_LOSS_MWH'] = filtered_details['ENERGY_LOSS_MWH'].round(2)
            st.dataframe(filtered_details)

    st.subheader("Average Downtime by Feeder")
    if feeder_downtime_filtered.empty:
        st.warning("No feeder data available for the selected filters.")
    else:
        feeder_options = sorted(feeder_downtime_filtered['SHORT_FEEDER_NAME'].unique())
        selected_feeders = st.multiselect(
            "Select Feeders to Display (uses short names)",
            options=feeder_options,
            default=feeder_options,
            help="Choose one or more feeders to compare their average downtime."
        )
        if selected_feeders:
            chart_data = feeder_downtime_filtered[feeder_downtime_filtered['SHORT_FEEDER_NAME'].isin(selected_feeders)]
        else:
            chart_data = feeder_downtime_filtered
            st.warning("No feeders selected. Displaying all feeders.")
        rating_colors = {'Excellent': '#006400', 'Good': '#32CD32', 'Fair': '#FFFF00', 'Poor': '#FF0000'}
        fig_downtime = px.bar(
            chart_data,
            x='SHORT_FEEDER_NAME',
            y='DOWNTIME_HOURS',
            color='RATING',
            title="Average Downtime by Feeder",
            color_discrete_map=rating_colors
        )
        selected_feeder = plotly_events(fig_downtime, key="downtime_click", click_event=True)
        st.plotly_chart(fig_downtime, use_container_width=True)
        if selected_feeder:
            short_name = selected_feeder[0]['x']
            full_feeder = chart_data[chart_data['SHORT_FEEDER_NAME'] == short_name]['11kV FEEDER'].iloc[0]
            with st.expander(f"Details for Feeder: {short_name} ({full_feeder})"):
                filtered_details = filtered_df[filtered_df['11kV FEEDER'] == full_feeder][detail_cols].copy()
                filtered_details['DOWNTIME_HOURS'] = filtered_details['DOWNTIME_HOURS'].round(2)
                filtered_details['CLEARANCE_TIME_HOURS'] = filtered_details['CLEARANCE_TIME_HOURS'].round(2)
                filtered_details['ENERGY_LOSS_MWH'] = filtered_details['ENERGY_LOSS_MWH'].round(2)
                st.dataframe(filtered_details)

    st.subheader("Frequent Tripping Feeders (>2 Trips)")
    st.dataframe(frequent_trippers_filtered)

    st.subheader("Maintenance Suggestions")
    cols_to_show = ['BUSINESS UNIT', '11kV FEEDER', 'FAULT/OPERATION', 'FAULT_TYPE', 'MAINTENANCE_SUGGESTION']
    if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        cols_to_show.insert(1, 'RESPONSIBLE UNDERTAKINGS')
    st.dataframe(filtered_df[cols_to_show].drop_duplicates())

    st.subheader("High-Priority Faults")
    cols_to_show = ['BUSINESS UNIT', '11kV FEEDER', 'FAULT_TYPE', 'CLEARANCE_TIME_HOURS', 'ENERGY_LOSS_MWH', 'PRIORITY_SCORE']
    if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        cols_to_show.insert(1, 'RESPONSIBLE UNDERTAKINGS')
    st.dataframe(filtered_df[cols_to_show].sort_values('PRIORITY_SCORE', ascending=False).head(10))

    st.subheader("Additional Insights")
    st.write("1. **Fault Clearance Time Distribution (0-48 Hours)**")
    filtered_clearance = filtered_df[(filtered_df['CLEARANCE_TIME_HOURS'] >= 0) & (filtered_df['CLEARANCE_TIME_HOURS'] <= 48)]
    fig_clearance = px.histogram(filtered_clearance, x='CLEARANCE_TIME_HOURS', nbins=24, title="Fault Clearance Time Distribution (0-48 Hours)")
    st.plotly_chart(fig_clearance, use_container_width=True)

    st.write("2. **Clearance Time Outliers (>48 Hours)**")
    st.dataframe(clearance_outliers_filtered)

    st.write("3. **Phase-Specific Faults**")
    fig_phase = px.bar(phase_faults_filtered, x='Phase Affected', y='Fault Count', title="Faults by Phase Affected")
    selected_phase = plotly_events(fig_phase, key="phase_click", click_event=True)
    st.plotly_chart(fig_phase, use_container_width=True)
    if selected_phase:
        phase = selected_phase[0]['x']
        with st.expander(f"Details for Phase Affected: {phase}"):
            filtered_details = filtered_df[filtered_df['PHASE_AFFECTED_LIST'].apply(lambda x: phase in x)][detail_cols].copy()
            filtered_details['DOWNTIME_HOURS'] = filtered_details['DOWNTIME_HOURS'].round(2)
            filtered_details['CLEARANCE_TIME_HOURS'] = filtered_details['CLEARANCE_TIME_HOURS'].round(2)
            filtered_details['ENERGY_LOSS_MWH'] = filtered_details['ENERGY_LOSS_MWH'].round(2)
            st.dataframe(filtered_details)

    st.subheader("Download Report")
    cols_to_export = ['BUSINESS UNIT', '11kV FEEDER', 'FAULT/OPERATION', 'FAULT_TYPE', 'ENERGY_LOSS_MWH', 'MONETARY_LOSS_NGN_MILLIONS', 'DOWNTIME_HOURS', 'CLEARANCE_TIME_HOURS', 'MAINTENANCE_SUGGESTION', 'PRIORITY_SCORE']
    if 'RESPONSIBLE UNDERTAKINGS' in df.columns:
        cols_to_export.insert(1, 'RESPONSIBLE UNDERTAKINGS')
    if 'FINDINGS/ACTION TAKEN' in df.columns:
        cols_to_export.append('FINDINGS/ACTION TAKEN')
    report_df = filtered_df[cols_to_export]
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
