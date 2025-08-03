Ikeja Electric Fault Clearance Dashboard
Overview
This dashboard provides a comprehensive analysis of fault clearance operations for Ikeja Electric, designed for technical team supervisors and management. It visualizes key metrics such as total faults, energy and monetary losses, downtime, fault types, and phase-specific faults. It also identifies high-priority faults and clearance time outliers (>48 hours) to guide maintenance decisions.
Features

Key Metrics: Total faults, energy loss (MWh), monetary loss (NGN), and downtime (hours).
Fault Classification: Categorizes faults (e.g., Earth Fault, Over Current, Broken Conductor, Emergency).
Feeder Analysis: Identifies frequent tripping feeders and average downtime with ratings.
Clearance Time Analysis: Histogram for 0-48 hours and table for outliers (>48 hours).
Phase Analysis: Counts faults by phase (Red, Yellow, Blue, Earth Fault, Neutral).
Maintenance Suggestions: Actionable recommendations based on fault types.
Dynamic Filtering: Filter by feeder for focused analysis.
Trend Analysis: Daily fault trends over time.
Priority Scoring: Ranks faults by clearance time, energy loss, and trip frequency.
Exportable Report: CSV download for sharing with stakeholders.

Prerequisites

Python 3.8 or higher
Git (for cloning the repository)
A GitHub account (to access the repository)
An Excel file with fault data (see Data Requirements)

Installation

Clone the repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Verify installation:
streamlit run fault_dashboard.py

This opens the dashboard in your default web browser.


Usage

Run the dashboard:
streamlit run fault_dashboard.py


Upload an Excel file via the dashboard interface.

Use the feeder filter to focus on specific feeders or view all data.

Explore sections:

Key Metrics: Overview of faults, losses, and downtime.
Total Downtime: Total hours of downtime.
Fault Classification: Distribution of fault types (e.g., Emergency, Earth Fault).
Daily Fault Trend: Fault counts over time.
Total Downtime by Feeder: Average downtime with ratings.
Frequent Tripping Feeders: Feeders with >2 trips.
Maintenance Suggestions: Recommendations for each fault.
High-Priority Faults: Top 10 faults by priority score.
Additional Insights: Clearance time distribution (0-48 hours), outliers (>48 hours), and phase-specific faults.


Download the CSV report for documentation or sharing.


Data Requirements
The Excel file must contain the following columns:

DATE REPORTED: Date of fault report (e.g., YYYY-MM-DD).
TIME REPORTED: Time of fault report (e.g., HH:MM:SS).
DATE CLEARED: Date fault was cleared.
TIME CLEARED: Time fault was cleared.
DATE RESTORED: Date power was restored.
TIME RESTORED: Time power was restored.
LOAD LOSS: Power loss in MW (numeric).
FAULT/OPERATION: Fault description (e.g., "E/F on Red", "Emergency O/C").
PHASE AFFECTED: Affected phases (e.g., "Red, Yellow", "Blue & Earth Fault").
11kV FEEDER: Feeder name.
BUSINESS UNIT: Business unit name (single unit expected).
WHO REPORTED: Reporter or area (optional, less critical).

Notes

Ensure date/time columns are in valid formats to avoid NaN values.
LOAD LOSS must be numeric (MW) or will be treated as NaN.
FAULT/OPERATION should use consistent terms (e.g., "E/F", "O/C", "B/C", "Emergency").
PHASE AFFECTED may contain multiple phases separated by "AND", "&", or ",".

Troubleshooting

Empty histogram: Check if CLEARANCE_TIME_HOURS has valid values (0-48 hours). Verify date/time columns for errors.
Missing outliers: Ensure DATE CLEARED and TIME CLEARED are correctly formatted.
NaN in metrics: Check LOAD LOSS for non-numeric values or invalid date/time entries.
Debug outputs: Expand the "Debug Data" section in the dashboard to inspect column values and types.
Create a GitHub issue for support.

Contributing

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. 
Contact
elvisebenuwah@gmail.com