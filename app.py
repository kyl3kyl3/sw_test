import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("SW Test Results")
st.subheader("Note that metric percentage deltas around the start of the proceeding month are excluded due to expected seasonal drops")
# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('COMBINED_TEST_DATA.csv')

data = load_data()

# Sidebar for user input filters
st.sidebar.header('Filters')
datasource_filter = st.sidebar.selectbox('Select DATASOURCE', ['All'] + sorted(data['DATASOURCE'].unique().tolist()))
device_filter = st.sidebar.selectbox('Select DEVICE', ['All'] + sorted(data['DEVICE'].unique().tolist()))
domain_filter = st.sidebar.selectbox('Select DOMAIN', ['All'] + sorted(data['DOMAIN'].unique().tolist()))

# Filtering the data
if datasource_filter != "All":
    data = data[data['DATASOURCE'] == datasource_filter]
if device_filter != "All":
    data = data[data['DEVICE'] == device_filter]
if domain_filter != "All":
    data = data[data['DOMAIN'] == domain_filter]

# Define test implementation start date
implementation_date = pd.to_datetime("2023-07-21")
data['DATE'] = pd.to_datetime(data['DATE'])
start_date = '2023-07-22'
# Group and calculate metrics
grouped = data.groupby('DATE').agg({
    'REVENUE': 'sum',
    'PAGEVIEWS': 'sum',
    'IMPRESSIONS': 'sum',
    'ADREQUESTS': 'sum',
    'VIEWABLEIMPRESSIONS': 'sum'
}).reset_index()
grouped['RPM'] = grouped['REVENUE'] / (grouped['PAGEVIEWS'] / 1000)
grouped['CPM'] = grouped['REVENUE'] / (grouped['IMPRESSIONS'] / 1000)
grouped['ADREQUESTS_PER_PAGEVIEW'] = grouped['ADREQUESTS'] / grouped['PAGEVIEWS']
grouped['VIEWABILITY'] = grouped['VIEWABLEIMPRESSIONS'] / grouped['IMPRESSIONS']

# Time series decomposition and visualization
metrics = ['REVENUE', 'PAGEVIEWS', 'IMPRESSIONS', 'ADREQUESTS', 
           'VIEWABLEIMPRESSIONS', 'RPM', 'CPM', 'ADREQUESTS_PER_PAGEVIEW', 'VIEWABILITY']

# Display summarized table
datewise_percent_deltas = {metric: [] for metric in metrics}
avg_percent_deltas = {}

for metric in metrics:
    try:
        decomposition = seasonal_decompose(grouped.set_index('DATE')[metric], model='additive', period=7)
        trend = decomposition.trend.dropna()
        percent_deltas = ((trend - trend.shift(1)) / trend.shift(1) * 100).dropna()
        avg_percent_deltas[metric] = percent_deltas.loc[implementation_date:].mean()
        datewise_percent_deltas[metric] = percent_deltas
    except:
        st.write(f"Not enough data on selected filters.")

summary_table = pd.DataFrame({
    'METRIC': metrics,
    'DATASOURCE': [datasource_filter] * len(metrics),
    'DOMAIN': [domain_filter] * len(metrics),
    # 'AVG % DELTA': [avg_percent_deltas[metric] for metric in metrics]
    'AVG % DELTA': [avg_percent_deltas.get(metric, "N/A") for metric in metrics]

})

start_date_for_avg_delta = pd.to_datetime("2023-07-22")

# for date in percent_deltas.index:
for date in datewise_percent_deltas['REVENUE'].index:
    if date < start_date_for_avg_delta:
        continue
    summary_table[date.strftime("%Y-%m-%d")] = [datewise_percent_deltas[metric].get(date, np.nan) for metric in metrics]


st.write(summary_table)

fig, axs = plt.subplots(len(metrics), 4, figsize=(30, 40))

for i, metric in enumerate(metrics):
    try:
        decomposition = seasonal_decompose(grouped.set_index('DATE')[metric], model='additive', period=7)
        trend = decomposition.trend.dropna()
        percent_deltas = ((trend - trend.shift(1)) / trend.shift(1) * 100).dropna()
        avg_delta = percent_deltas.loc[implementation_date:].mean()
        # start_date_for_avg_delta = pd.to_datetime("2023-07-01")
        # avg_delta = percent_deltas.loc[start_date_for_avg_delta:].mean()

        axs[i, 0].plot(grouped['DATE'], grouped[metric], label='Observed')
        axs[i, 1].plot(trend.index, trend.values, label='Trend')
        axs[i, 1].annotate(f"Avg %Δ: {avg_delta:.2f}%", (0.75, 0.75), xycoords='axes fraction', fontsize=20, color='red', bbox=dict(boxstyle='round, pad=0.5', edgecolor='black', facecolor='yellow'))
        axs[i, 1].axvline(implementation_date, color='red', linestyle='--', label='Test Start Date')
        axs[i, 2].plot(trend.index, decomposition.seasonal[trend.index], label='Seasonal')
        axs[i, 3].plot(trend.index, decomposition.resid[trend.index], label='Residual')

        for j in range(4):
            # axs[i, j].legend(loc='upper left', fontsize=20)
            axs[i, j].set_title(f"{metric} - {axs[i, j].get_legend_handles_labels()[1][0]}", fontsize=20)
            axs[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    except:
        for j in range(4):
            axs[i, j].text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center', fontsize=20)
plt.tight_layout()
st.pyplot(fig)

# # Display summarized table
# datewise_percent_deltas = {metric: [] for metric in metrics}
# avg_percent_deltas = {}

# for metric in metrics:
#     decomposition = seasonal_decompose(grouped.set_index('DATE')[metric], model='additive', period=7)
#     trend = decomposition.trend.dropna()
#     percent_deltas = ((trend - trend.shift(1)) / trend.shift(1) * 100).dropna()
#     avg_percent_deltas[metric] = percent_deltas.loc[implementation_date:].mean()
#     datewise_percent_deltas[metric] = percent_deltas

# summary_table = pd.DataFrame({
#     'METRIC': metrics,
#     'DATASOURCE': [datasource_filter] * len(metrics),
#     'DOMAIN': [domain_filter] * len(metrics),
#     'AVG % DELTA': [avg_percent_deltas[metric] for metric in metrics]
# })

# for date in datewise_percent_deltas['REVENUE'].index:
#     summary_table[date.strftime("%Y-%m-%d")] = [datewise_percent_deltas[metric].get(date, np.nan) for metric in metrics]

# st.write(summary_table)

# if __name__ == '__main__':
    # st.title("Streamlit App for Data Analysis")
