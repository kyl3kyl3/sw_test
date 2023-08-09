
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data
@st.cache
def load_data():
    return pd.read_csv('COMBINED_TEST_DATA.csv')

data = load_data()

# Compute the %deltas based on trend values
def compute_deltas(data, metrics, implementation_date):
    data_agg = data.groupby('DATE')[metrics].sum()
    df_deltas = pd.DataFrame(columns=['METRIC', 'DATE', 'TREND VALUE', '% DELTA'])
    
    for metric in metrics:
        ts = data_agg[metric]
        result = seasonal_decompose(ts, model='additive', period=7)
        
        for date in result.trend.index[result.trend.index > pd.to_datetime(implementation_date)]:
            previous_value = result.trend.loc[date - pd.DateOffset(1)]
            current_value = result.trend.loc[date]

            if pd.notnull(previous_value) and pd.notnull(current_value):  # Skip missing values
                percent_diff = ((current_value - previous_value) / abs(previous_value)) * 100
                new_row = {'METRIC': metric, 'DATE': date, 'TREND VALUE': current_value, '% DELTA': percent_diff}
                df_deltas = df_deltas.append(new_row, ignore_index=True)
                
    return df_deltas

def plot_metric_deltas(df_deltas, metric):
    metric_data = df_deltas[df_deltas['METRIC'] == metric]
    fig = px.line(metric_data, x='DATE', y='% DELTA', title=f"% Delta Trend for {metric}")
    return fig

# Filter data based on user input
def filter_data(data, datasource, device, domain):
    if datasource:
        data = data[data['DATASOURCE'] == datasource]
    if device:
        data = data[data['DEVICE'] == device]
    if domain:
        data = data[data['DOMAIN'] == domain]
    return data

# Streamlit UI
st.title('Data Visualization Interface')

# Dropdowns for DATASOURCE, DEVICE, and DOMAIN
datasource = st.selectbox('Select DATASOURCE', options=['All'] + list(data['DATASOURCE'].unique()))
device = st.selectbox('Select DEVICE', options=['All'] + list(data['DEVICE'].unique()))
domain = st.selectbox('Select DOMAIN', options=['All'] + list(data['DOMAIN'].unique()))

# Filter the data based on selections
if datasource == 'All':
    datasource = None
if device == 'All':
    device = None
if domain == 'All':
    domain = None

filtered_data = filter_data(data, datasource, device, domain)

# Compute deltas for visualization
metrics = ['PAGEVIEWS', 'ADREQUESTS_PER_PAGEVIEW', 'REVENUE', 'RPM', 'CPM', 'IMPRESSIONS', 'VIEWABILITY']
implementation_date = '2023-07-21'
df_deltas = compute_deltas(filtered_data, metrics, implementation_date)

# Plot and display data
selected_metric = st.selectbox('Select METRIC for Visualization', options=metrics)
fig = plot_metric_deltas(df_deltas, selected_metric)
st.plotly_chart(fig)

# Display summarized table
pivot_deltas = df_deltas.pivot(index='METRIC', columns='DATE', values='% DELTA').reset_index()
pivot_deltas['DATASOURCE'] = datasource
pivot_deltas['DOMAIN'] = domain
avg_deltas = df_deltas.groupby('METRIC')['% DELTA'].mean()
pivot_deltas['AVG % DELTA'] = pivot_deltas['METRIC'].map(avg_deltas)
st.write(pivot_deltas)
