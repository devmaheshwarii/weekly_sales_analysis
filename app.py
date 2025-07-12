import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    columns = [
        "Entity Name", "Branch Region", "Branch", "Division", "Due Date", 
        "Top Level Customer ID", "Top Level Customer Name", "Customer ID", 
        "Customer", "Billing Group ID", "Billing Group", "Invoice ID", 
        "Invoice #", "Issue Date", "Total", "Outstanding", "Delivery", "Status"
    ]

    df_nsw = pd.read_csv('NSW.csv', names=columns, header=None)
    df_qld = pd.read_csv('QLD.csv', names=columns, header=None)
    df_wa = pd.read_csv('WA.csv', names=columns, header=None)

    df_nsw['Branch'] = 'NSW'
    df_qld['Branch'] = 'QLD'
    df_wa['Branch'] = 'WA'

    df = pd.concat([df_nsw, df_qld, df_wa], ignore_index=True)

    df.columns = df.columns.str.strip()
    df['Customer'] = df['Customer'].astype(str).str.strip()
    df['Branch'] = df['Branch'].astype(str).str.strip()
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], dayfirst=True, errors='coerce')
    df['Year'] = df['Issue Date'].dt.year
    df['Month'] = df['Issue Date'].dt.to_period('M').astype(str)
    df['Total'] = pd.to_numeric(df['Total'].astype(str).str.replace(',', ''), errors='coerce')

    return df.dropna(subset=['Issue Date', 'Total', 'Branch'])

df = load_data()

st.title("📊 Invoice & Customer Analysis Dashboard")

# ---- Filters ---- #
branch_options = df['Branch'].dropna().unique().tolist()
branch = st.sidebar.multiselect("Select Branch(es)", options=branch_options, default=branch_options)

customer_options = df['Customer'].dropna().unique().tolist()
customer = st.sidebar.multiselect("Select Customer(s)", options=customer_options)

year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider("Select Year Range", year_min, year_max, (year_min, year_max))

date_range = st.sidebar.date_input("Filter by Issue Date Range", [df['Issue Date'].min(), df['Issue Date'].max()])
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# ---- Apply filters ---- #
filtered_df = df[
    df['Branch'].isin(branch) &
    df['Year'].between(*year_range) &
    df['Issue Date'].between(start_date, end_date)
]
if customer:
    filtered_df = filtered_df[filtered_df['Customer'].isin(customer)]

# ---- Annual Sales ---- #
import plotly.express as px
import pandas as pd
import streamlit as st

st.header("📈 Annual Sales Report")

# Grouping data by Year and Branch
annual_sales = filtered_df.groupby(['Year', 'Branch'])['Total'].sum().reset_index()

# Create pivot table for display
pivot_table = annual_sales.pivot(index='Year', columns='Branch', values='Total')

# Add a row at the bottom for total sales across all years
total_row = pivot_table.sum().rename("Total")
pivot_table_with_total = pd.concat([pivot_table, pd.DataFrame([total_row])])

# Show DataFrame
st.dataframe(pivot_table_with_total, use_container_width=True)

# Plotly Bar Chart
fig = px.bar(
    annual_sales,
    x='Year',
    y='Total',
    color='Branch',
    barmode='group',
    title='Annual Branch Sales Comparison',
    text_auto=True,
    hover_data={'Total': ':.2f', 'Branch': True, 'Year': True}
)

fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Total Sales',
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)


# ---- Monthly Sales ---- #
# ---- Monthly Sales ---- #
st.header("📅 Monthly Branch Sales")
monthly_sales = filtered_df.groupby(['Month', 'Branch'])['Total'].sum().reset_index()

fig_month = px.line(
    monthly_sales, x="Month", y="Total", color="Branch", markers=True,
    title="Monthly Sales by Branch", hover_data={"Total": True}
)
fig_month.update_traces(mode='lines+markers', hovertemplate='%{x}<br>Sales: %{y:.2f}')
fig_month.update_layout(xaxis_title="Month", yaxis_title="Sales")
st.plotly_chart(fig_month, use_container_width=True)


# ---- Dropping & Rising Customers ---- #
st.header(" Customer Trends (Drop vs Rise)")

customer_sales = df[df['Branch'].isin(branch)].groupby(['Customer', 'Year'])['Total'].sum().reset_index()
sales_pivot = customer_sales.pivot(index='Customer', columns='Year', values='Total').fillna(0)

years = sorted(sales_pivot.columns)
if len(years) >= 2:
    sales_pivot['Drop?'] = sales_pivot[years[-1]] < sales_pivot[years[-2]]
    sales_pivot['Rise?'] = sales_pivot[years[-1]] > sales_pivot[years[-2]]

    dropping_customers = sales_pivot[sales_pivot['Drop?']].reset_index()
    rising_customers = sales_pivot[sales_pivot['Rise?']].reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"⬇ Dropping Customers ")
        st.dataframe(dropping_customers[['Customer', years[-2], years[-1]]])

    with col2:
        st.subheader(f"⬆ Rising Customers ")
        st.dataframe(rising_customers[['Customer', years[-2], years[-1]]])
else:
    st.info("Not enough years for drop/rise analysis.")

# ---- Customer Purchase View ---- #
st.header("🧾 Customer-wise Purchase Detail")

# Multiselect Customer
cust_df = filtered_df.groupby(['Customer', 'Year'])['Total'].sum().reset_index()
if not cust_df.empty:
    selected_customers = st.multiselect(
        "Select Customer(s) to Analyze",
        options=cust_df['Customer'].unique(),
        default=cust_df['Customer'].unique()[:1]
    )

    # Date Range Filter
    cust_date_range = st.date_input(
        "Select Date Range for Purchase Analysis",
        [filtered_df['Issue Date'].min(), filtered_df['Issue Date'].max()]
    )
    cust_start_date = pd.to_datetime(cust_date_range[0])
    cust_end_date = pd.to_datetime(cust_date_range[1])

    # Filter based on selection
    cust_purchase = filtered_df[
        (filtered_df['Customer'].isin(selected_customers)) &
        (filtered_df['Issue Date'].between(cust_start_date, cust_end_date))
    ]

    # Show drop warnings
    if 'dropping_customers' in locals():
        for cust in selected_customers:
            if cust in dropping_customers['Customer'].values:
                st.warning(f"⚠️ {cust} is a **dropping customer** (sales declined from {years[-2]} to {years[-1]}).")

    # Show raw purchase records
    st.subheader("Filtered Purchase Records")
    st.dataframe(
        cust_purchase[['Customer', 'Issue Date', 'Branch', 'Invoice ID', 'Total']],
        use_container_width=True
    )

    # Year-wise Total Purchases (Bar Chart)
    st.subheader("📊 Year-wise Purchase Totals")
    cust_yearly = cust_purchase.groupby(['Customer', 'Year'])['Total'].sum().reset_index()

    if not cust_yearly.empty:
        fig_year = px.bar(
            cust_yearly, x="Year", y="Total", color="Customer", barmode='group',
            title="Yearly Purchase Summary"
        )
        fig_year.update_traces(hovertemplate='Year: %{x}<br>Total: %{y:.2f}')
        fig_year.update_layout(xaxis_title="Year", yaxis_title="Total Purchase")
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.info("No yearly data available for selected customers/date range.")

    # Monthly Trend (Line Chart)
    st.subheader("📈 Monthly Purchase Trend")
    cust_purchase['Month'] = cust_purchase['Issue Date'].dt.to_period('M').astype(str)
    cust_monthly = cust_purchase.groupby(['Customer', 'Month'])['Total'].sum().reset_index()

    if not cust_monthly.empty:
        fig_monthly = px.line(
            cust_monthly, x="Month", y="Total", color="Customer", markers=True,
            title="Monthly Purchase Trend"
        )
        fig_monthly.update_traces(hovertemplate='Month: %{x}<br>Total: %{y:.2f}')
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Total Purchase")
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("No monthly data available for selected customers/date range.")

else:
    st.info("No customers found for the selected filters.")
