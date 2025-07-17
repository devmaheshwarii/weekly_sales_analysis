import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(layout="wide")

@st.cache_data
# ///

# ///


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

def load_historical_sales_data():
    """Loads and preprocesses the historical weekly sales data from Excel sheets,
    handling the two-row header structure and selecting relevant columns."""
    excel_file_path = 'HISTORICAL_REPORT.xlsx' # This is now the single Excel file
    sheet_names = ['WA', 'QLD', 'NSW'] # These are the sheet names within the Excel file

    all_historical_df = []

    try:
        for sheet_name in sheet_names:
            # Read the specific sheet from the Excel file with no header initially
            df_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)

            # Extract the relevant header information from the first two rows
            header_row_0 = df_raw.iloc[0] # Contains 'Financial Year', '18/19', 'Variance YOY', etc.

            # Identify the indices of the actual sales year columns in header_row_0
            # These are columns like '18/19', '19/20', etc., which are strings containing '/'
            sales_year_indices = [i for i, val in enumerate(header_row_0) if isinstance(val, str) and '/' in val]

            # Construct the list of new column names for the DataFrame
            # The first column will be 'Week' (from 'Week No' in row 1, which is df_raw.iloc[1,0])
            new_column_names = ['Week']
            for idx in sales_year_indices:
                new_column_names.append(str(header_row_0[idx]))

            # Select the actual data rows (starting from index 2, i.e., the 3rd row)
            # and only the columns that correspond to our new_column_names
            data_columns_to_select = [0] + sales_year_indices

            df_processed = df_raw.iloc[2:, data_columns_to_select].copy()
            df_processed.columns = new_column_names # Assign the new, clean column names

            # Filter out summary rows (Q1 Total, Totals, etc.)
            df_processed = df_processed[df_processed['Week'].astype(str).str.contains(r'Week\s\d+', na=False)]

            # Melt the DataFrame to unpivot the year columns
            id_vars_for_melt = ['Week']
            value_vars_for_melt = [col for col in new_column_names if col != 'Week']

            df_melted = df_processed.melt(id_vars=id_vars_for_melt, value_vars=value_vars_for_melt,
                                           var_name='Financial Year', value_name='Total')

            # Add Branch column
            df_melted['Branch'] = sheet_name

            # Convert 'Week' to numeric
            df_melted['Week'] = df_melted['Week'].astype(str).str.replace('Week ', '').astype(int)

            # Convert 'Total' to numeric, handling commas and errors, then to float
            df_melted['Total'] = pd.to_numeric(df_melted['Total'].astype(str).str.replace(',', ''), errors='coerce').astype(float)

            all_historical_df.append(df_melted)

    except FileNotFoundError:
        st.error(f"Error: Excel file '{excel_file_path}' not found. Please ensure it's in the same directory as the script.")
    except Exception as e:
        st.error(f"An error occurred while processing Excel file '{excel_file_path}' for sheet '{sheet_name}': {e}")

    if all_historical_df:
        return pd.concat(all_historical_df, ignore_index=True).dropna(subset=['Total'])
    else:
        return pd.DataFrame(columns=['Week', 'Financial Year', 'Total', 'Branch'])

# Load dataframes
df = load_data()
historical_df = load_historical_sales_data()

st.title("📊 Invoice & Customer Analysis Dashboard")

# ---- Filters ---- #
branch_options = df['Branch'].dropna().unique().tolist()
branch = st.sidebar.multiselect("Select Branch(es)", options=branch_options, default=branch_options)

# Historical data filters
if not historical_df.empty:
    financial_year_options = sorted(historical_df['Financial Year'].dropna().unique().tolist())

    # Add a "Select All" checkbox
    select_all_years = st.sidebar.checkbox("Select All Financial Years", value=True)

    if select_all_years:
        selected_financial_years = financial_year_options
    else:
        # Default to the first year if "Select All" is not checked and no years are selected
        # Or you can choose to default to an empty list or specific years as per preference
        default_selection = [financial_year_options[0]] if financial_year_options else []
        selected_financial_years = st.sidebar.multiselect(
            "Select Financial Year(s) (Historical Data)",
            options=financial_year_options,
            default=default_selection # Default to only the first year when "Select All" is off
        )
        # Handle case where user deselects all after unchecking "Select All"
        if not selected_financial_years and not select_all_years:
            st.sidebar.warning("Please select at least one financial year or 'Select All'.")
            selected_financial_years = [] # Ensure it's an empty list to filter nothing

    # Apply branch filter to historical data
    filtered_historical_df = historical_df[
        historical_df['Branch'].isin(branch) &
        historical_df['Financial Year'].isin(selected_financial_years)
    ].copy() # Ensure filtered_historical_df is a copy
else:
    st.info("Historical sales data not loaded. Week-wise analysis will be empty.")
    filtered_historical_df = pd.DataFrame()


customer_options = df['Customer'].dropna().unique().tolist()
customer_options = sorted(customer_options)
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
# import plotly.express as px
# import pandas as pd
# import streamlit as st

# st.header("📈 Annual Sales Report")

# # Grouping data by Year and Branch
# annual_sales = filtered_df.groupby(['Year', 'Branch'])['Total'].sum().reset_index()

# # Create pivot table for display
# pivot_table = annual_sales.pivot(index='Year', columns='Branch', values='Total')

# # Add a row at the bottom for total sales across all years
# total_row = pivot_table.sum().rename("Total")
# pivot_table_with_total = pd.concat([pivot_table, pd.DataFrame([total_row])])

# # Show DataFrame
# st.dataframe(pivot_table_with_total, use_container_width=True)

# # Plotly Bar Chart
# fig = px.bar(
#     annual_sales,
#     x='Year',
#     y='Total',
#     color='Branch',
#     barmode='group',
#     title='Annual Branch Sales Comparison',
#     text_auto=True,
#     hover_data={'Total': ':.2f', 'Branch': True, 'Year': True}
# )

# fig.update_layout(
#     xaxis_title='Year',
#     yaxis_title='Total Sales',
#     hovermode='x unified'
# )

# st.plotly_chart(fig, use_container_width=True)

# --- Week-wise Sales Analysis (Uses Historical Data) ---
st.header("📅 Annual Sales Analysis")

if not filtered_historical_df.empty:
    # --- 2. Enhanced Quarter/Week Range Analysis ---
    st.subheader("Quarter/Week Range Analysis")

    # Option to select Quarters
    quarter_options = ["All Quarters", "Q1 (Weeks 1-13)", "Q2 (Weeks 14-26)", "Q3 (Weeks 27-39)", "Q4 (Weeks 40-52)"]
    selected_quarters_display = st.multiselect("Select Quarter(s)", quarter_options, default=["All Quarters"])

    quarter_mapping = {
        "Q1 (Weeks 1-13)": (1, 13),
        "Q2 (Weeks 14-26)": (14, 26),
        "Q3 (Weeks 27-39)": (27, 39),
        "Q4 (Weeks 40-52)": (40, 52)
    }

    # Option to select specific week ranges
    all_weeks = sorted(filtered_historical_df['Week'].unique().tolist())
    selected_weeks = st.multiselect("Or, Select Specific Week(s)", all_weeks)

    # Filter data based on quarter or week range selection
    quarter_week_filtered_df = filtered_historical_df.copy()

    # Apply quarter filter if selected
    if "All Quarters" not in selected_quarters_display:
        quarter_weeks = []
        for q_display in selected_quarters_display:
            start_week, end_week = quarter_mapping[q_display]
            quarter_weeks.extend(range(start_week, end_week + 1))
        quarter_week_filtered_df = quarter_week_filtered_df[quarter_week_filtered_df['Week'].isin(quarter_weeks)]

    # Apply specific week filter if selected (overrides quarter if both selected)
    if selected_weeks:
        quarter_week_filtered_df = quarter_week_filtered_df[quarter_week_filtered_df['Week'].isin(selected_weeks)]

    if not quarter_week_filtered_df.empty:
        st.write(f"**Detailed Sales for Selected Range**")
        st.dataframe(quarter_week_filtered_df[['Branch', 'Financial Year', 'Week', 'Total']].sort_values(['Branch', 'Financial Year', 'Week']), use_container_width=True)

        total_sales_for_range = quarter_week_filtered_df['Total'].sum()
        st.metric(label=f"Total Sales for Selected Range", value=f"${total_sales_for_range:,.2f}")

        # Line chart for selected week range/quarter
        st.subheader("Sales Trend for Selected Range")
        fig_quarter_week_trend = px.line(
            quarter_week_filtered_df,
            x='Week',
            y='Total',
            color='Branch',
            line_dash='Financial Year',
            markers=True,
            title='Sales Trend by Week for Selected Range',
            hover_data={'Total': ':.2f', 'Week': True, 'Financial Year': True, 'Branch': True}
        )
        fig_quarter_week_trend.update_layout(
            xaxis_title='Week Number',
            yaxis_title='Total Sales',
            hovermode='x unified'
        )
        fig_quarter_week_trend.update_xaxes(dtick=1)
        st.plotly_chart(fig_quarter_week_trend, use_container_width=True)

    else:
        st.info("No sales data available for the selected quarter(s) or week range based on current filters.")

    st.markdown("---") # Separator

    # 2. User can compare the total sale of the whole financial year using bar chart.
    st.subheader("Financial Year Total Sales Comparison")
    annual_historical_sales = filtered_historical_df.groupby(['Financial Year', 'Branch'])['Total'].sum().reset_index()
    if not annual_historical_sales.empty:
        fig_annual_hist = px.bar(
            annual_historical_sales,
            x='Financial Year',
            y='Total',
            color='Branch',
            barmode='group',
            title='Total Sales per Financial Year by Branch',
            text_auto=True,
            hover_data={'Total': ':.2f', 'Branch': True, 'Financial Year': True}
        )
        fig_annual_hist.update_layout(
            xaxis_title='Financial Year',
            yaxis_title='Total Sales',
            hovermode='x unified'
        )
        st.plotly_chart(fig_annual_hist, use_container_width=True)
    else:
        st.info("No annual historical sales data available for comparison.")

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
                st.warning(f" {cust} is a **dropping customer** (sales declined from {years[-2]} to {years[-1]}).")

    # Show raw purchase records
# Show raw purchase records
    st.subheader("Filtered Purchase Records")
    st.dataframe(
        cust_purchase[['Customer', 'Issue Date', 'Branch', 'Invoice ID', 'Total']],
        use_container_width=True
    )

    # Calculate and display the total sum of purchases
    total_filtered_purchase = cust_purchase['Total'].sum()
    st.metric(label="Total Purchase for Filtered Records", value=f"${total_filtered_purchase:,.2f}")

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
