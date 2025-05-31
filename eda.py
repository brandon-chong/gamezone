#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from ydata_profiling import ProfileReport
import plotly.graph_objects as go
import pySankey as plt_sk
%matplotlib inline

# %%
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR / 'gamezone-orders-data.xlsx'
df = pd.read_excel(BASE_DIR / 'gamezone-orders-data.xlsx', sheet_name='orders_cleaned')
df['PURCHASE_DATE_CLEANED'] = pd.to_datetime(df['PURCHASE_DATE_CLEANED'])
df.drop_duplicates(inplace=True)
# %%
df.drop(columns=['PURCHASE_TS', 'PRODUCT_NAME', 'ACCOUNT_CREATION_METHOD', 'MARKETING_CHANNEL'], inplace=True)

#%%
print(df.columns)
profile_report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile_report.to_file("output.html")

# %% [markdown]

# Is there a relation between the type of product and the time of year that those sales takes place?

#%%
products = df['PRODUCT_NAME_CLEANED'].unique()
n_products = len(products)
ncols = 2
nrows = int(np.ceil(n_products / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharex=True)
axes = axes.flatten()

for i, product in enumerate(products):
    df_product = df[df['PRODUCT_NAME_CLEANED'] == product][['PURCHASE_DAY_OF_YEAR', 'PURCHASE_DATE_CLEANED']]

    # Trimming off 2021 data as it is incomplete
    tdf = df_product[df_product['PURCHASE_DATE_CLEANED'].dt.year < 2021]['PURCHASE_DAY_OF_YEAR']

    axes[i].hist(tdf, bins=52, alpha=0.5, label=product)
    axes[i].set_title(f"Sales of {product} by week")
    axes[i].set_xlabel("Week")
    axes[i].set_ylabel("Sales")
    axes[i].legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# %% [markdown]

# There appears to be a seasonal trend for the entirety of sales, but the JBL Quantum Headset appears to be the least seasonal, surprisingly the dell gaming mouse is the most seasonal, though this could be due to a small sample size.


# %% [markdown]
# Let's try to quantify the seasonal trend by looking at december sales in comparison to the rest of the year, leaving off the Razer Pro Gaming Headset and Acer Nitro 5 as they sell too few units to analyze

#%%
products = df['PRODUCT_NAME_CLEANED'].unique()
products = products[~np.isin(products, ['Razer Pro Gaming Headset', 'Acer Nitro 5'])]

seasonal_sales_ratio = {}

for product in products:
    december_value = df[(df['PURCHASE_DATE_CLEANED'].dt.month == 12) & (df['PRODUCT_NAME_CLEANED'] == product)].nunique()['ORDER_ID']
    year_value = df[(df['PURCHASE_DATE_CLEANED'].dt.month != 12) & (df['PRODUCT_NAME_CLEANED'] == product)].nunique()['ORDER_ID']

    seasonal_sales_ratio[product] = december_value / year_value * 12

seasonal_sales_ratio = pd.DataFrame(seasonal_sales_ratio.items())
seasonal_sales_ratio.columns = ['PRODUCT_NAME_CLEANED', 'SEASONAL_SALES_RATIO']
seasonal_sales_ratio

# %% [markdown]
# The JBL Quantum Headset, Nintendo Switch and 4K montior have the smallest increase in sales, surprisingly, the Dell Gaming mouse and Acer Nitro have the largest increase in sales, not the standard gaming consoles. 

# %%
df['PURCHASE_DATE_CLEANED'].describe()
df[df['PRODUCT_NAME_CLEANED'] == 'Dell Gaming Mouse']['PURCHASE_DATE_CLEANED'].describe()

# %% [markdown]
# It appears that the Dell Gaming Mouse was released in the beginning of 2020, if the release of a new product of its type has a particular trend in sales, then this might explain the hard skew towards the end of the year, though I would expect this to be the opposite, I'll assume that the trend isn't caused by the release, though this would be worth talking to someone with more experience in the field about.


# %%
products = df['PRODUCT_NAME_CLEANED'].unique()
n_products = len(products)
ncols = 2
nrows = int(np.ceil(n_products / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharex=True)
axes = axes.flatten()

for i, product in enumerate(products):
    df_product = df[df['PRODUCT_NAME_CLEANED'] == product].copy()
    df_product['MONTH'] = df_product['PURCHASE_DATE_CLEANED'].dt.month
    month_counts = df_product['MONTH'].value_counts().sort_index()
    axes[i].bar(month_counts.index, month_counts.values, alpha=0.7, label=product)
    axes[i].set_title(f"Sales of {product} by month")
    axes[i].set_xlabel("Month")
    axes[i].set_ylabel("Sales")
    axes[i].set_xticks(range(1, 13))
    axes[i].legend()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%% [markdown]
# Let's look at repeat customers

# %%

repeat_customers = df.groupby('USER_ID').nunique()['ORDER_ID'].reset_index()
repeat_customers = repeat_customers[repeat_customers['ORDER_ID']>=2]
repeat_customers


# %% [markdown]
# Let's begin by looking to see if purchasing any particular product makes one more likely to puchase another one

#%%

# product_ids_to_names = df[['PRODUCT_ID', 'PRODUCT_NAME_CLEANED']].drop_duplicates().set_index('PRODUCT_ID')['PRODUCT_NAME_CLEANED'].to_dict()

repeat_purchases = pd.merge(df, df, on=['USER_ID'], suffixes=('_1', '_2'))

repeat_purchases = repeat_purchases[repeat_purchases['PURCHASE_DATE_CLEANED_1'] < repeat_purchases['PURCHASE_DATE_CLEANED_2']]

follow_up_purchases = repeat_purchases[['PRODUCT_NAME_CLEANED_1', 'PRODUCT_NAME_CLEANED_2']]

follow_up_purchase_ids = list(follow_up_purchases['PRODUCT_NAME_CLEANED_1'].unique())

# indecies = 
flows = []

for leader in follow_up_purchase_ids:
    for follower in follow_up_purchase_ids:
        count = follow_up_purchases[(follow_up_purchases['PRODUCT_NAME_CLEANED_1'] == leader) & (follow_up_purchases['PRODUCT_NAME_CLEANED_2'] == follower)].shape[0]
        flows.append((follow_up_purchase_ids.index(leader), follow_up_purchase_ids.index(follower), count))

flows_array = np.array(flows)

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label= follow_up_purchase_ids + follow_up_purchase_ids,
    ),
    link=dict(
        source=flows_array[:, 0], # leaders
        target=flows_array[:, 1] + len(follow_up_purchase_ids), # followers
        value=flows_array[:, 2] , # counts
    )
))

fig.update_layout(title_text="Purchase patterns", font_size=12)
fig.show()


# follow_up_purchases[]

# multiple_purchases = pd.merge(repeat_purchases, repeat_purchases, on=['USER_ID', ''], suffixes=('_1', '_2'))

# repeat_purchases[]

# repeat_purchases = repeat_purchases[['USER_ID', 'PURCHASE_DATE_CLEANED', 'ORDER_ID', 'PRODUCT_ID', 'PRODUCT_NAME_CLEANED']]
# repeat_purchases.drop_duplicates(inplace=True)

# repeat_purchases.groupby('USER_ID').nunique()['ORDER_ID'].reset_index()
# repeat_purchases.pivot(index=['USER_ID', 'ORDER_ID'], columns='PURCHASE_DATE_CLEANED', values='PRODUCT_ID').reset_index()
# repeat_purchases.sort_values(by=['USER_ID', 'PURCHASE_DATE_CLEANED']).drop_duplicates()



# %%
