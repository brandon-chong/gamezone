#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from ydata_profiling import ProfileReport
%matplotlib inline
# %%
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR / 'gamezone-orders-data.xlsx'
df = pd.read_excel(BASE_DIR / 'gamezone-orders-data.xlsx', sheet_name='orders_cleaned')
# %%
df.drop(columns=['PURCHASE_TS', 'PRODUCT_NAME', 'ACCOUNT_CREATION_METHOD', 'MARKETING_CHANNEL'], inplace=True)

#%%
print(df.columns)
profile_report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile_report.to_file("output.html")

# %% [markdown]

# Is there a relation between the category of product and the time of year that those sales takes place?

#%%
products = df['PRODUCT_NAME_CLEANED'].unique()
n_products = len(products)
ncols = 2
nrows = int(np.ceil(n_products / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharex=True)
axes = axes.flatten()  # Flatten in case of single row

for i, product in enumerate(products):
    df_product = df[df['PRODUCT_NAME_CLEANED'] == product]['PURCHASE_DATE']
    axes[i].hist(df_product, bins=52, alpha=0.5, label=product)
    axes[i].set_title(f"Sales of {product} by week")
    axes[i].set_xlabel("Week")
    axes[i].set_ylabel("Sales")
    axes[i].legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# %% [markdown]

# There appears to be a seasonal trend for the entirety of sales, but they do not seem to particularly apply to any one category. There is however much more variance in the sales of laptops than the other two categories.

# %%

print(df_consoles.describe())

print(df_laptops.describe())

print(df_peripherals.describe())
