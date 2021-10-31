##############################################################
#CLTV Prediction with  BG-NBD and Gamma-Gamma Submodel
##############################################################

# Columns:
# 0-Invoice – Fatura Numarası (Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.)
# 1-StockCode – Ürün kodu (Her bir ürün için eşsiz numara.)
# 2-Description – Ürün ismi
# 3-Quantity – Ürün adedi (Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.)
# 4-InvoiceDate – Fatura tarihi
# 5-Price – Fatura fiyatı (Sterlin)
# 6-Customer ID – Eşsiz müşteri numarası
# 7-Country – Ülke ismi


import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


retail = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = retail.copy()
print(df.shape)
df.head()

# Data Preparation
print(df.describe().T)
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["Country"].value_counts()

df = df[df["Country"] == "United Kingdom"]

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda tp: tp.sum()})

cltv_df.head()

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

cltv_df["frequency"] = cltv_df["frequency"].astype(int)

# Expected Sales Forecasting with BG-NBD

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

cltv_df["expected_purc_6_months"] = bgf.predict(24,
                                                cltv_df['frequency'],
                                                cltv_df['recency'],
                                                cltv_df['T'])

cltv_df.sort_values(by="expected_purc_6_months", ascending=False).head(10)

# Expected Average Profit with Gamma-Gamma submodel

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values(by="expected_average_profit", ascending=False).head(10)

# Calculating CLTV with BG-NBD and Gamma-Gamma submodel

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(5)

# Creating Segments by CLTV

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()

cltv_final.sort_values(by="clv", ascending=False).head()

cltv_final.sort_values(by="clv", ascending=False).head()
