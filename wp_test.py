import pandas as pd



params = {'job_code': ['Aa', 'A']}

df1 = pd.DataFrame({
'dispatch_area': ['A', 'A', 'A'],
'job_code': ['A', 'b', 'c'],
'collection_amount': [10, 20, 30]
})


df2 = pd.DataFrame({
'dispatch_area': ['A', 'C'],
'threshold_value': [22, 44]
})

merged_df = pd.merge(df1, df2, on="dispatch_area", how="left")

merged_df["collection_amount"] = merged_df["collection_amount"].astype(float).fillna(0.0)

print((merged_df["collection_amount"] <= merged_df["threshold_value"]))

print((df1['job_code'].isin(params['job_code'])))

merged_df = merged_df[~((df1['job_code'].isin(params['job_code'])) & (merged_df["collection_amount"] <= merged_df["threshold_value"]))].drop("threshold_value", axis=1)

print(merged_df.shape)