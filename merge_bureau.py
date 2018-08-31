import pandas as pd
import matplotlib.pyplot as plt


def merge_bureau():
    data_dir = "C:/Users/Kyohei Harada/Desktop/HomeCreditDefaultRisk/all"
    bureau = pd.read_csv(data_dir + "/bureau.csv")
    bureau_length = bureau.shape[0]

    bureau_balance = pd.read_csv(data_dir + "/bureau_balance.csv")
    bureau_balance_length = bureau_balance.shape[0]
    
    null_sum = 0
    for col in bureau.columns:
        #欠損の補間
        drop_flag = False
        null_sum = bureau[col].isnull().sum()
        if null_sum > 0:
            if null_sum/bureau_length >= 0.6:
                drop_flag = True
            else:
                if bureau[col].dtype == object:
                    bureau[col] = bureau[col].fillna(bureau[col].mode()[0])
                else:
                    bureau[col] = bureau[col].fillna(bureau[col].mean())
        elif bureau[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
            bureau[col] = bureau[col].replace("XNA",bureau[col].mode())
            bureau[col] = bureau[col].replace("Unknown",bureau[col].mode())
            bureau[col] = bureau[col].replace("Maternity leave",bureau[col].mode())

        #one-hotに
        if bureau[col].dtype == "object":
            bureau = pd.concat([bureau, pd.get_dummies(bureau[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            bureau = bureau.drop(col, axis=1)

    null_sum = 0
    for col in bureau_balance.columns:
        #欠損の補間
        drop_flag = False
        null_sum = bureau_balance[col].isnull().sum()
        if null_sum > 0:
            if null_sum/bureau_balance_length >= 0.6:
                drop_flag = True
            else:
                if bureau_balance[col].dtype == object:
                    bureau_balance[col] = bureau_balance[col].fillna(bureau_balance[col].mode()[0])
                else:
                    bureau_balance[col] = bureau_balance[col].fillna(bureau_balance[col].mean())
        
        #one-hotに
        if bureau_balance[col].dtype == "object":
            bureau_balance = pd.concat([bureau_balance, pd.get_dummies(bureau_balance[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            bureau_balance = bureau_balance.drop(col, axis=1)

    month = bureau_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].idxmax()
    bureau_balance = bureau_balance.loc[month, :]

    merged_bureau = bureau.merge(bureau_balance, on="SK_ID_BUREAU", how="left", suffixes=["BB",""])
    merged_bureau = merged_bureau.drop("SK_ID_BUREAU", axis=1)

    merged_length = merged_bureau.shape[0]

    for col in merged_bureau.columns:
        #欠損の補間
        drop_flag = False
        null_sum = merged_bureau[col].isnull().sum()
        if null_sum > 0:
            if null_sum/merged_length >= 0.6:
                drop_flag = True
            else:
                merged_bureau[col] = merged_bureau[col].fillna(0)

    merged_bureau = merged_bureau.groupby("SK_ID_CURR").mean()

    merged_bureau.to_csv("merged_bureau.csv")

    return merged_bureau

if __name__ == "__main__":
    merge_bureau()
