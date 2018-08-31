import pandas as pd
import matplotlib.pyplot as plt


def merge_prev_app():
    data_dir = "C:/Users/Kyohei Harada/Desktop/HomeCreditDefaultRisk/all"
    prev_app = pd.read_csv(data_dir + "/previous_application.csv")
    prev_app_length = prev_app.shape[0]

    pos_cash = pd.read_csv(data_dir + "/POS_CASH_balance.csv")
    pos_cash_length = pos_cash.shape[0]
    
    inst_pay = pd.read_csv(data_dir + "/installments_payments.csv")
    inst_pay_length = inst_pay.shape[0]

    credit_card = pd.read_csv(data_dir + "/credit_card_balance.csv")
    credit_card_length = credit_card.shape[0]

    null_sum = 0
    for col in prev_app.columns:
        #欠損の補間
        drop_flag = False
        null_sum = prev_app[col].isnull().sum()
        if null_sum > 0:
            if null_sum/prev_app_length >= 0.6:
                drop_flag = True
            else:
                if prev_app[col].dtype == object:
                    prev_app[col] = prev_app[col].fillna(prev_app[col].mode()[0])
                else:
                    prev_app[col] = prev_app[col].fillna(prev_app[col].mean())
        elif prev_app[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
            prev_app[col] = prev_app[col].replace("XNA",prev_app[col].mode())
            prev_app[col] = prev_app[col].replace("Unknown",prev_app[col].mode())
            prev_app[col] = prev_app[col].replace("Maternity leave",prev_app[col].mode())

        #one-hotに
        if prev_app[col].dtype == "object":
            prev_app = pd.concat([prev_app, pd.get_dummies(prev_app[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            prev_app = prev_app.drop(col, axis=1)

    null_sum = 0
    for col in pos_cash.columns:
        #欠損の補間
        drop_flag = False
        null_sum = pos_cash[col].isnull().sum()
        if null_sum > 0:
            if null_sum/pos_cash_length >= 0.6:
                drop_flag = True
            else:
                if pos_cash[col].dtype == object:
                    pos_cash[col] = pos_cash[col].fillna(pos_cash[col].mode()[0])
                else:
                    pos_cash[col] = pos_cash[col].fillna(pos_cash[col].mean())
        elif pos_cash[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
            pos_cash[col] = pos_cash[col].replace("XNA",pos_cash[col].mode())
            pos_cash[col] = pos_cash[col].replace("Unknown",pos_cash[col].mode())
            pos_cash[col] = pos_cash[col].replace("Maternity leave",pos_cash[col].mode())
        
        #one-hotに
        if pos_cash[col].dtype == "object":
            pos_cash = pd.concat([pos_cash, pd.get_dummies(pos_cash[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            pos_cash = pos_cash.drop(col, axis=1)

    null_sum = 0
    for col in inst_pay.columns:
        #欠損の補間
        drop_flag = False
        null_sum = inst_pay[col].isnull().sum()
        if null_sum > 0:
            if null_sum/inst_pay_length >= 0.6:
                drop_flag = True
            else:
                if inst_pay[col].dtype == object:
                    inst_pay[col] = inst_pay[col].fillna(inst_pay[col].mode()[0])
                else:
                    inst_pay[col] = inst_pay[col].fillna(inst_pay[col].mean())
        elif inst_pay[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
            inst_pay[col] = inst_pay[col].replace("XNA",inst_pay[col].mode())
            inst_pay[col] = inst_pay[col].replace("Unknown",inst_pay[col].mode())
            inst_pay[col] = inst_pay[col].replace("Maternity leave",inst_pay[col].mode())

        #one-hotに
        if inst_pay[col].dtype == "object":
            inst_pay = pd.concat([inst_pay, pd.get_dummies(inst_pay[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            inst_pay = inst_pay.drop(col, axis=1)

    null_sum = 0
    for col in credit_card.columns:
        #欠損の補間
        drop_flag = False
        null_sum = credit_card[col].isnull().sum()
        if null_sum > 0:
            if null_sum/credit_card_length >= 0.6:
                drop_flag = True
            else:
                if credit_card[col].dtype == object:
                    credit_card[col] = credit_card[col].fillna(credit_card[col].mode()[0])
                else:
                    credit_card[col] = credit_card[col].fillna(credit_card[col].mean())
        elif credit_card[col].isin(['XNA', 'Unknown', 'Maternity leave']).any():
            credit_card[col] = credit_card[col].replace("XNA",credit_card[col].mode())
            credit_card[col] = credit_card[col].replace("Unknown",credit_card[col].mode())
            credit_card[col] = credit_card[col].replace("Maternity leave",credit_card[col].mode())

        #one-hotに
        if credit_card[col].dtype == "object":
            credit_card = pd.concat([credit_card, pd.get_dummies(credit_card[col], prefix=col)], axis=1)
            drop_flag = True

        #不要列の削除
        if drop_flag:
            credit_card = credit_card.drop(col, axis=1)


    month = pos_cash.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
    pos_cash = pos_cash.loc[month, :]
    prev_app = prev_app.merge(pos_cash, on=["SK_ID_PREV","SK_ID_CURR"], how="left", suffixes=["PCB",""])

    month = inst_pay.groupby("SK_ID_PREV")["NUM_INSTALMENT_NUMBER"].max()
    inst_pay = inst_pay.loc[month, :]
    prev_app = prev_app.merge(inst_pay, on=["SK_ID_PREV","SK_ID_CURR"], how="left", suffixes=["IP",""])

    month = credit_card.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
    credit_card = credit_card.loc[month, :]
    merged_prev_app = prev_app.merge(credit_card, on=["SK_ID_PREV","SK_ID_CURR"], how="left", suffixes=["CCB",""])

    merged_length = merged_prev_app.shape[0]
    null_sum = 0
    for col in merged_prev_app.columns:
        #欠損の補間
        drop_flag = False
        null_sum = merged_prev_app[col].isnull().sum()
        if null_sum > 0:
            if null_sum/merged_length >= 0.6:
                drop_flag = True
            else:
                merged_prev_app[col] = merged_prev_app[col].fillna(0)

    merged_prev_app = merged_prev_app.drop("SK_ID_PREV", axis=1)
    merged_prev_app = merged_prev_app.groupby("SK_ID_CURR").mean()

    merged_prev_app.to_csv("merged_prev_app.csv")

    return merged_prev_app

if __name__ == "__main__":
    merge_prev_app()
