"""
Created on Thu Nov  7 20:59:22 2024

@author: Pavan
"""
import numpy as np
import pandas as pd
import scipy.stats as sct
import statsmodels.api as sm
import pickle


data = pd.read_parquet("data_FIN509535_Fall2024.parquet")
names = pd.read_csv("names.csv")

names['date'] = pd.to_datetime(names['date'])
names['date'] = names['date'] - pd.offsets.Day() + pd.offsets.MonthEnd()
names.rename({'date': 'eom', 'PERMNO': 'permno'}, axis=1, inplace=True)

data2 = data.droplevel(0, axis=1).join(
    names.set_index(['eom', 'permno']),
    how='left',
    lsuffix='',
    rsuffix='_fromnames'
)


with open('famafrench5factors.pkl', 'rb') as fp:
    ff = pickle.load(fp)

ff5 = ff[0].to_timestamp(freq='M') / 100
ff5.index.rename('eom', inplace=True)
ffn = [n for n in ff5.columns if n != 'RF']

rets = data2['ret_exc_lead1m'].reset_index(level='permno').pivot(
    columns='permno').droplevel(0, axis=1).dropna(axis=1)
ff49 = data2['ff49'].reset_index(level='permno').pivot(
    columns='permno').droplevel(0, axis=1).dropna(axis=1)

industries = [
    [48, 'Financial Trading'],
    [36, 'Computer Software'],
    [19, 'Steel Works Etc']
]

print()

for ind_code, ind_name in industries:
    chosen = ff49.columns[ff49.loc['2000-01-31', :] == ind_code]
    retschosen = rets.loc[:, chosen].join(ff5.shift(-1))
    regs = sm.OLS(
        endog=retschosen[chosen],
        exog=sm.add_constant(retschosen[ffn])
    ).fit()
    
    param_unc = pd.DataFrame(
        index=regs.params.index, columns=regs.resid.columns,
        data=regs.params.values
    )
    resid_unc = regs.resid
    Sigma_unc = resid_unc.T @ resid_unc / resid_unc.shape[0]

    print(f"Industry: {ind_name}")

    print("\nQuestion 1: Five-Factor Regression (Highest Beta)")
    for factor in ffn:
        highest_permno = param_unc.loc[factor].idxmax()
        highest_company = data2.reorder_levels([1, 0]).loc[highest_permno, 'COMNAM'].unique()[0]
        print(f"  Factor: {factor} - Company: {highest_company} - Beta: {param_unc.loc[factor, highest_permno]:.2f}")
    
    regs_nocons = sm.OLS(
        endog=retschosen[chosen],
        exog=retschosen[ffn]
    ).fit()
    
    resid_con_nocons = regs_nocons.resid
    Sigma_con = resid_con_nocons.T @ resid_con_nocons / resid_con_nocons.shape[0]
    J = resid_unc.shape[0] * (
        np.linalg.slogdet(Sigma_con)[1] - np.linalg.slogdet(Sigma_unc)[1]
    )
    pval = 1 - sct.chi2.cdf(J, df=len(chosen))

    reject_apt = "reject" if pval < 0.05 else "fail to reject"
    print("\nQuestion 2: APT Hypothesis")
    print(f"  J-statistic: {J:.2f}, p-value: {pval:.4f} -> {reject_apt} at 5% level")
    

    print("\nQuestion 3: Zero-Beta Hypothesis Tests")
    for factor in ffn:
        
        other_factors = [x for x in ffn if x != factor]
    
        regs_zero_beta = sm.OLS(
            endog=retschosen[chosen],
            exog=sm.add_constant(retschosen[other_factors])
        ).fit()

        resid_con_zero_beta = regs_zero_beta.resid
        Sigma_con_zero_beta = resid_con_zero_beta.T @ resid_con_zero_beta / resid_con_zero_beta.shape[0]
        J_zero_beta = resid_unc.shape[0] * (
            np.linalg.slogdet(Sigma_con_zero_beta)[1] - np.linalg.slogdet(Sigma_unc)[1]
        )
        pval_zero_beta = 1 - sct.chi2.cdf(J_zero_beta, df=len(chosen))

        reject_zero_beta = "reject" if pval_zero_beta < 0.05 else "fail to reject"
        print(f"  Factor: {factor} Zero-Beta Hypothesis: J-statistic = {J_zero_beta:.2f}, p-value = {pval_zero_beta:.4f} -> {reject_zero_beta} at 5% level")
    
    print("\n")
    