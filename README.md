# unsolved-problem1
unsolved
import pandas as pd
import numpy as np


def get_industry(ticker, date):
    """ ticker 处输入字符串 """
    ticker = prune_zeros(ticker)
    f_date = pd.to_datetime(date)
    try:
        return industry_map[(industry_map.ticker == ticker) &
                            (industry_map.intoDate <= f_date) &
                            (industry_map.outDate >= f_date)].industryName1.iloc[0]
    except IndexError:
        return np.nan


def prune_zeros(ticker):
    return ticker.lstrip('0')


path_of_market_data = 'E:/workarea/test/tianchi/Market Data.csv'
path_of_industry_chart = 'E:/workarea/test/tianchi/sw_industry_map2.csv'
market_data = pd.read_csv(path_of_market_data)
industry_map = pd.read_csv(path_of_industry_chart)
industry_map.fillna('2018/6/1', inplace=True)

#  转换日期格式
market_data.END_DATE_ = pd.to_datetime(market_data.END_DATE_, format='%Y-%m-%d')
industry_map.intoDate = pd.to_datetime(industry_map.intoDate, format='%Y-%m-%d')
industry_map.outDate = pd.to_datetime(industry_map.outDate, format='%Y-%m-%d')

#  选取2015年以后的数据
market_data = market_data[market_data.END_DATE_ >= '2015-01-01']

industries = market_data.TYPE_NAME_CN.unique()
group_by_industry = market_data.groupby('TYPE_NAME')

#  pivot_table = pd.pivot_table(market_data, index='TYPE_NAME_CN',
#                             values=['TURNOVER_VALUE', 'MARKET_VALUE'], aggfunc=[np.mean, np.std, len])
#  balance_sheet = pd.read_excel('E:/workarea/test/tianchi/Balance Sheet.xls')
#  income_statement = pd.read_excel('E:/workarea/test/tianchi/Income Statement.xls')
balance_sheets = []
income_statements = []
sheet_names = ['一般工商业_General Business', '保险业_Insurance', '银行业_Bank', '证券业_Securities']
for i in sheet_names:
    sheet = pd.read_excel('E:/workarea/test/tianchi/Balance Sheet.xls', sheet_name=i)
    balance_sheets.append(sheet)
    statement = pd.read_excel('E:/workarea/test/tianchi/Income Statement.xls', sheet_name=i)
    income_statements.append(statement)

join_sheets = []
for n in range(4):
    join_sheets.append(pd.merge(balance_sheets[n], income_statements[n], how='inner',
                                on=['TICKER_SYMBOL_股票代码', 'END_DATE_截止日期']))
    join_sheets[n].END_DATE_截止日期 = pd.to_datetime(join_sheets[n].END_DATE_截止日期, format='%Y-%m-%d')
    industry_name = pd.Series(np.empty(len(join_sheets[n])), dtype=np.string_, name='type')
    join_sheets[n].TICKER_SYMBOL_股票代码 = join_sheets[n].TICKER_SYMBOL_股票代码.apply(str)
    join_sheets[n] = pd.concat([join_sheets[n], industry_name], axis=1)
    join_sheets[n] = join_sheets[n][join_sheets[n].END_DATE_截止日期 >= '2015-01-01']
    join_sheets[n].type = join_sheets[n].apply(lambda x:
                                               get_industry(x.TICKER_SYMBOL_股票代码, x.END_DATE_截止日期), axis=1)
