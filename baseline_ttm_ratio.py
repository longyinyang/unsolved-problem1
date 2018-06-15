import xgboost as xgb
import pandas as pd
import numpy as np
from datetime import datetime
from solution.utils import (get_quarter_cumul_income,
                            weighted_error,
                            get_report_date,
                            get_ticker_property)


def get_revenue_ttm_ratio(test_report_date, year):
    """
    营业收入的ttm ratio = 前四季度单季度营业收入增长率的均值
    为方便计算，增长率取对数增长率，那么增长率的均值 = 1/4 *(log(x_4) - log(x_0))

    修改：
        增加year: 过去year年到过去一年数据的增长率平均值
    """
    # 获取当前报告日之前的第一个季度报告日
    prev_report_date = get_report_date(test_report_date, -4)
    ret = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]

    # 获取当前报告日之前的第五个季度报告日
    prev_report_date = get_report_date(test_report_date, -4*year)
    temp = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]
    ret = pd.merge(ret, temp, on='ticker')
    ret.dropna(inplace=True)

    ret['ttm_ratio'] = (np.log(ret['quarter_revenue_x']) - np.log(ret['quarter_revenue_y'])) * (year - 1)

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(), test_report_date, ['industry', 'market_value'])
    industry = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ret = pd.merge(ret, industry, on='ticker')
    ret = pd.merge(ret, market_value, on='ticker')

    # 去掉所有na
    ret.dropna(inplace=True)
    return ret


def revenue_predict(pred_report_date, tickers, year):
    """
    预测本期的营业收入

    如果要预测的ticker不在计算ttm的DataFrame中，那么以行业中市值相近的ticker替代
    """
    # 获取营业收入的ttm值
    ret = get_revenue_ttm_ratio(pred_report_date, year)
    ret = ret[ret['ticker'].isin(tickers)]
    if len(tickers) > len(ret['ticker']):
        missing_tickers = [ticker for ticker in tickers if ticker not in ret['ticker'].tolist()]
        missing_tickers_property = get_ticker_property(missing_tickers, pred_report_date, ['industry', 'market_value'])
        for ticker in missing_tickers_property['ticker'].tolist():
            market_value = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['market_value'].values[0]
            industry = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['industry'].values[0]
            temp = ret[ret['industry'] == industry]
            closest_df = temp.iloc[(temp['market_value'] - market_value).abs().argsort()[:1]].copy()
            closest_df.iloc[0, 0] = ticker
            ret = pd.concat([ret, closest_df])
    return ret


def ttm_ratio_model(test_report_date, year):
    # 真实值
    # 获得测试报告的真实累计营收
    quarter_income_cumul = get_quarter_cumul_income(test_report_date)[['ticker', 'revenue']]
    tickers = quarter_income_cumul['ticker'].tolist()
    # 预测值
    pred_quarter_income = revenue_predict(test_report_date, tickers, year)

    # 获得测试上一期的累计营收
    prev_report_date = get_report_date(test_report_date, num_quarter=-4)
    prev_quarter_cumul_income = get_quarter_cumul_income(prev_report_date, cumul=True)[['ticker', 'revenue']]
    prev_quarter_cumul_income.rename(columns={'revenue': 'prev_quarter_revenue_cumul'}, inplace=True)

    # 获得测试上一期的非累计营收
    prev_quarter_income = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]
    prev_quarter_income.rename(columns={'quarter_revenue': 'prev_quarter_revenue'}, inplace=True)

    # 合并到同一个DataFrame里面
    temp = pd.merge(prev_quarter_cumul_income, prev_quarter_income, on='ticker')
    temp2 = pd.merge(temp, pred_quarter_income, on='ticker')
    ret = pd.merge(temp2, quarter_income_cumul, on='ticker')

    # 还是有个别ticker 是nan， 暂且去除掉
    ret.replace([np.inf, -np.inf], np.nan, inplace=True)
    ret.dropna(inplace=True)

    # 预测 = 上一季累计 + 上一季非累计 * ttm_ratio
    # 如果是一季度，那么不需要上一季累计的值
    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    """
    if quarter == 1:
        ret['ttm'] = (1 + ret['ttm_ratio']) * ret['prev_quarter_revenue']
    else:
        ret['ttm'] = ret['prev_quarter_revenue_cumul'] + (1 + ret['ttm_ratio']) * ret['prev_quarter_revenue']
    """
    ret['ttm'] = (1 + ret['ttm_ratio']) * ret['prev_quarter_revenue_cumul']
    print('Baseline model weighted error: {0}'.format(weighted_error(pred_y=ret['ttm'].values,
                                                                     test_y=ret['revenue'].values,
                                                                     market_values=ret['market_value'].values)))

    return


if __name__ == '__main__':
    for year in range(1, 2):
        print('year:', year)
        ttm_ratio_model('2017-06-30', year)
