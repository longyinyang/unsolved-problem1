import pandas as pd
import numpy as np
from datetime import datetime
from solution.utils import (get_quarter_cumul_income,
                            weighted_error,
                            get_report_date,
                            get_ticker_property)


def get_quarter(date):
    report_date_dt = datetime.strptime(date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    return quarter


def get_revenue_ttm(test_report_date):
    """
    营业收入的ttm = 前四季度单季度营业收入的和
    如果当前是2季度，则返回 ttm * 2 / 4
    """
    # 获取当前报告日之前的四个季度报告日
    prev_report_dates = [get_report_date(test_report_date, -i) for i in range(1, 9)]

    ret = pd.DataFrame()
    for report_date in prev_report_dates:
        quarterly_income = get_quarter_cumul_income(report_date, cumul=False)[['ticker', 'quarter_revenue']]
        if ret.empty:
            ret = quarterly_income
            ret['ttm'] = ret['quarter_revenue']
        else:
            ret = pd.merge(ret, quarterly_income, on='ticker')
            ret['ttm'] += ret['quarter_revenue']
        ret.rename(columns={'quarter_revenue': 'quarter_revenue_' + report_date}, inplace=True)

    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    ret['ttm'] = ret['ttm'] * quarter / 8.0

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(),
                                          get_report_date(test_report_date, -1), ['industry', 'market_value'])
    industry = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ret = pd.merge(ret, industry, on='ticker')
    ret = pd.merge(ret, market_value, on='ticker')

    # 去掉所有na
    ret.dropna(inplace=True)
    return ret


def get_ewma_revenue(test_report_date, alpha, s):
    """
    :param test_report_date: 报告日期
    :param alpha: 该系数越高，说明对过去值赋予的权重越低
    :param s:
    :return: ['ticker', 'ewma']
    """

    prev_report_dates = [get_report_date(test_report_date, -i) for i in range(1, s+2)]

    ret = pd.DataFrame()
    for report_date in prev_report_dates:
        quarterly_income = get_quarter_cumul_income(report_date, cumul=False)[['ticker', 'quarter_revenue']]
        if ret.empty:
            ret = quarterly_income
        else:
            ret = pd.merge(ret, quarterly_income, on='ticker')
        ret.rename(columns={'quarter_revenue': report_date}, inplace=True)

    #  以递归的方式计算ewma
    def cal_ewma(test_date):
        if test_date == prev_report_dates[-2]:
            return alpha * ret[test_date] + (1 - alpha) * ret[prev_report_dates[-1]]
        else:
            return alpha * ret[test_date] + (1 - alpha) * cal_ewma(get_report_date(test_date, -1))

    ewma = cal_ewma(get_report_date(test_report_date, -1))
    ewma = pd.DataFrame(ewma, columns=['ewma'])
    ewma = pd.concat([ret['ticker'], ewma], axis=1)

    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    ewma['ewma'] = ewma['ewma'] * quarter

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(),
                                          get_report_date(test_report_date, -1), ['industry', 'market_value'])
    industry = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ewma = pd.merge(ewma, industry, on='ticker')
    ewma = pd.merge(ewma, market_value, on='ticker')

    # 去掉所有na
    ewma.dropna(inplace=True)

    return ewma


def cal_industry_ratio(test_report_date, year):
    """
    计算行业调整指数
    return_every = True 时， 返回元组（ind_dict, ticker_dict）
    """

    q1 = pd.DataFrame()
    q2 = pd.DataFrame()
    q3 = pd.DataFrame()
    q4 = pd.DataFrame()
    data_classified_by_quarter = [q1, q2, q3, q4]
    for y in range(year):
        sum_data = pd.DataFrame()
        for i in range(1, 5):
            back_q_num = y * 4 + i
            prev_report_date = get_report_date(test_report_date, -back_q_num)
            quarter = get_quarter(prev_report_date)
            revenue_data = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]

            # 生成总数
            if i == 1:
                sum_data = revenue_data[['ticker', 'quarter_revenue']]
            else:
                sum_data = pd.merge(sum_data, revenue_data[['ticker', 'quarter_revenue']], on='ticker')
                sum_data['quarter_revenue'] = sum_data['quarter_revenue_x'] + sum_data['quarter_revenue_y']
                sum_data.drop(['quarter_revenue_x', 'quarter_revenue_y'], axis=1, inplace=True)

        sum_data.rename(columns={'quarter_revenue': 'sum'}, inplace=True)

        # 计算占比
        for i in range(1, 5):
            back_q_num = y * 4 + i
            prev_report_date = get_report_date(test_report_date, -back_q_num)
            quarter = get_quarter(prev_report_date)
            revenue_data = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]
            revenue_data = pd.merge(revenue_data, sum_data, on='ticker')
            revenue_data['ratio'] = revenue_data['quarter_revenue'] / revenue_data['sum']

            # 合并industry
            industry = get_ticker_property(revenue_data['ticker'].tolist(),
                                                  prev_report_date, ['industry'])
            revenue_data = pd.merge(revenue_data, industry, on='ticker').drop(['quarter_revenue', 'sum'], axis=1)
            if y == 0:
                data_classified_by_quarter[quarter-1] = revenue_data
            else:
                data_classified_by_quarter[quarter-1] = pd.concat([data_classified_by_quarter[i-1], revenue_data], axis=0)

    d1 = pd.DataFrame()
    d2 = pd.DataFrame()
    d3 = pd.DataFrame()
    d4 = pd.DataFrame()
    industry_ratio_map = [d1, d2, d3, d4]
    for i in range(4):
        industry_ratio_map[i] = data_classified_by_quarter[i].groupby('industry').mean()


    return industry_ratio_map


def cal_ticker_ratio(test_report_date, year):
    """
    calculate ticker-ratio
    返回的list里的DataFrames以ticker为index
    """
    q1 = pd.DataFrame()
    q2 = pd.DataFrame()
    q3 = pd.DataFrame()
    q4 = pd.DataFrame()
    data_classified_by_quarter = [q1, q2, q3, q4]

    for y in range(year):
        sum_data = pd.DataFrame()
        for i in range(1, 5):
            back_q_num = y * 4 + i
            prev_report_date = get_report_date(test_report_date, -back_q_num)
            quarter = get_quarter(prev_report_date)
            revenue_data = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]

            # 生成总数
            if i == 1:
                sum_data = revenue_data[['ticker', 'quarter_revenue']]
            else:
                sum_data = pd.merge(sum_data, revenue_data[['ticker', 'quarter_revenue']], on='ticker')
                sum_data['quarter_revenue'] = sum_data['quarter_revenue_x'] + sum_data['quarter_revenue_y']
                sum_data.drop(['quarter_revenue_x', 'quarter_revenue_y'], axis=1, inplace=True)

        sum_data.rename(columns={'quarter_revenue': 'sum'}, inplace=True)
        for i in range(1, 5):
            back_q_num = y * 4 + i
            prev_report_date = get_report_date(test_report_date, -back_q_num)
            quarter = get_quarter(prev_report_date)
            revenue_data = get_quarter_cumul_income(prev_report_date, cumul=False)[['ticker', 'quarter_revenue']]
            revenue_data = pd.merge(revenue_data, sum_data, on='ticker')
            revenue_data['ratio'] = revenue_data['quarter_revenue'] / revenue_data['sum']

            if y == 0:
                data_classified_by_quarter[quarter - 1] = revenue_data
            else:
                data_classified_by_quarter[quarter - 1] = pd.concat([data_classified_by_quarter[i - 1], revenue_data],
                                                                    axis=0)

    d1 = pd.DataFrame()
    d2 = pd.DataFrame()
    d3 = pd.DataFrame()
    d4 = pd.DataFrame()
    ticker_ratio_map = [d1, d2, d3, d4]
    for i in range(4):
        ticker_ratio_map[i] = data_classified_by_quarter[i].groupby('ticker').mean()

    return ticker_ratio_map


def get_adjusted_ttm(test_report_date, year):
    ind_ratio_map = cal_industry_ratio(test_report_date, year)

    prev_report_dates = [get_report_date(test_report_date, -i) for i in range(1, 9)]

    ret = pd.DataFrame()
    for report_date in prev_report_dates:
        report_date_dt = datetime.strptime(report_date, '%Y-%m-%d')
        quarter = report_date_dt.month // 4 + 1
        dict = ind_ratio_map[quarter-1]['ratio'].to_dict()

        # 读取revenue数据
        quarterly_income = get_quarter_cumul_income(report_date, cumul=False)[['ticker', 'quarter_revenue']]
        # 读取市值与产业数据
        ticker_property = get_ticker_property(quarterly_income['ticker'].tolist(),
                                              get_report_date(test_report_date, -1), ['industry', 'market_value'])
        industry = ticker_property[['ticker', 'industry']]
        market_value = ticker_property[['ticker', 'market_value']]

        quarterly_income = pd.merge(quarterly_income, industry, on='ticker')
        quarterly_income = pd.merge(quarterly_income, market_value, on='ticker')
        quarterly_income['ratio_based_weight'] = quarterly_income['industry'].map(dict)

        if ret.empty:
            ret = quarterly_income
            ret['ttm'] = ret['quarter_revenue'] * ret['ratio_based_weight']
            ret.drop('ratio_based_weight', axis=1, inplace=True)
        else:
            ret = pd.merge(ret, quarterly_income, on='ticker')
            ret['ttm'] += ret['quarter_revenue'] * ret['ratio_based_weight']
            ret.drop('ratio_based_weight', axis=1, inplace=True)
        ret.rename(columns={'quarter_revenue': 'quarter_revenue_' + report_date}, inplace=True)

    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    ret['ttm'] = ret['ttm'] * quarter / 2.0

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(),
                                          get_report_date(test_report_date, -1), ['industry', 'market_value'])
    industry = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ret = pd.merge(ret, industry, on='ticker')
    ret = pd.merge(ret, market_value, on='ticker')

    # 去掉所有na

    return ret


def get_adjusted_again_ttm(test_report_date, year, ttm_term=4):
    ind_ratio_map= cal_ticker_ratio(test_report_date, year)
    every_ticker_map = cal_ticker_ratio(test_report_date, year)
    prev_report_dates = [get_report_date(test_report_date, -i) for i in range(1, ttm_term+1)]

    ret = pd.DataFrame()  # 用于存放分季度的数据
    for report_date in prev_report_dates:
        quarter = get_quarter(report_date)
        ind_map = ind_ratio_map[quarter-1]
        ticker_map = every_ticker_map[quarter-1]

        # 读取revenue数据
        quarterly_income = get_quarter_cumul_income(report_date, cumul=False)[['ticker', 'quarter_revenue']]
        # quarterly_income.rename(columns={'quarter_revenue': quarter}, inplace=True)

        # 读取市值与产业数据
        # ticker_property = get_ticker_property(quarterly_income['ticker'].tolist(),
        #                                       get_report_date(test_report_date, -1), ['industry', 'market_value'])
        # industry = ticker_property[['ticker', 'industry']]
        # market_value = ticker_property[['ticker', 'market_value']]

        # quarterly_income = pd.merge(quarterly_income, industry, on='ticker')
        # quarterly_income = pd.merge(quarterly_income, market_value, on='ticker')

        # 根据ticker_map中是否包含数据中的ticker将数据分成两组

        in_data = quarterly_income[quarterly_income['ticker'].isin(ticker_map.index)].copy()  # 找得到对应ticker的数据
        not_in_data = quarterly_income.drop(in_data.index).copy()  # 找不到dict中对应ticker的数据，用行业数据代替

        # 处理in_data
        in_data['ratio_based_weight'] = in_data['ticker'].map(ticker_map['ratio'].to_dict())
        # 处理not_in_data
        if not_in_data.empty:
            processed_data = in_data

        else:
            """
                ticker_property = get_ticker_property(not_in_data['ticker'].tolist(),
                         get_report_date(test_report_date, -1), ['industry', 'market_value'])
                industry = ticker_property[['ticker', 'industry']]
                not_in_data = pd.merge(not_in_data, industry, on='ticker')

                not_in_data['ratio'] = not_in_data['ticker'].map(ind_map['ratio'].to_dict())
                not_in_data.drop('industry', axis=1, inplace=True)
            """

            not_in_data['ratio_based_weight'] = 0.25

            # 纵向合并两个DF
            processed_data = pd.concat([in_data, not_in_data])

        if ret.empty:
            ret = processed_data
            ret['ttm'] = ret['quarter_revenue'] * ret['ratio_based_weight']
            ret.drop('ratio_based_weight', axis=1, inplace=True)
        else:
            ret = pd.merge(ret, processed_data, on='ticker')
            ret['ttm'] += ret['quarter_revenue'] * ret['ratio_based_weight']
            ret.drop('ratio_based_weight', axis=1, inplace=True)
        ret.rename(columns={'quarter_revenue': 'quarter_revenue_' + report_date}, inplace=True)

    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    ret['ttm'] = ret['ttm'] * quarter / (ttm_term/4.0)

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(),
                                          get_report_date(test_report_date, -1), ['industry', 'market_value'])
    industry = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ret = pd.merge(ret, industry, on='ticker')
    ret = pd.merge(ret, market_value, on='ticker')

    # 去掉所有na

    return ret

def revenue_predict(pred_report_date, tickers, alpha, year):
    """
    预测本期的营业收入
    如果要预测的ticker不在计算ttm的DataFrame中，那么以行业中市值相近的ticker替代
    """
    # 获取营业收入的ttm值
    # ret = get_ewma_revenue(pred_report_date, alpha, 4)  # 此处有一超参数
    # ret = get_revenue_ttm(pred_report_date)
    # ret = get_adjusted_ttm(pred_report_date, year)
    ret = get_adjusted_again_ttm(pred_report_date, year, 4)
    ret = ret[ret['ticker'].isin(tickers)]
    if len(tickers) > len(ret['ticker']):
        missing_tickers = [ticker for ticker in tickers if ticker not in ret['ticker'].tolist()]
        missing_tickers_property = get_ticker_property(missing_tickers,
                                                       get_report_date(pred_report_date, -1), ['industry', 'market_value'])

        # unsolved_ticker = []
        for ticker in missing_tickers_property['ticker'].tolist():
            market_value = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['market_value'].values[0]
            industry = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['industry'].values[0]
            temp = ret[ret['industry'] == industry]

            if temp.empty:
                # unsolved_ticker.append(ticker)
                continue
            else:
                closest_df = temp.iloc[(temp['market_value'] - market_value).abs().argsort()[:1]].copy()
                closest_df.iloc[0, 0] = ticker
                ret = pd.concat([ret, closest_df])
        # ret.drop(unsolved_ticker, inplace=True)
    return ret


def ttm_model(test_report_date, alpha, year):
    # 真实值
    # 获得测试报告的真实累计营收
    quarter_income_cumul = get_quarter_cumul_income(test_report_date)[['ticker', 'revenue']]
    tickers = quarter_income_cumul['ticker'].tolist()
    # 预测值
    pred_quarter_income = revenue_predict(test_report_date, tickers, alpha, year)
    pred_quarter_income.drop(['market_value', 'industry'], axis=1, inplace=True)

    ticker_property = get_ticker_property(tickers, test_report_date, ['industry', 'market_value'])
    property = ticker_property[['ticker', 'industry', 'market_value']]
    pred_quarter_income = pd.merge(pred_quarter_income, property, on='ticker')

    # 合并到同一个DataFrame里面
    ret = pd.merge(quarter_income_cumul, pred_quarter_income, on='ticker')

    # 还是有个别ticker 是nan， 暂且去除掉
    ret.dropna(inplace=True)

    print('Baseline model weighted error: {0}'.format(weighted_error(pred_y=ret['ttm'].values,
                                                                     test_y=ret['revenue'].values,
                                                                     market_values=ret['market_value'].values)))

    return


if __name__ == '__main__':
    """
    for i in range(30, 90, 5):
        i /= 100
        print('alpha:', i)
        ttm_model('2016-12-31', i, year=0)
    """
    # ttm_model('2016-06-30', 0.55, year=0)
    # ttm_model('2017-03-31', 0, 4)

    """
    
    for year in range(2, 7, 1):
        print('year =', year)
        ttm_model('2017-03-31', 0, year)
    """
    #  print(get_ewma_revenue('2017-12-31', 0.4, 8))

    for year in range(2, 7, 1):
        print('year=', year)
        ttm_model('2017-12-31', 0, year)
    # -- * --
