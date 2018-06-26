import pandas as pd
from datetime import datetime
from solution.utils import (get_quarter_cumul_income,
                            weighted_error,
                            get_report_date,
                            get_ticker_property)
import matplotlib.pyplot as plt


def get_quarter(date):
    report_date_dt = datetime.strptime(date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    return quarter


def get_ewma_revenue(test_report_date, ttm_term=4, alpha=0.55, industry=None):
    """
    :param test_report_date: 报告日期
    :param alpha: 该系数越高，说明对过去值赋予的权重越低
    :param ttm_term: 计算ewma的过去期数，虽然不是ttm，这样命名是为了模型系数统一
    :return: ['ticker', 'ewma']
    """

    prev_report_dates = [get_report_date(test_report_date, -i) for i in range(1, ttm_term + 2)]

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

    ewma_data = cal_ewma(get_report_date(test_report_date, -1))
    ewma_data = pd.DataFrame(ewma_data, columns=['ewma'])
    ewma_data = pd.concat([ret['ticker'], ewma_data], axis=1)

    report_date_dt = datetime.strptime(test_report_date, '%Y-%m-%d')
    quarter = report_date_dt.month // 4 + 1
    ewma_data['ewma'] = ewma_data['ewma'] * quarter

    # 添加 行业和市值 信息
    ticker_property = get_ticker_property(ret['ticker'].tolist(),
                                          get_report_date(test_report_date, -1), ['industry', 'market_value'])
    industry_info = ticker_property[['ticker', 'industry']]
    market_value = ticker_property[['ticker', 'market_value']]
    ewma_data = pd.merge(ewma_data, industry_info, on='ticker')
    ewma_data = pd.merge(ewma_data, market_value, on='ticker')

    # 去掉所有na
    ewma_data.dropna(inplace=True)
    # """
    if industry is None:
        pass
    else:
        ewma_data = ewma_data[ewma_data['industry'].isin(list(industry))]
    # """
    return ewma_data


def revenue_predict(pred_report_date, tickers, alpha=0.5, ttm_term=4, year=4, industry=None):
    """
    预测本期的营业收入
    如果要预测的ticker不在计算ttm的DataFrame中，那么以行业中市值相近的ticker替代
    method可选：'ttm', 'ewma', 'ind_ttm', 'ticker_ttm'
    """

    # 获取营业收入的ttm值

    ret = get_ewma_revenue(pred_report_date, ttm_term, alpha, industry)

    ret = ret[ret['ticker'].isin(tickers)]
    if len(tickers) > len(ret['ticker']):
        missing_tickers = [ticker for ticker in tickers if ticker not in ret['ticker'].tolist()]
        missing_tickers_property = get_ticker_property(missing_tickers,
                                                       get_report_date(pred_report_date, -1),
                                                       ['industry', 'market_value'])
        if industry is None:
            pass
        else:
            missing_tickers_property = missing_tickers_property[missing_tickers_property['industry'].isin(industry)]

        # unsolved_ticker = []
        for ticker in missing_tickers_property['ticker'].tolist():
            market_value = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['market_value'].values[0]
            industry_info = missing_tickers_property[missing_tickers_property['ticker'] == ticker]['industry'].values[0]
            temp = ret[ret['industry'] == industry_info]

            if temp.empty:
                # unsolved_ticker.append(ticker)
                continue
            else:
                closest_df = temp.iloc[(temp['market_value'] - market_value).abs().argsort()[:1]].copy()
                closest_df.iloc[0, 0] = ticker
                # try:
                 #   last_q_revenue_data =
                ret = pd.concat([ret, closest_df])

        # ret.drop(unsolved_ticker, inplace=True)
    return ret


def ttm_model(test_report_date, alpha, ttm_term, year, industry=None):
    # 真实值
    # 获得测试报告的真实累计营收
    quarter_income_cumul = get_quarter_cumul_income(test_report_date)[['ticker', 'revenue']]
    tickers = quarter_income_cumul['ticker'].tolist()
    # 预测值
    pred_quarter_income = revenue_predict(test_report_date, tickers, alpha, ttm_term, year, industry)
    pred_quarter_income.drop(['market_value', 'industry'], axis=1, inplace=True)

    ticker_property = get_ticker_property(tickers, test_report_date, ['industry', 'market_value'])
    property = ticker_property[['ticker', 'industry', 'market_value']]
    pred_quarter_income = pd.merge(pred_quarter_income, property, on='ticker')

    # 合并到同一个DataFrame里面
    ret = pd.merge(quarter_income_cumul, pred_quarter_income, on='ticker')

    ret.dropna(inplace=True)

    # 追踪数据
    ret.to_csv('../tracing_data_{0}.csv'.format(test_report_date))

    w_err = weighted_error(pred_y=ret['ewma'].values, test_y=ret['revenue'].values,
                           market_values=ret['market_value'].values)
    print('Baseline model weighted error: {0}'.format(w_err))

    return w_err


if __name__ == '__main__':
    # report_dates = [get_report_date('2018-03-31', -i) for i in range(1, 9)]
    report_dates = ['2017-06-30']
    t_term = 4  # 这也是一个超参数
    print('ttm_term=', t_term)
    models = ['ewma_0.4', 'ewma_0.45','ewma_0.5', 'ewma_0.55', 'ewma_0.6', 'ewma_0.65', 'ewma_0.7']
    errors = []

    result = pd.DataFrame(index=models)
    for report_date in report_dates:
        print('report_date:', report_date)
        for a in range(40, 75, 5):
            alpha = a/100
            errors.append(ttm_model(report_date, alpha, t_term, year=None, industry=['银行'])) # 此处可添加industry参数
        result[report_date] = errors
        errors.clear()

    result = result.transpose()
    result.sort_index(inplace=True)
    result.plot(grid=True)
    result.to_csv('result_01.csv')
    plt.show()
