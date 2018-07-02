from solution.feature_engineering.feature_preprocess import features_train_test_main
from solution.utils import weighted_error, get_report_date, get_ticker_property
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression,
                                  Ridge)
from sklearn.preprocessing import PolynomialFeatures

UNIT = 100000000


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
        quarterly_income = pd.read_csv('../data/train_data/quarterly_income_{0}.csv'.format(report_date),
                                       usecols=['ticker', 'quarter_revenue'])
        quarterly_income['ticker'] = quarterly_income['ticker'].astype(str).str.zfill(6)

        if ret.empty:
            ret = quarterly_income
        else:
            ret = pd.merge(ret, quarterly_income, on='ticker', how='outer')
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
                ret = pd.concat([ret, closest_df])


        # ret.drop(unsolved_ticker, inplace=True)
    return ret


def main(train_report_dates, test_report_date, industry):
    train, test, scaler = features_train_test_main(train_report_dates,
                                                   test_report_date,
                                                   industry,
                                                   merge_balance_sheet=False,
                                                   merge_cash_flow_statement=False,
                                                   merge_income_statement=False)
    test['ticker'] = test['ticker'].map(lambda x: str(x)[:6])
    test.dropna(inplace=True)
    drop_x_cols = ['quarter_revenue',
                   'market_value',
                   'ticker',
                   'year',
                   'set',
                   'quarter_revenue_norm',
                   'market_value_norm',
                   'board',
                   'quarter',
                   'exchange_XSHE',
                   'exchange_XSHG',
                   'close_change']
    print('Feature preprocess finished')
    train_x = train.drop(drop_x_cols, axis=1)
    train_y = train['quarter_revenue_norm']
    test_x = test.drop(drop_x_cols, axis=1)
    # rf = LinearRegression()
    # rf.fit(train_x, train_y)
    # model = xgb.XGBRegressor()


    model = LinearRegression()
    model.fit(train_x, train_y)
    pred_y_scaled = model.predict(test_x)

    pred_y = scaler.inverse_transform(pred_y_scaled.reshape(-1, 1))

    pred_y_by_ewma = revenue_predict(test_report_date, test['ticker'], industry=[industry])
    pred_y_df = pd.DataFrame(pred_y, columns=['linear_model'])
    pred_y_df['ticker'] = test['ticker']

    pred_y_merge = pd.merge(pred_y_df, pred_y_by_ewma, on='ticker')
    pred_y_merge['merge'] = pred_y_merge['linear_model'] + pred_y_merge['ewma']

    pred_y_merge['result'] = \
        pred_y_merge.apply(lambda x:
                           0.3 * x['linear_model']+ 0.7 * x['ewma'] if np.abs(x['linear_model']-x['ewma'])/x['ewma']<0.5 else x['ewma'], axis=1)

    test_error = weighted_error(pred_y=pred_y_merge['result'].values,
                                test_y=test['quarter_revenue'].values,
                                market_values=test['market_value'].values,
                                unit=UNIT)
    print('weighted error on test set is {0}'.format(test_error))

    return


if __name__ == '__main__':
    main(train_report_dates=['2016-06-30', '2017-03-31', '2016-09-30', '2015-06-30'],
          test_report_date='2017-06-30',
          industry='房地产') #LinearRegression, 2.96

    # main(train_report_dates=['2016-06-30', '2017-03-31', '2016-09-30', '2015-06-30'],
    #      test_report_date='2017-06-30',
    #      industry='房地产') LinearRegression, 2.96
    #
    # main(train_report_dates=['2016-06-30', '2017-03-31', '2016-09-30'],
    #      test_report_date='2017-09-30',
    #      industry='非银金融') Ridge, 2.11

    # main(train_report_dates=['2016-06-30', '2017-03-31', '2016-09-30'],
    #      test_report_date='2016-06-30',
    #      industry='非银金融') Ridge
