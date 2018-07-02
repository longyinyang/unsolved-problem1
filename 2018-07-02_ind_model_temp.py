from solution.feature_engineering.feature_preprocess import features_train_test_main
from solution.utils import weighted_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  BayesianRidge,
                                  HuberRegressor,
                                  TheilSenRegressor,
                                  SGDRegressor)
import pandas as pd
import matplotlib.pyplot as plt


UNIT = 100000000


def main(train_report_dates, test_report_date, industry, method = 'linear_regression'):
    train, test, scaler = features_train_test_main(train_report_dates,
                                                   test_report_date,
                                                   industry,
                                                   merge_balance_sheet=False)
    drop_x_cols = ['quarter_revenue',
                   'market_value',
                   'ticker',
                   'year',
                   'set',
                   'quarter_revenue_norm',
                   'market_value_norm',
                   'board']
    print('Feature preprocess finished')
    train_x = train.drop(drop_x_cols, axis=1)
    train_y = train['quarter_revenue_norm']
    test_x = test.drop(drop_x_cols, axis=1)
    # rf = LinearRegression()
    # rf.fit(train_x, train_y)
    # model = xgb.XGBRegressor()
    # model = BayesianRidge()  # 3.080
    # model = HuberRegressor()  # 2.952
    # model = TheilSenRegressor()  # 2.986

    if method == 'linear_regression':
        model = LinearRegression()
    elif method == 'ridge':
        model = Ridge()
    elif method == 'B_ridge':
        model = BayesianRidge()
    elif method == 'Huber':
        model = HuberRegressor()
    elif method == 'Theil':
        model = TheilSenRegressor()

    model.fit(train_x, train_y)
    pred_y_scaled = model.predict(test_x)

    pred_y = scaler.inverse_transform(pred_y_scaled.reshape(-1, 1))
    test_error = weighted_error(pred_y=pred_y,
                                test_y=test['quarter_revenue'].values,
                                market_values=test['market_value'].values,
                                unit=UNIT)
    test_output = pd.concat([pd.DataFrame(pred_y), pd.DataFrame(test['quarter_revenue'].values)], axis=1)
    test_output.columns = ['pred_y', 'test_y']
    test_output.to_csv('../analysis/ind_model/test_output_{0}_{1}.csv'.format(industry, method))
    print('weighted error on test set is {0}'.format(test_error))

    return test_error


if __name__ == '__main__':

    methods = ['linear_regression', 'ridge', 'B_ridge', 'Huber', 'Theil']
    industries = ['综合', '农林渔牧']

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
    plt.rcParams['axes.unicode_minus'] = False

    for ind in industries:
        temp = pd.Series(index=methods)  # 存放同一个行业应用不同模型的结果
        for method in methods:
            err = main(train_report_dates=['2016-06-30', '2017-03-31', '2016-09-30', '2015-06-30'],
                       test_report_date='2017-06-30', industry=ind, method=method)  # LinearRegression, 2.96
            temp[method] = err

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(len(methods)), temp)
        ticks = ax.set_xticks(range(len(methods)))
        labels = ax.set_xticklabels(methods, rotation=30,  fontsize='small')
        ax.set_title(ind)
        plt.savefig('../analysis/ind_model/ind_fig/test_output_{0}.jpg'.format(ind))


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
