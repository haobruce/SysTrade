import pandas as pd
import numpy as np
# import urllib2
import datetime
import quandl


def get_futures_info():
    df = pd.read_csv('https://raw.githubusercontent.com/haobruce/SysTrade/master/SysTrade_FuturesContracts.csv')
    return df


def construct_futures_symbols(symbol, start_year=2006, end_year=2016):
    """Constructs a list of futures contract codes for a
    particular symbol and time frame."""
    futures = []

    # append expiration month code to symbol name
    if symbol == 'C':
        months = 'HMNUZ'  # corn futures can expire in an additional month N
    elif symbol == 'NG':
        months = 'FGHJKMNQUVXZ'  # natural gas futures can expire in any month
    else:
        months = 'HMUZ'  # March, June, September and December delivery codes

    for y in range(start_year, end_year + 1):
        for m in months:
            futures.append("%s%s%s" % (symbol, m, y))
    return futures


def download_historical_prices(symbol):
    auth_token = 'g1CWzGxxg2WxNVbV5n9y'
    full_name = 'CME/' + symbol
    prices = quandl.get(full_name, authtoken=auth_token)
    prices = prices['Settle']
    # add contract_sort in order to sort by year then by month using contract name
    prices = pd.DataFrame({'Settle': pd.Series(prices), 'Contract': symbol, 'Contract_Sort': symbol[-4:]+symbol[:3] })
    return prices


def compile_historical_prices(symbol, start_year=2006, end_year=2016):
    symbol_list = construct_futures_symbols(symbol, start_year, end_year)
    prices = download_historical_prices(symbol_list[0])
    for sym in symbol_list[1:]:
        next_prices = download_historical_prices(sym)
        prices = next_prices.append(prices)
    return prices


def get_active_contracts(symbol, start_year=2006, end_year=2016):
    full_prices = compile_historical_prices(symbol, start_year, end_year)
    # find unique dates since fullPrices repeats date index as contracts overlap
    unique_dates = full_prices.sort_index().index.unique()
    # create data frame with unique dates as index
    df = pd.DataFrame({'Date': unique_dates})
    df = df.set_index(['Date'])
    df.insert(0, 'Symbol', symbol)
    # find active contract for each date in dataframe
    for dt in df.index:
        # check that there are at least two contracts available on a given date
        # sort by contract_sort but use contract name
        if len(full_prices.sort_index()[dt:dt].sort_values('Contract_Sort')['Contract'].index) >= 2:
            active_contract = full_prices.sort_index()[dt:dt].sort_values('Contract_Sort')['Contract'][1]
            df.loc[dt, 'Contract'] = active_contract
    # delete empty rows
    df = df[df['Contract'] == df['Contract']]
    return df


def get_active_prices(symbol, start_year=2006, end_year=2016):
    full_prices = compile_historical_prices(symbol, start_year, end_year)
    df = get_active_contracts(symbol, start_year, end_year)
    # add settle prices to most recent contract
    contract = df['Contract'].unique()[::-1][0]
    prices = full_prices[full_prices['Contract'] == contract]
    # add column for unadjusted settle prices for easier checking
    df.loc[df['Contract'] == contract, 'SettleRaw'] = prices['Settle']
    df.loc[df['Contract'] == contract, 'Settle'] = prices['Settle']
    # stitch settle prices to remainder of contracts
    for contract in df['Contract'].unique()[::-1][1:]:
        prices = full_prices[full_prices['Contract'] == contract]
        end_date = df[df['Settle'] == df['Settle']].sort_index()[:1].index.to_pydatetime()[0]
        adjustment = df['Settle'][end_date] - prices['Settle'][end_date]
        df.loc[df['Contract'] == contract, 'SettleRaw'] = prices['Settle']
        df.loc[df['Contract'] == contract, 'Settle'] = prices['Settle'] + adjustment
    return df


def get_forecast_inputs(symbol, start_year=2006, end_year=2016):
    full_prices = compile_historical_prices(symbol, start_year, end_year)
    # get data for active contract
    df = get_active_prices(symbol, start_year, end_year)
    # add month and year to data frame
    df['Month'] = df['Contract'].str[2]
    df['Year'] = pd.to_numeric(df['Contract'].str[-4:])
    # add previous month
    df.loc[df['Month'].str[0] == 'H', 'PrevMonth'] = 'Z'
    df.loc[df['Month'].str[0] == 'M', 'PrevMonth'] = 'H'
    df.loc[df['Month'].str[0] == 'U', 'PrevMonth'] = 'M'
    df.loc[df['Month'].str[0] == 'Z', 'PrevMonth'] = 'U'
    # for March decrease year by one to previous year
    df.loc[df['Month'].str[0] == 'H', 'Year'] = df['Year']-1

    # add data for prev contract
    df['PrevContract'] = df['Contract'].str[0:2] + df['PrevMonth'] + df['Year'].astype(str)
    # add raw settle prices for prev contract
    for contract in df['PrevContract'].unique()[::-1]:
        prices = full_prices[full_prices['Contract'] == contract]
        df.loc[df['PrevContract'] == contract, 'PrevSettleRaw'] = prices['Settle']

    # add data for return volatility based on raw price data
    df['ReturnDay'] = df['SettleRaw'] - df['SettleRaw'].shift(1)
    df = df[1:]  # drop first day without return
    df['ReturnDaySq'] = df['ReturnDay'] ** 2
    df['Variance'] = df['ReturnDaySq']
    # skip first row for volatility calculation
    lambda_36 = 2 / (36 + 1) # for 36 day look back
    for i in list(range(1, df.shape[0])):
        df['Variance'].iloc[i] = df['Variance'].iloc[i - 1] * (1 - lambda_36) + df['ReturnDaySq'].iloc[i] * lambda_36
    df['PriceVolatility'] = df['Variance'] ** 0.5
    df['PriceVolatilityPerc'] = df['PriceVolatility'] / df['SettleRaw']

    # df.sort_index(ascending=False, inplace=True)
    return df


def ewmac(forecast_inputs, fast_days, slow_days):
    df = forecast_inputs
    lambda_fast = 2 / (fast_days + 1)
    lambda_slow = 2 / (slow_days + 1)
    df['Fast'] = df['Settle']
    df['Slow'] = df['Settle']
    for i in list(range(1, df.shape[0])):
        df['Fast'].iloc[i] = df['Fast'].iloc[i - 1] * (1 - lambda_fast) + df['Settle'].iloc[i] * lambda_fast
        df['Slow'].iloc[i] = df['Slow'].iloc[i - 1] * (1 - lambda_slow) + df['Settle'].iloc[i] * lambda_slow
    df['RawCrossover'] = df['Fast'] - df['Slow']
    df['VolAdjCrossover'] = df['RawCrossover'] / df['PriceVolatility']
    df['ScalarUnpooled'] = 10 / np.median(np.abs(df['VolAdjCrossover']))
    df['ScalarPooled'] = df['ScalarUnpooled']  # placeholder until replaced by function
    df['Forecast'] = df['VolAdjCrossover'] * df['ScalarPooled']
    df['ForecastCapped'] = df['Forecast']
    df['ForecastCapped'].loc[df['Forecast'] > 20] = 20
    df['ForecastCapped'].loc[df['Forecast'] < -20] = -20
    return df


def carry(forecast_inputs):
    df = forecast_inputs
    df['PrevLessActive'] = df['PrevSettleRaw'] - df['SettleRaw']
    df['PrevLessActiveAnn'] = df['PrevLessActive']
    df['VolAdjCarry'] = df['PrevLessActiveAnn'] / df['PriceVolatility']
    df['ScalarUnpooled'] = 10 / np.median(np.abs(df['VolAdjCarry']))
    df['ScalarPooled'] = df['ScalarUnpooled']  # placeholder until replaced by function
    df['Forecast'] = df['VolAdjCarry'] * df['ScalarPooled']
    df['ForecastCapped'] = df['Forecast']
    df['ForecastCapped'].loc[df['Forecast'] > 20] = 20
    df['ForecastCapped'].loc[df['Forecast'] < -20] = -20
    return df
