import pandas as pd
import numpy as np
import datetime as dt
# import urllib2
import quandl


# set display width
pd.set_option('display.width', 100)


def get_futures_info():
    """Retrieves futures contract data from Github."""
    df = pd.read_csv('https://raw.githubusercontent.com/haobruce/SysTrade/master/SysTrade_FuturesContracts.csv')
    return df


def get_strategy_info():
    """Retrieves strategy information from Github."""
    df = pd.read_csv('https://raw.githubusercontent.com/haobruce/SysTrade/master/SysTrade_Strategies.csv')
    df['AdjWeight'] = df['UnadjustedWeight'] * df['CostAdjustment']
    df = df[['Rule', 'Variation', 'UnadjustedWeight', 'CostAdjustment', 'AdjWeight', 'ScalarPooled']]
    return df


def construct_futures_symbols(symbol, start_year=2006, end_year=2016):
    """Constructs a list of futures contract codes for a
    particular symbol and time frame."""
    futures = []

    # append expiration month code to symbol name
    futures_info = get_futures_info()
    months = futures_info['ExpMonths'].loc[futures_info['Symbol'] == symbol].values[0]

    for y in range(start_year, end_year + 1):
        for m in months:
            futures.append("%s%s%s" % (symbol, m, y))
    return futures


def download_historical_prices(symbol):
    """Downloads futures pricing data from Quandl for a specific contract."""
    auth_token = 'g1CWzGxxg2WxNVbV5n9y'

    # add exchange prefix to symbol name
    futures_info = get_futures_info()
    prefix = futures_info['Exchange'].loc[futures_info['Symbol'] == symbol[:-5]].values[0]  # strip off month and year
    full_name = prefix + '/' + symbol

    # download prices from quandl using full_name
    prices = quandl.get(full_name, authtoken=auth_token)
    prices = prices['Settle']
    # add contract_sort in order to sort by year then by month using contract name
    prices = pd.DataFrame({'Settle': pd.Series(prices),
                           'Contract': symbol,
                           'Contract_Sort': symbol[-4:] + symbol[-5:-4] + symbol[:-5]})
    return prices


def compile_historical_prices(symbol, start_year=2006, end_year=2016):
    """Combines futures pricing data for contracts within specified date range
    for a specific symbol."""
    symbol_list = construct_futures_symbols(symbol, start_year, end_year)
    prices = download_historical_prices(symbol_list[0])
    for sym in symbol_list[1:]:
        next_prices = download_historical_prices(sym)
        prices = next_prices.append(prices)
    return prices


def get_active_contracts(symbol, full_prices):
    """Constructs a data frame of active contracts, i.e. next-nearest contract, by date
    within specified date range."""
    # find unique dates since fullPrices repeats date index as contracts overlap
    unique_dates = full_prices.sort_index().index.unique()
    # create data frame with unique dates as index
    df = pd.DataFrame({'Date': unique_dates})
    df = df.set_index(['Date'])
    df.insert(0, 'Symbol', symbol)
    # find active contract for each date in data frame
    for d in df.index:
        # check that there are at least two contracts available on a given date
        # sort by contract_sort but use contract name
        if len(full_prices.sort_index()[d:d].sort_values('Contract_Sort')['Contract'].index) >= 2:
            df.loc[d, 'Contract'] = full_prices.sort_index()[d:d].sort_values('Contract_Sort')['Contract'][1]
            df.loc[d, 'Contract_Sort'] = full_prices.sort_index()[d:d].sort_values('Contract_Sort')['Contract_Sort'][1]
    # delete empty rows
    df = df[df['Contract'] == df['Contract']]
    return df


def get_active_prices(symbol, full_prices):
    """Stitches together futures prices based on Panama Method."""
    df = get_active_contracts(symbol, full_prices)
    # add settle prices to most recent contract
    contract = df['Contract'].sort_values(ascending=False).unique()[0]
    prices = full_prices[full_prices['Contract'] == contract]
    # add column for unadjusted settle prices for easier checking
    df.loc[df['Contract'] == contract, 'SettleRaw'] = prices['Settle']
    df.loc[df['Contract'] == contract, 'Settle'] = prices['Settle']
    # stitch settle prices to remainder of contracts
    for contract in df.sort_values('Contract_Sort', ascending=False)['Contract'].unique()[1:]:
        prices = full_prices[full_prices['Contract'] == contract]
        df_prices = df[df['Settle'] == df['Settle']]  # exclude rows with missing prices
        df_prices = df_prices.loc[df_prices.index.isin(prices.index)]  # include only dates in both df and prices
        end_date = df_prices.index.min().to_pydatetime()  # identify earliest date
        adjustment = df['Settle'][end_date] - prices['Settle'][end_date]
        df.loc[df['Contract'] == contract, 'SettleRaw'] = prices['Settle']
        df.loc[df['Contract'] == contract, 'Settle'] = prices['Settle'] + adjustment
    return df


def get_forecast_inputs(symbol, start_year=2006, end_year=2016):
    """Constructs data frame with necessary data for all strategy forecast
     calculations, e.g. EWMAC, carry, etc."""
    full_prices = compile_historical_prices(symbol, start_year, end_year)
    # get data for active contract
    df = get_active_prices(symbol, full_prices)
    # add month and year to data frame
    df['Month'] = df['Contract'].str[-5]
    df['Year'] = pd.to_numeric(df['Contract'].str[-4:])

    # add previous month
    futures_info = get_futures_info()
    months = futures_info['ExpMonths'].loc[futures_info['Symbol'] == symbol].values[0]
    for i, month in enumerate(months):
        if i == 0:  # first month
            df.loc[df['Month'] == month, 'PrevMonth'] = months[-1]
            df.loc[df['Month'] == month, 'Year'] = df['Year']-1
        else:
            df.loc[df['Month'] == month, 'PrevMonth'] = months[i-1]

    # add data for prev contract
    df['PrevContract'] = df['Symbol'] + df['PrevMonth'] + df['Year'].astype(str)
    # add raw settle prices for prev contract
    for contract in df['PrevContract'].unique()[::-1]:
        prices = full_prices[full_prices['Contract'] == contract]
        df.loc[df['PrevContract'] == contract, 'PrevSettleRaw'] = prices['Settle']
    # remove NaN rows as some contracts have longer histories than others
    df = df[pd.notnull(df['PrevSettleRaw'])]

    # add data for return volatility based on raw price data
    df['ReturnDay'] = df['SettleRaw'] - df['SettleRaw'].shift(1)
    df = df[1:]  # drop first day without return
    df['ReturnDaySq'] = df['ReturnDay'] ** 2
    df['Variance'] = pd.ewma(df['ReturnDaySq'], span=36)
    df['PriceVolatility'] = df['Variance'] ** 0.5
    df['PriceVolatilityPct'] = df['PriceVolatility'] / df['SettleRaw']
    return df


def calc_ewmac_crossovers(forecast_inputs, fast_days, slow_days):
    """Constructs data frame comprised of volatility adjusted crossovers for a specified
    forecasts_input data frame and speed parameters for the EWMAC strategies."""
    df = forecast_inputs.copy()
    df['Fast'] = pd.ewma(df['Settle'], span=fast_days)
    df['Slow'] = pd.ewma(df['Settle'], span=slow_days)
    df['RawCrossover'] = df['Fast'] - df['Slow']
    df['VolAdjCrossover'] = df['RawCrossover'] / df['PriceVolatility']
    df['ScalarUnpooled'] = 10 / np.nanmedian(np.abs(df['VolAdjCrossover']))
    return df


def calc_ewmac_scalars(fast_days, slow_days):
    """Calculates pooled scalar value for EWMAC strategies."""
    scalar_list = []
    symbols_list = get_futures_info()['Symbol'].tolist()
    for symbol in symbols_list:
        forecast_inputs = get_forecast_inputs(symbol)
        ewmac_crossovers = calc_ewmac_crossovers(forecast_inputs, fast_days, slow_days)
        scalar_list.append(ewmac_crossovers['ScalarUnpooled'][0])
    return np.median(scalar_list)


def calc_ewmac_forecasts(forecast_inputs, fast_days, slow_days):
    """Constructs data frame comprised of forecasts for a specified
    forecasts_input data frame and speed parameters for the EWMAC strategies."""
    df = calc_ewmac_crossovers(forecast_inputs, fast_days, slow_days)
    variation = str(fast_days) + "," + str(slow_days)
    strategies = get_strategy_info()
    scalar_pooled = strategies.loc[strategies['Variation'] == variation]['ScalarPooled'].values[0]
    # uncomment line below if scalar_pooled to recalculate
    # scalar_pooled = calc_ewmac_scalars(fast_days, slow_days)
    df['ScalarPooled'] = scalar_pooled
    df['Forecast'] = df['VolAdjCrossover'] * df['ScalarPooled']
    df['ForecastCapped'] = df['Forecast']
    df['ForecastCapped'].loc[df['Forecast'] > 20] = 20
    df['ForecastCapped'].loc[df['Forecast'] < -20] = -20
    return df


def calc_carry_crossovers(forecast_inputs):
    """Constructs data frame comprised of forecasts for a specified
    forecasts_input data for the carry strategy."""
    df = forecast_inputs.copy()
    df['PrevLessActive'] = df['PrevSettleRaw'] - df['SettleRaw']
    df['PrevLessActiveAnn'] = df['PrevLessActive']
    df['VolAdjCarry'] = df['PrevLessActiveAnn'] / df['PriceVolatility']
    df['ScalarUnpooled'] = 10 / np.nanmedian(np.abs(df['VolAdjCarry']))
    return df


def calc_carry_scalars():
    """Calculates pooled scalar value for carry strategies."""
    scalar_list = []
    symbols_list = get_futures_info()['Symbol'].tolist()
    for symbol in symbols_list:
        forecast_inputs = get_forecast_inputs(symbol)
        carry_crossovers = calc_carry_crossovers(forecast_inputs)
        scalar_list.append(carry_crossovers['ScalarUnpooled'][0])
    return np.median(scalar_list)


def calc_carry_forecasts(forecast_inputs):
    """Constructs data frame comprised of forecasts for a specified
    forecasts_input data frame and speed parameters for the carry strategies."""
    df = calc_carry_crossovers(forecast_inputs)
    strategies = get_strategy_info()
    scalar_pooled = strategies.loc[strategies['Rule'] == 'CARRY']['ScalarPooled'].values[0]
    # uncomment line below if scalar_pooled to recalculate
    # scalar_pooled = calc_ewmac_scalars()
    df['ScalarPooled'] = scalar_pooled
    df['Forecast'] = df['VolAdjCarry'] * df['ScalarPooled']
    df['ForecastCapped'] = df['Forecast']
    df['ForecastCapped'].loc[df['Forecast'] > 20] = 20
    df['ForecastCapped'].loc[df['Forecast'] < -20] = -20
    return df


def calc_instrument_forecasts(symbol, start_year=2006, end_year=2016):
    """Constructs data frame comprised of instrument forecasts based on available
    strategies and weights."""
    forecast_inputs = get_forecast_inputs(symbol, start_year, end_year)
    df = forecast_inputs.copy()
    # loop through strategies and add strategy forecasts as separate columns
    strategies = get_strategy_info()
    for i in list(range(0, len(strategies))):
        if strategies.loc[i, 'Rule'] == 'EWMAC':
            fast = int(strategies.loc[i, 'Variation'].split(',')[0])
            slow = int(strategies.loc[i, 'Variation'].split(',')[1])
            strategy_name = strategies.loc[i, 'Rule'] + strategies.loc[i, 'Variation']
            df[strategy_name] = calc_ewmac_forecasts(forecast_inputs, fast, slow)['ForecastCapped']
        elif strategies.loc[i, 'Rule'] == 'CARRY':
            strategy_name = strategies.loc[i, 'Rule']
            df[strategy_name] = calc_carry_forecasts(forecast_inputs)['ForecastCapped']

    # calculate instrument forecast as weighted average of strategy forecasts
    forecasts = df[['EWMAC16,64', 'EWMAC32,128', 'EWMAC64,256', 'CARRY']]
    weights = strategies['AdjWeight']
    df['InstrumentForecast'] = np.dot(forecasts, weights)

    # calculate instrument value vol
    futures_info = get_futures_info()
    df['BlockSize'] = futures_info.loc[futures_info['Symbol'] == symbol, 'BlockSize'].values[0]
    df['BlockValue'] = df['SettleRaw'] * df['BlockSize'] * 0.01
    df['InstrumentCurVol'] = df['BlockValue'] * df['PriceVolatilityPct'] * 100

    # *************************************************************************
    # need to add functionality to handle historical exchange rates
    # update instrument value volatility once exchange rates are available
    exchange_rate = futures_info.loc[futures_info['Symbol'] == symbol, 'FX']
    df['InstrumentValueVol'] = df['InstrumentCurVol']
    # *************************************************************************

    return df[['Symbol', 'Contract', 'SettleRaw', 'PriceVolatility', 'PriceVolatilityPct', 'InstrumentForecast',
               'BlockSize', 'BlockValue', 'InstrumentValueVol']]


# *************************************************************************
# set end_year = current year; start_date = beginning of previous year
# *************************************************************************
def run_backtest(symbols_list=['ES', 'TY'], start_date=dt.date(2015, 1, 1), starting_capital = 100000,
                 volatility_target = 0.25):
    """Conducts backtest of available strategies on specified futures contracts over
    specified period of time."""
    # set scalar variables
    start_year = start_date.year
    end_year = dt.date.today().year
    min_date = pd.Timestamp(dt.date(1900, 1, 1))
    max_date = pd.Timestamp(dt.date.today())
    instrument_weight = 1 / len(symbols_list)
    # *************************************************************************
    # change the multiplier to be dynamic based on number of instruments
    instrument_diversifier_multiplier = 1.41
    # *************************************************************************
    position_inertia = 0.1

    # construct dictionary to combine forecast with position and gain loss
    data_dict = {}
    for symbol in symbols_list:
        df = calc_instrument_forecasts(symbol, start_year, end_year)
        # placeholders below
        df['VolatilityScalar'] = 0.0
        df['SubsystemPosition'] = 0.0
        df['SystemPosition'] = 0.0
        df['StartingPosition'] = 0.0
        df['EndingPosition'] = 0.0
        df['PositionChange'] = 0.0
        df['PositionCost'] = 0.0
        df['PositionValue'] = 0.0
        df['GainLossCum'] = 0.0
        data_dict[symbol] = df
        if df.index.min() > min_date: min_date = df.index.min()
        if df.index.max() < max_date: max_date = df.index.max()

    # construct data frame from min and max dates
    min_date += pd.Timedelta(days=90)  # start back test 90 days after min date to ensure ample data for forecasts
    backtest_df = pd.DataFrame({'Date': pd.bdate_range(min_date, max_date)})
    backtest_df = backtest_df.set_index(['Date'])
    backtest_df['PortfolioValue'] = starting_capital
    backtest_df['DailyCashTargetVol'] = starting_capital * volatility_target / (256 ** 0.5)
    backtest_df['TotalPositionCost'] = 0.0
    backtest_df['TotalPositionValue'] = 0.0
    backtest_df['TotalGainLoss'] = 0.0

    # iterate through each date in df to retrieve ForecastCapped and InstrumentValueVol
    for key in data_dict.keys():
        for i in list(range(0, len(backtest_df))):
            # check if date in df exists in data_dict
            if backtest_df.index[i] in data_dict[key].index:
                active_date = backtest_df.index[i]

                # update capital balance and volatility targets based on gain loss in backtest_df
                if i != 0:  # skip first day
                    backtest_df['TotalPositionCost'][active_date] += data_dict[key]['PositionCost'][prev_date]
                    backtest_df['TotalPositionValue'][active_date] += data_dict[key]['PositionValue'][prev_date]
                    backtest_df['TotalGainLoss'][active_date] += data_dict[key]['GainLossCum'][prev_date]
                    backtest_df['PortfolioValue'][active_date] = starting_capital + backtest_df['TotalGainLoss'][active_date]
                    backtest_df['DailyCashTargetVol'][active_date] = backtest_df['PortfolioValue'][active_date] * \
                                                                     volatility_target / (256 ** 0.5)

                data_dict[key]['VolatilityScalar'][active_date] = backtest_df['DailyCashTargetVol'][active_date] / \
                                                                  data_dict[key]['InstrumentValueVol'][active_date]
                data_dict[key]['SubsystemPosition'][active_date] = data_dict[key]['InstrumentForecast'][active_date] / \
                                                                   10.0 * data_dict[key]['VolatilityScalar'][active_date]
                data_dict[key]['SystemPosition'][active_date] = data_dict[key]['SubsystemPosition'][active_date] * \
                                                                instrument_weight * instrument_diversifier_multiplier
                if i != 0:  # skip first day
                    data_dict[key]['StartingPosition'][active_date] = data_dict[key]['EndingPosition'].loc[prev_date]

                # determine trade based on starting_position, ending_position and system_position
                # define variable to minimize space
                starting_position = data_dict[key]['StartingPosition'][active_date]
                ending_position = starting_position
                system_position = data_dict[key]['SystemPosition'][active_date]
                block_size = data_dict[key]['BlockSize'][active_date]
                block_price = data_dict[key]['SettleRaw'][active_date]

                if starting_position == 0 or (np.abs((system_position - starting_position) / starting_position) >
                                                  position_inertia):
                    ending_position = np.round(system_position, 0)
                data_dict[key]['EndingPosition'][active_date] = ending_position
                data_dict[key]['PositionChange'][active_date] = ending_position - starting_position
                if i != 0:  # skip first day; else set PositionCost equal to previous value
                    data_dict[key]['PositionCost'][active_date] = data_dict[key]['PositionCost'].loc[prev_date]
                data_dict[key]['PositionCost'][active_date] += (ending_position - starting_position) * \
                                                               block_size * block_price
                data_dict[key]['PositionValue'][active_date] = ending_position * block_size * block_price
                data_dict[key]['GainLossCum'][active_date] = data_dict[key]['PositionValue'][active_date] - \
                                                             data_dict[key]['PositionCost'][active_date]

                prev_date = active_date

    return backtest_df, data_dict
