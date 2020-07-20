# -*- coding: utf-8 -*-
from mexbots.strategy import Strategy
from mexbots.indicator import *
import pandas as pd
import numpy as np
import math
import sys
import time

breakout_in = 22
breakout_out = 5

file = "const.csv"
data = pd.read_csv(file, sep=",")  # куда фиксируется баланс
data = data[["balance", 'take', 'stop', 'hedge', 'entry', 'newHigh', 'newLow', 'lvl', 'limit', 'entryqty', 'qty']]

JPY = 'BTC/JPY'
USD = 'BTC/USD'

JPYX = 100000000    # множители для перевода пнл в человеческий вид из 0.0000000000001 в 0.001 например
USDX = 1000000

Coin = USD  # торгуемая пара см переменные выше
Multi = 0

movearound = 7
fibblvl = -2.163
limitlvl = 0.382
stopplacelvl = 0.5


if Coin == JPY:
    Multi = JPYX
elif Coin == USD:
    Multi = USDX


def truncate(f, n):
    # 'Truncates/pads a float f to n decimal places without rounding'
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def signal(ohlcv, ticker, sma1, sma2):
    s = np.array(ohlcv.close)
    s = np.append(s, ticker.last)
    s = pd.DataFrame(s)

    sma1 = sma(s, sma1)  # тема
    sma2 = sma(s, sma2)

    sma1 = np.array(sma1)
    sma1 = sma1.ravel()  # преобразование из pd.Dataframe в одномерный numpy массив
    sma2 = np.array(sma2)
    sma2 = sma2.ravel()

    sma11 = sma1[-2]
    sma22 = sma2[-2]

    delta = sma11 - sma22  # разница тема чтобы понять какая выше если значение положительно tema1(c параметром 7) выше-лонг

    sma33 = sma1[-1]
    sma44 = sma2[-1]

    delta1 = sma33 - sma44

    return delta, delta1


def stopladder(side, ohlcv, last, qty, strategy):
    if side == "long":
        if data.loc[0, "newHigh"] != 0:
            data.loc[0, "lvl"] = round(data.loc[0, "newHigh"] - (data.loc[0, "newHigh"] - data.loc[0, "stop"]) * stopplacelvl)
            data.to_csv(file)
        if ohlcv.high[-1] > data.loc[0, "newHigh"]:
            if data.loc[0, "stop"] < data.loc[0, "newLow"] <= data.loc[0, "lvl"]:
                data.loc[0, "stop"] = data.loc[0, "newLow"]
                strategy.order('S', 'sell', qty=qty, stop=data.loc[0, "newLow"] - 10)
            data.loc[0, "newHigh"] = ohlcv.high[-1]
            data.loc[0, "newLow"] = 0       # перемещение стопа в это условие
            data.to_csv(file)
        elif data.loc[0, "newLow"] == 0 and ohlcv.high[-1] < data.loc[0, "newHigh"]:
            data.loc[0, "newLow"] = ohlcv.low[-1]
            data.to_csv(file)
        elif data.loc[0, "newLow"] != 0 and ohlcv.low[-1] < data.loc[0, "newLow"]:
            data.loc[0, "newLow"] = ohlcv.low[-1]
            data.to_csv(file)
    elif side == "short":
        if data.loc[0, 'newLow'] != 0:
            data.loc[0, "lvl"] = round(data.loc[0, "newLow"] - (data.loc[0, "newLow"] - data.loc[0, "stop"]) * stopplacelvl)
            data.to_csv(file)
        if data.loc[0, "newLow"] == 0:
            data.loc[0, "newLow"] = ohlcv.low[-1]
            data.loc[0, "newHigh"] = 0
            data.to_csv(file)
        elif data.loc[0, "newLow"] != 0 and data.loc[0, "newHigh"] < ohlcv.high[-1]:
            data.loc[0, "newHigh"] = ohlcv.high[-1]
            data.to_csv(file)
        elif data.loc[0, "newLow"] != 0 and ohlcv.low[-1] < data.loc[0, "newLow"]:
            if data.loc[0, "stop"] > data.loc[0, "newHigh"] >= data.loc[0, "lvl"]:
                data.loc[0, "stop"] = data.loc[0, "newHigh"]
                strategy.order('L', 'buy', qty=qty, stop=data.loc[0, "newHigh"] + 10)
            data.loc[0, "newLow"] = ohlcv.low[-1]
            data.loc[0, "newHigh"] = 0      # перемещение стопа в это условие
            data.to_csv(file)


def stopsearch(ohlcv, ticker):
    long_entry_price = -999999999999999
    short_entry_price = 999999999999999

    maximum = 0
    minimum = round(last(highest(ohlcv.high, 22)) + 1)
    indexer1 = -100
    indexer2 = -100
    i = -1

    while i != -100:
        if ohlcv.close[i] > ohlcv.open[i] and ohlcv.volume[i] > ohlcv.volume[i - 1] and ohlcv.high[i] > ticker.last \
                and ohlcv.volume[i] > ohlcv.volume[i - 2] and ohlcv.high[i] > maximum and i >= indexer1:
            maximum = ohlcv.high[i]
            indexer1 = i
            long_entry_price = maximum
        elif ohlcv.close[i] < ohlcv.open[i] and ohlcv.volume[i] > ohlcv.volume[i - 1] and ohlcv.low[i] < ticker.last \
                and ohlcv.volume[i] > ohlcv.volume[i - 2] and ohlcv.low[i] < minimum and i >= indexer2:
            minimum = ohlcv.low[i]
            indexer2 = i
            short_entry_price = minimum
        i -= 1
    # ==================================================================================================================
    if indexer1 >= -9:
        indexer1 = indexer1 - movearound
        while indexer1 != - 1:
            if ohlcv.high[indexer1] > long_entry_price:
                long_entry_price = ohlcv.high[indexer1]
            indexer1 += 1
    elif indexer1 < -9:
        end = indexer1 + 7
        indexer1 = indexer1 - movearound
        while indexer1 != end:
            if ohlcv.high[indexer1] > long_entry_price:
                long_entry_price = ohlcv.high[indexer1]
            indexer1 += 1
    # ==================================================================================================================
    if indexer2 >= -9:
        indexer2 = indexer2 - movearound
        while indexer2 != - 1:
            if ohlcv.low[indexer2] < short_entry_price:
                short_entry_price = ohlcv.low[indexer2]
            indexer2 += 1
    elif indexer2 < -9:
        end = indexer2 + 7
        indexer2 = indexer2 - movearound
        while indexer2 != end:
            if ohlcv.low[indexer2] < short_entry_price:
                short_entry_price = ohlcv.low[indexer2]
            indexer2 += 1

    long_entry_price = long_entry_price + 10
    short_entry_price = short_entry_price - 10
    return long_entry_price, short_entry_price


def mylogic(ticker, ohlcv, position, balance, strategy):

    signall1 = signal(ohlcv, ticker, 7, 17)

    # разница тема чтобы понять какая выше если значение положительно sma1 выше-лонг
    delta1 = signall1[1]
    # и наоборот

    total = float(balance.BTC.total)
    fndstop = stopsearch(ohlcv, ticker)

    long_entry_price = fndstop[0]
    short_entry_price = fndstop[1]

    qty_lot = int(balance.BTC.free * 3 * ticker.last)

    print("qty", qty_lot)   # рассчитывает баланс для торговли кроссом

    if position.currentQty == 0:    # ставим позицию
        strategy.cancel_order_all()

        if total > data.loc[0, "balance"]:  # апдейтит баланс в файл при росте
            data.loc[0, "balance"] = total
            data.to_csv(file)

        elif total <= (data.loc[0, "balance"] * 0.8):  # отключение после потери 20% баланса
            data.loc[0, "balance"] = total
            data.to_csv(file)
            sys.exit()

        if ticker.last > strategy.stop and strategy.lshort == -1 and strategy.stopswitcher == 0:   # sleep after stop
            strategy.stopswitcher = 1
            if position.currentQty == 0:
                strategy.cancel_order_all()
                time.sleep(15 * 60 * 3)

        elif ticker.last < data.loc[0, 'stop'] and strategy.lshort == 1 and strategy.stopswitcher == 0:
            strategy.stopswitcher = 1
            if position.currentQty == 0:
                strategy.cancel_order_all()
                time.sleep(15 * 60 * 3)

        strategy.switcher = 0
        data.loc[0, "stop"] = 0
        data.loc[0, "take"] = 0
        data.loc[0, 'entry'] = 0
        data.loc[0, "hedge"] = 0
        data.loc[0, "newHigh"] = 0
        data.loc[0, "newLow"] = 0
        data.loc[0, "lvl"] = 0
        data.loc[0, "limit"] = 0
        data.loc[0, "entryqty"] = 0
        data.loc[0, "qty"] = 0
        data.to_csv(file)
# ======================================================================================================================
        if delta1 > 15:
            lenn = ticker.last - short_entry_price
            if 20 < lenn <= 200 and ticker.last > short_entry_price:
                take = ticker.last + lenn * 5
                round(take)
                strategy.lshort = 1
                strategy.stop = short_entry_price
                strategy.stopswitcher = 0
                data.loc[0, "stop"] = short_entry_price
                data.loc[0, "take"] = take
                data.loc[0, "hedge"] = ticker.last + lenn
                if qty_lot > round(qty_lot / (math.fabs(lenn / 100))):
                    qty_lot = round(qty_lot / (math.fabs(lenn / 100)))
                # buy
                data.loc[0, 'entryqty'] = qty_lot
                data.loc[0, 'qty'] = qty_lot
                strategy.order('L', 'buy', qty=qty_lot)
                strategy.order('S', 'sell', qty=qty_lot, stop=short_entry_price)
                strategy.order('S', 'sell', qty=round(qty_lot / 2), limit=ticker.last + lenn)
                data.to_csv(file)
            # ----------------------------------------------------------------------------------------------------------

        elif delta1 < -15:
            lenn = long_entry_price - ticker.last
            if 20 < lenn <= 200 and ticker.last < long_entry_price:
                take = ticker.last - lenn * 5
                round(take)
                strategy.lshort = -1
                strategy.stop = long_entry_price
                strategy.stopswitcher = 0
                data.loc[0, "stop"] = long_entry_price
                data.loc[0, "take"] = take
                data.loc[0, "hedge"] = ticker.last - lenn
                if qty_lot > round(qty_lot / (math.fabs(lenn / 100))):
                    qty_lot = round(qty_lot / (math.fabs(lenn / 100)))
                # sell
                data.loc[0, 'entryqty'] = qty_lot
                data.loc[0, 'qty'] = qty_lot
                strategy.order('S', 'sell', qty=qty_lot)
                strategy.order('L', 'buy', qty=qty_lot, stop=long_entry_price)
                strategy.order('L', 'buy', qty=round(qty_lot / 2), limit=ticker.last - lenn)
                data.to_csv(file)

    if position.currentQty != 0:
        valusd = strategy.position['currentQty']
        entry = strategy.position['avgEntryPrice']
        if data.loc[0, 'entry'] == 0:
            data.loc[0, 'entry'] = strategy.position['avgEntryPrice']
            data.to_csv(file)
        currentqty = position.currentQty
        llong = ticker.last - entry
        module = math.fabs(llong)
        pnl = (1 / entry - 1 / ticker.last) * Multi
        pnl = round(float(truncate(pnl, 10)), 4) * (valusd / math.fabs(valusd))

        if position.currentQty > 0:
            onepercent = 1

            if pnl < 5:
                stopladder("long", ohlcv, ticker.last, currentqty, strategy)     # стоп лесенкой

            if pnl != 0:
                onepercent = round(module / (pnl * 100))  # сколько пунктов на 1% прибыли

            currentqty = math.fabs(currentqty)
# ======================================================================================================================
            if delta1 < -5:  # закрытие при пересечении
                strategy.cancel_order_all()
                strategy.order('S', 'sell', qty=currentqty)
                data.loc[0, "stop"] = 0
                data.loc[0, "take"] = 0
                data.loc[0, 'entry'] = 0
                data.loc[0, "hedge"] = 0
                data.loc[0, "newHigh"] = 0
                data.loc[0, "newLow"] = 0
                data.loc[0, "lvl"] = 0
                data.loc[0, "limit"] = 0
                data.loc[0, "entryqty"] = 0
                data.loc[0, "qty"] = 0
                data.to_csv(file)

            elif ticker.last < data.loc[0, "hedge"] and delta1 > 0 and position.currentQty != 0 and len(strategy.open_orders()) != 2:
                strategy.cancel_order_all()

                strategy.order('S', 'sell', qty=currentqty, stop=data.loc[0, "stop"])

                if data.loc[0, "entryqty"] == currentqty:
                    strategy.order('S', 'sell', qty=round(currentqty / 2), limit=data.loc[0, "hedge"])
                elif data.loc[0, "entryqty"] != currentqty:
                    strategy.order('S', 'sell', qty=currentqty, limit=data.loc[0, "take"])

            elif (strategy.switcher == 0 and delta1 > 0 and ticker.last > data.loc[0, "hedge"] and position.currentQty != 0 and pnl < 5) or (delta1 > 0 and ticker.last > data.loc[0, "hedge"] and position.currentQty != 0 and len(strategy.open_orders()) != 2 and pnl < 5):
                strategy.cancel_order_all()
                strategy.switcher = 1

                strategy.order('S', 'sell', qty=currentqty, stop=data.loc[0, "stop"])
                strategy.order('S', 'sell', qty=currentqty, limit=data.loc[0, "take"])

            elif (delta1 > 0 and pnl > 5 and ticker.last > data.loc[0, "hedge"] and position.currentQty != 0 and strategy.switcher == 1) or (delta1 > 0 and ticker.last > data.loc[0, "hedge"] and position.currentQty != 0 and len(strategy.open_orders()) != 2 and pnl > 5):
                strategy.cancel_order_all()
                strategy.switcher = 2

                strategy.order('S', 'sell', qty=currentqty, trailing_offset=onepercent * -9)
                strategy.order('S', 'sell', qty=currentqty, limit=data.loc[0, "take"])

        elif position.currentQty < 0:  # как выше только при шорте
            onepercent = 1

            if pnl < 5:
                stopladder("short", ohlcv, ticker.last, currentqty, strategy)  # стоп лесенкой

            if pnl != 0:
                onepercent = round(module / (pnl * 100))

            currentqty = math.fabs(currentqty)

            # ======================================================================================================================
            if delta1 > 5:
                strategy.cancel_order_all()
                strategy.order('L', 'buy', qty=currentqty)
                data.loc[0, "stop"] = 0
                data.loc[0, "take"] = 0
                data.loc[0, 'entry'] = 0
                data.loc[0, "hedge"] = 0
                data.loc[0, "newHigh"] = 0
                data.loc[0, "newLow"] = 0
                data.loc[0, "lvl"] = 0
                data.loc[0, "limit"] = 0
                data.loc[0, "entryqty"] = 0
                data.loc[0, "qty"] = 0
                data.to_csv(file)

            elif ticker.last > data.loc[0, "hedge"] and delta1 < 0 and position.currentQty != 0 and len(strategy.open_orders()) != 2:
                strategy.cancel_order_all()

                strategy.order('L', 'buy', qty=currentqty, stop=data.loc[0, "stop"])

                if data.loc[0, "entryqty"] == currentqty:
                    strategy.order('L', 'buy', qty=round(currentqty / 2), limit=data.loc[0, "hedge"])
                elif data.loc[0, "entryqty"] != currentqty:
                    strategy.order('L', 'buy', qty=currentqty, limit=data.loc[0, "take"])

            elif (strategy.switcher == 0 and delta1 < 0 and ticker.last < data.loc[0, "hedge"] and position.currentQty != 0 and pnl < 5) or (delta1 < 0 and ticker.last < data.loc[0, "hedge"] and position.currentQty != 0 and len(strategy.open_orders()) != 2 and pnl < 5):
                strategy.cancel_order_all()
                strategy.switcher = 1

                strategy.order('L', 'buy', qty=currentqty, stop=data.loc[0, "stop"])
                strategy.order('L', 'buy', qty=currentqty, limit=data.loc[0, "take"])

            elif (delta1 < 0 and pnl > 5 and ticker.last < data.loc[0, "hedge"] and position.currentQty != 0 and strategy.switcher == 1) or (delta1 < 0 and ticker.last < data.loc[0, "hedge"] and position.currentQty != 0 and len(strategy.open_orders()) != 2 and pnl > 5):
                strategy.cancel_order_all()
                strategy.switcher = 2

                strategy.order('L', 'buy', qty=currentqty, trailing_offset=onepercent * 9)
                strategy.order('L', 'buy', qty=currentqty, limit=data.loc[0, "take"])

        print("ROE", pnl)

    print("-------STOPS-------")
    print("if short stop   :", long_entry_price, '\n'"if long stop    :", short_entry_price)

    print("-------ORDERS--------")
    print(strategy.open_orders())
    print("Orders qty:  ", len(strategy.open_orders()))

    print("-------BALANCE-------")
    print("last balance:  ", balance.BTC.total)

    print("-------CURRENT QTY-------")
    print("position current qty:  ", position.currentQty)

    print("-------CURRENT DELTAS-------")
    print('current:  ', delta1)


if __name__ == '__main__':
    import settings
    import logging
    import logging.config

    logging.config.dictConfig(settings.loggingConf('sample.log'))
    logger = logging.getLogger("sample")
    strategy = Strategy(mylogic)
    strategy.settings.timeframe = '1h'
    strategy.settings.interval = 8
    strategy.settings.symbol = 'BTC/USD'
    strategy.settings.use_websocket = True
    strategy.settings.apiKey = settings.apiKey
    strategy.settings.secret = settings.secret
    strategy.testnet.use = False
    strategy.testnet.apiKey = settings.testnet_apiKey
    strategy.testnet.secret = settings.testnet_secret
    strategy.start()
