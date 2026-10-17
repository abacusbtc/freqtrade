"""
TrendFollowingStrategy - long/short trend following, intended for dry-run learning.

Idea (the same logic a futures trend follower such as ES1! would use):
  * Trend filter: fast EMA above slow EMA and price above the slow EMA.
  * Strength filter: ADX above a threshold, so we skip choppy ranges.
  * Entry: breakout, close above the highest high of the last N candles.
  * Exit: ATR-based trailing stop (let winners run, cut losers), plus a
    trend-break exit when the fast EMA crosses below the slow EMA.

Built for a perpetual future (BTC/USDT:USDT on Bybit): trades both directions,
shorts mirror longs (fast EMA below slow EMA, breakdown below the lowest low).
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy, stoploss_from_absolute


class TrendFollowingStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    can_short = True

    # Trailing stop does the real exit work; ROI is effectively disabled.
    minimal_roi = {"0": 100}
    stoploss = -0.15
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count = 220

    ema_fast_len = IntParameter(20, 100, default=50, space="buy", optimize=True)
    ema_slow_len = IntParameter(100, 250, default=200, space="buy", optimize=True)
    adx_min = IntParameter(15, 35, default=22, space="buy", optimize=True)
    breakout_len = IntParameter(10, 60, default=20, space="buy", optimize=True)

    atr_len = 14
    atr_mult = 3.0  # trailing stop distance in ATRs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for n in range(20, 101, 10):
            dataframe[f"ema_fast_{n}"] = ta.EMA(dataframe, timeperiod=n)
        for n in range(100, 251, 10):
            dataframe[f"ema_slow_{n}"] = ta.EMA(dataframe, timeperiod=n)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_len)
        for n in range(10, 61, 5):
            dataframe[f"hh_{n}"] = dataframe["high"].rolling(n).max().shift(1)
            dataframe[f"ll_{n}"] = dataframe["low"].rolling(n).min().shift(1)
        return dataframe

    @staticmethod
    def _snap(value: int, lo: int, step: int) -> int:
        return lo + round((value - lo) / step) * step

    def _cols(self):
        fast = self._snap(self.ema_fast_len.value, 20, 10)
        slow = self._snap(self.ema_slow_len.value, 100, 10)
        hh = self._snap(self.breakout_len.value, 10, 5)
        return f"ema_fast_{fast}", f"ema_slow_{slow}", f"hh_{hh}", f"ll_{hh}"

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast, slow, hh, ll = self._cols()
        dataframe.loc[
            (dataframe[fast] > dataframe[slow])
            & (dataframe["close"] > dataframe[slow])
            & (dataframe["adx"] > self.adx_min.value)
            & (dataframe["close"] > dataframe[hh])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe[fast] < dataframe[slow])
            & (dataframe["close"] < dataframe[slow])
            & (dataframe["adx"] > self.adx_min.value)
            & (dataframe["close"] < dataframe[ll])
            & (dataframe["volume"] > 0),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast, slow, _, _ = self._cols()
        dataframe.loc[
            (dataframe[fast] < dataframe[slow]) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        dataframe.loc[
            (dataframe[fast] > dataframe[slow]) & (dataframe["volume"] > 0),
            "exit_short",
        ] = 1
        return dataframe

    # ATR trailing stop: stop trails at atr_mult * ATR below the best close since entry.
    use_custom_stoploss = True

    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df.empty:
            return None
        atr = df.iloc[-1]["atr"]
        if not atr or atr != atr:
            return None
        if trade.is_short:
            stop_price = min(current_rate, trade.min_rate or current_rate) + self.atr_mult * atr
        else:
            stop_price = max(current_rate, trade.max_rate or current_rate) - self.atr_mult * atr
        # Convert the absolute price to a leverage-aware stoploss ratio.
        return stoploss_from_absolute(
            stop_price, current_rate, is_short=trade.is_short, leverage=trade.leverage
        )

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        return min(2.0, max_leverage)
