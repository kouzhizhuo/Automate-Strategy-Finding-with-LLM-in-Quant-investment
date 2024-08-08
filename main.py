from rqfactor import *
from rqfactor.extension import *
import rqdatac
import pandas as pd
import modules.alpha_formula_reader as afr
from tqdm import tqdm

rqdatac.init()


def result_to_excel(result, path="result.xlsx"):
    writer = pd.ExcelWriter(path)
    result["rank_ic_analysis"].summary().to_excel(writer, "ic-summary")
    result["quantile"].quantile_returns.to_excel(writer, "各期分组累计收益率序列")
    result["quantile"].quantile_turnover.to_excel(writer, "各期分组换手率")
    result["return"].factor_returns.to_excel(writer, "因子累计收益率")
    result["return"].max_drawdown().to_excel(writer, "最大回撤值")
    result["return"].std().to_excel(writer, "因子收益率波动率")
    writer.close()


CLOSE = Factor("close")
LOW = Factor("low")
VOLUME = Factor("volume")
HIGH = Factor("high")
RETURNS = (CLOSE - REF(CLOSE, 1)) / REF(CLOSE, 1)
MARKET_CAP = Factor("market_cap")
CAP = Factor("market_cap")
VWAP = Factor("total_turnover") / Factor("volume")
RSI = Factor("RSI10")
BOLL_UP = Factor("BOLL_UP")
BOLL_DOWN = Factor("BOLL_DOWN")
EPS = Factor("basic_eps")
EBITDA = Factor("ebitda_ttm")

PriceMomentum = CLOSE - DELAY(CLOSE, 14)
VolumeMomentum = VOLUME - DELAY(VOLUME, 14)
RSIMomentum = RSI - DELAY(RSI, 14)

MeanReversion = MA(CLOSE, 20) - CLOSE
ZScoreMeanReversion = (CLOSE - MA(CLOSE, 20)) / STD(CLOSE, 20)
BollingerBands = Factor("BOLL")

StandardDeviation = STD(CLOSE, 20)
ATR = Factor("ATR")
BollingerBandWidth = (BOLL_UP - BOLL_DOWN) / SMA(CLOSE, 20)

PE = CLOSE / EPS
PB = Factor("pb_ratio_ttm")

TradingVolume = VOLUME
AverageTradingVolume = MA(VOLUME, 20)

GrossProfitMargin = Factor("gross_profit_margin_ttm")
OperatingProfitMargin = Factor("profit_from_operation_to_revenue_ttm")
EarningsGrowthRate = Factor("peg_ratio_ttm")
EBITDAGrowthRate = EBITDA / DELAY(EBITDA, 1) - 1

MovingAverage = SMA(CLOSE, 20)
ExponentialMovingAverage = EMA(CLOSE, 20)

MonthlyReturn = (CLOSE - DELAY(CLOSE, 21)) / DELAY(CLOSE, 21)
QuarterlyReturn = (CLOSE - DELAY(CLOSE, 63)) / DELAY(CLOSE, 63)

formulas = afr.read_alpha_formula_from_excel("./data/Seed Alpha.xlsx")
for i, formula in enumerate(tqdm(formulas)):
    # if i != 22:
    #     continue
    path = f"./result/alpha_analysis/{i + 1}.xlsx"
    f = eval(formula)

    d1 = "20220930"
    d2 = "20221231"

    df = execute_factor(f, rqdatac.index_components("000016.XSHG", d1), d1, d2)
    # 000300.XSHG

    # 实例化引擎
    engine = FactorAnalysisEngine()
    # 构建管道，对因子进行预处理
    engine.append(
        (
            "neutralization",
            Neutralization(
                industry="citics_2019",
                style_factors=[
                    "size",
                    "beta",
                    "earnings_yield",
                    "growth",
                    "liquidity",
                    "leverage",
                    "book_to_price",
                    "residual_volatility",
                    "non_linear_size",
                ],
            ),
        )
    )
    # 构建管道，添加因子分析器
    engine.append(
        (
            "rank_ic_analysis",
            ICAnalysis(rank_ic=True, industry_classification="sws", max_decay=20),
        )
    )
    engine.append(("quantile", QuantileReturnAnalysis(quantile=5, benchmark=None)))
    engine.append(("return", FactorReturnAnalysis()))
    # 调仓周期为1，3，5
    result = engine.analysis(
        df, "daily", ascending=True, periods=[1, 3, 5], keep_preprocess_result=True
    )

    result_to_excel(result, path)
