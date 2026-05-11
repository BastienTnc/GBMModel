# Lib Imports

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model

# ticker LVMH
ticker = "MC.PA"

# données
df = yf.download(ticker, start="2020-01-01")

#rendements log
df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

# Modèle GARCH(1,1)
model = arch_model(df['Log_Returns'].dropna(), vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')

print(model_fit.summary())
