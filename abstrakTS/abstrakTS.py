# Bismillahirrahmanirrahim

"""
Title                   : 'abstrakTS'
Type                    : Python Module
Developer               : Rizal Purnawan
Date of first creation  : 2024-01-09
Update                  : 2024-01-16

Description
-----------
This module contains a Python class to systematically analyse a time
series dataset and to build a ML-based forecasting model. The
framework of this module can be thoroughly observed in the technical
documentation.

This module is part of project Abstrak.
"""

# REQUIRED LIBRARIES
# ------------------
from random import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objects

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

# CLASS AbstarkTimeSeries
# ----------------
class AbstrakTS:
    # ---------------------------------------------------------------
    # INITIALIZATIONS
    def __init__(self):
        self.__Sequential = keras.Sequential
        self.__layers = keras.layers
        self.__Dense = keras.layers.Dense
        self.__LSTM = keras.layers.LSTM
        self.__BatchNormalization = keras.layers.BatchNormalization
        self.__Dropout = keras.layers.Dropout
        self.__EarlyStopping = keras.callbacks.EarlyStopping
        self.__go = graph_objects


    # ---------------------------------------------------------------
    # SIMPLE PLOT
    def simple_plot(
            self, variables, cmap= plt.cm.winter,
            figsize= None, title= None, xlabel= None, ylabel= None,
            style= 'classic', labels= None
            ):
        """
        Title       : 'simple_plot'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Descritption
        ------------
        This function helps visualise a time series dataset.
        """
        with plt.style.context(style):
            # Setting colormap:
            dlt = 1 / (len(variables) + 1)
            cmap_frac = [dlt]
            while True:
                if cmap_frac[-1] >= 1:
                    break
                cmap_frac.append(cmap_frac[-1] + dlt)

            # Setting default labels:
            if labels is None:
                labels = [None] *len(variables)
            fig = plt.figure(figsize= figsize)
            for P, r, l in zip(variables, cmap_frac, labels):
                fig = plt.plot(P[0], P[1], color= cmap(r), label= l)
            if all(l is not None for l in labels):
                fig = plt.legend()
            fig = plt.title(title)
            fig = plt.xlabel(xlabel)
            fig = plt.ylabel(ylabel)
            fig = plt.show()


    # ---------------------------------------------------------------
    # MULTIPLOT
    def multiplot(
            self, list_of_df, x= None, exclude_columns= None,
            cmap= "cool", top_bottom= True, title= "Multiplot",
            title_y= 0.9, figsize= None, wspace= 0.4, hspace= 0.4
            ):
        if top_bottom == True:
            nrows, ncols = len(list_of_df), 1
        else:
            nrows, ncols = 1, len(list_of_df)

        if exclude_columns is not None:
            select_columns = [
                [
                    col for col in df.columns
                    if col not in exclude_columns
                    ]
                for df in list_of_df
            ]
            dfs = [
                df.loc[:, cols].copy()
                for df, cols in zip(
                    list(list_of_df), list(select_columns)
                    )
                ]
        else:
            dfs = [df.copy() for df in list_of_df]

        fig, axes = plt.subplots(
            nrows= nrows, ncols= ncols, figsize= figsize
            )

        for k in range(len(dfs)):
            if x is not None:
                fig = dfs[k].set_index(x).plot(
                    cmap= cmap, ax= axes[k]
                    )
            else:
                fig = dfs[k].plot(cmap= cmap, ax= axes[k])
        fig = plt.subplots_adjust(wspace= wspace, hspace= hspace)
        fig = plt.suptitle(title, y= title_y)
        fig = plt.show()


    # ---------------------------------------------------------------
    # SUBPLOTS
    def subplots(
            self,
            X_list, Y_List,
            nrows, ncols,
            title= "Subplots", y= 0.91,
            labels= None, figsize= None, colors= None, cmap= None
            ):
        """
        Title       : 'subplots'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Descritption
        ------------
        This function uses matplotlib to simultaneously plot multiple
        graphs on a single figure.
        """
        fig, axes = plt.subplot(
            nrows= nrows, ncols= ncols, figsize= figsize
            )
        if colors is None:
            if cmap is None:
                colors = [
                    plt.cm.cool(k / range(len(X_list)) + 1)
                    for k in range(len(X_list))
                    ]
            else:
                colors = [
                    cmap(k / range(len(X_list)) + 1)
                    for k in range(len(X_list))
                    ]
        if labels is None:
            labels = [[None, None]] *len(Y_List)
        for k in range(len(X_list)):
            for y, l in zip(list(Y_List[k]), list(labels[k])):
                axes[k].plot(
                    X_list[k], y,
                    color= colors[k],
                    label= l
                )
                if l is not None:
                    axes[k].legend()
        fig = plt.suptitle(title, y= y)
        fig = plt.show()


    # ---------------------------------------------------------------
    # CANDLESTICK (ONLY FOR STOCK)
    def candlestick(
            self, x, open, high, low, close,
            inc_color= "cyan", dec_color= "purple",
            width= 1000, height= 600,
            title= "Candlestick Plot",
            title_x= 0.5,
            font_family= 'Arial'
            ):
        """
        Title       : 'canldestick'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function plots a stock price time series into a
        candlestick plot.

        Note: Only applies to dataset such as stock prices.
        """
        fig = self.__go.Figure(
            data= [
                self.__go.Candlestick(
                    x= x,
                    open= open, high=high, low= low, close= close,
                    increasing_line_color= inc_color,
                    decreasing_line_color= dec_color
                )
            ]
        )
        fig.update_layout(
            width= width, height= height,
            title= title, title_x= title_x,
            font_family= font_family
        )
        fig.show()


    # ---------------------------------------------------------------
    # MOVING AVERAGE:
    def moving_average(self, X, w):
        """
        Title       : 'moving_average'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function is used to compute the moving average of a
        random variable.
        """
        X = list(X)
        ma = list()
        for k in range(len(X)):
            if k < w:
                ma.append(np.mean(X[0: k + w]))
            elif len(X) - k < w:
                ma.append(np.mean(X[k - w:]))
            else:
                ma.append(np.mean(X[k - w: k + w]))
        return ma


    # ---------------------------------------------------------------
    # R-NORM:
    def r_norm1(self, X, X_pred):
        """
        Title       : 'r_norm1'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes r-norm on L1 space (Purnawan, 2023)
        of random variables.

        Reference:
        Purnawan, Rizal. (2023). An Exploration on a Normed Space
            Called r-Normed Space: Some Properties and an
            Application. MDPI preprints.org
        """
        EX = [np.mean(list(X))] *len(X)
        return (
            sum([abs(a - b) for a, b in zip(list(X), list(X_pred))])
            / sum([abs(a - b) for a, b in zip(list(X), EX)])
        )
    

    # ---------------------------------------------------------------
    # SQUARED R-NORM:
    def r_norm2(self, X, X_pred):
        """
        Title       : 'r_norm2'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the squared value of r-norm on L2
        space (Purnawan, 2023) of random variables.

        Reference:
        Purnawan, Rizal. (2023). An Exploration on a Normed Space
            Called r-Normed Space: Some Properties and an
            Application. MDPI preprints.org
        """
        EX = [np.mean(list(X))] *len(X)
        sq_rn = (
            sum([(a - b)**2 for a, b in zip(list(X), list(X_pred))])
            /
            sum([(a - b)**2 for a, b in zip(list(X), EX)])
        )
        return sq_rn


    # ---------------------------------------------------------------
    # MEAN ABSOLUTE ERROR:
    def mean_absolute_error(self, X, X_pred):
        """
        Title       : 'mean_absolute_error'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the mean absolute error of two random
        variables.
        """
        return np.mean(
            [abs(a - b) for a, b in zip(list(X), list(X_pred))]
        )

   
    # ---------------------------------------------------------------
    # LAG OPERATOR
    def kappa(self, j, Y):
        """
        Title       : 'kappa'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates a lag feature given the original
        value of the random variable in 'Y'.

        Requirement:
        'Y' must be an indexed pandas series or data frame.
        """
        index = Y.index
        Y_lagged = [list(Y)[k] for k in range(len(Y) - j)]
        return pd.Series(Y_lagged, index= index[j:])


    # ---------------------------------------------------------------
    # VECTOR LAG OPERATOR
    def kappa_vec(self, J, Y, col_names= None):
        """
        Title       : 'kappa_vec'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates a finite dimensional Euclidean vector
        of lag feature given the original value of the random
        variable in 'Y'.
        
        This is a finite dimensional vector version of function
        'kappa'.

        Requirement:
        'Y' must be an indexed pandas series or data frame.
        """
        k_col = [Y]
        for j in J:
            k_col.append(self.kappa(j, Y))
        df = pd.concat(k_col, axis= 1)
        if col_names is None:
            df.columns = ["Y"] + [f"k_{j}(Y)" for j in J]
        return df


    # ---------------------------------------------------------------
    # HADAMARD PRODUCT
    def hadamard(self, X, Y, euc_vec= False):
        """
        Title       : 'hadamard'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the Hadamard product on a pair of
        m x n matrices X and Y. The function returns an m x n
        matrix.

        Info:
        If 'euc_vec' is set to be True, then 'X' and 'Y' shall be
        an n dimensional Euclidean vector.
        """
        if euc_vec == True:
            return [x *y for x, y in zip(X, Y)]
        else:
            return [
                [x[k] *y[k] for k in range(len(x))]
                for x, y in zip(X, Y)
            ]


    # ---------------------------------------------------------------
    # EXPECTATION
    def expectation(self, X, euc_vec= False):
        """
        Title       : 'expectation'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes an expectation of a random variable.

        Info:
        If 'euc_vec' is set to be True, it returns a constant vector
        of the expectation with the same dimension to that of 'X'.
        """
        EX = np.mean(X)
        if euc_vec == True:
            return [EX] *len(X)
        else:
            return EX


    # ---------------------------------------------------------------
    # COVARIANCE
    def cov(self, X, Y):
        """
        Title       : 'cov'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the covariance of random variables 'X'
        and 'Y'.
        """
        E = self.expectation
        had = self.hadamard
        return E(had(X, Y, euc_vec= True)) - E(X) *E(Y)
    

    # ---------------------------------------------------------------
    # MOVING VARIANCE:
    def moving_variance(self, X, w):
        """
        Title       : 'moving_variance'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the moving variance.
        """
        X = list(X)
        mv = list()
        for k in range(len(X)):
            if k < w:
                mv.append(
                    self.cov(X[0: k + w], X[0: k + w])
                    )
            elif len(X) - k < w:
                mv.append(
                    self.cov(X[k - w:], X[k - w:])
                    )
            else:
                mv.append(
                    self.cov(X[k - w: k + w], X[k - w: k + w])
                    )
        return mv


    # ---------------------------------------------------------------
    # MOVING STANDARD DEVIATION
    def moving_std(self, X, w):
        """
        Title       : 'moving_std'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the moving standard deviation.
        """
        mv = self.moving_variance(X, w)
        return [np.sqrt(x) for x in mv]


    # ---------------------------------------------------------------
    # CORRELATION
    def corr(self, X, Y, method= "Pearson"):
        """
        Title       : 'corr'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the correlation between a pair of
        random variables 'X' and 'Y'. It returns a numeric value in
        between -1 and 1 inclusive.

        There are two methods can be used, namely 'Pearson' or
        'Spearman'. If 'Pearson' is used, the function returns the
        linear correlation between 'X' and 'Y'. If 'Spearman' is
        used, the function returns the rank correlation between 'X'
        and 'Y'.

        Requirement:
        For 'Pearson' method, the random variables cannot be
        almost surely constant.
        """
        cov = self.cov
        try:
            if method == 'Pearson':
                return cov(X, Y) / np.sqrt(cov(X, X) *cov(Y, Y))
            elif method == 'Spearman':
                # Computing the rank of X:
                X_uniq = sorted(list(set(X)))
                rank_X = dict(zip(X_uniq, range(1, len(X_uniq) + 1)))
                rX = [rank_X[x] for x in X]
                # Computing the rank of Y:
                Y_uniq = sorted(list(set(Y)))
                rank_Y = dict(zip(Y_uniq, range(1, len(Y_uniq) + 1)))
                rY = [rank_Y[y] for y in Y]
                # Computing the Spearman correlation:
                return cov(rX, rY) / np.sqrt(cov(rX, rX) *cov(rY, rY))
            else:
                print("ERROR: Unknown method!")
                raise ValueError
        except:
            print("ERROR: Invalid random variables!")
            raise ValueError


    # ---------------------------------------------------------------
    # PARTIAL AUTO CORRELATION
    def pacf(self, Y, start_lag= 1, n_lags= 10, tolerance= 10**(-6)):
        """
        Title       : 'pacf'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function computes the Partial Autocorrelation Function
        (PACF) of a random variable 'Y'.
        """
        corr = self.corr
        J = list(range(start_lag, start_lag + n_lags))
        Y_series = pd.Series(Y)
        Y_lags = self.kappa_vec(J, Y_series)
        # Y_lags = Y_lags.dropna()
        cols = list(Y_lags.columns)
        pacf_list = [1]
        for k in range(2, len(cols) + 1):
            df = Y_lags.loc[:, cols[:k]]
            df = df.dropna()
            y0, y1 = df[cols[0]], df[cols[k - 1]]
            
            if len(df.columns) <= 2:
                if any(len(set(y)) == 1 for y in [y0, y1]):
                    pacf_list.append(1)
                else:
                    pacf_list.append(corr(y1, y0))
            else:
                X = df[cols[1: k - 1]]
                
                y0_model = LinearRegression()
                _ = y0_model.fit(X, y0)
                y0_pred = y0_model.predict(X)

                y1_model = LinearRegression()
                _ = y1_model.fit(X, y1)
                y1_pred = y1_model.predict(X)

                u0 = list(y0 - y0_pred)
                u1 = list(y1 - y1_pred)

                # Removing zero sensitivity:
                for k in range(len(u0)):
                    if u0[k] < tolerance:
                        u0[k] = 0

                    if u1[k] < tolerance:
                        u1[k] = 0
                
                if any(len(set(u)) == 1 for u in [u0, u1]):
                    pacf_list.append(0)
                else:
                    pacf_list.append(
                        corr(u1, u0)
                    )
        return pd.DataFrame({"lags": [0] + J, "pacf": pacf_list})


    # ---------------------------------------------------------------    
    # VISUALIZING PACF
    def pacf_figure(
            self, Y, start_lag= 1, n_lags= 10,
            title= "PACF", color= "mediumaquamarine"):
        """
        Title       : 'pacf_figure'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function plots the result of PACF values on a random
        variable.
        """
        pacf = self.pacf(Y, start_lag= start_lag, n_lags= n_lags)
        fig = pacf.set_index("lags").plot(
            kind= "bar",
            xlabel= "Lag", ylabel= "Correlation",
            ylim= (-1.05, 1.05),
            title= title,
            color= color,
            legend= False
        )
        line_index = [list(pacf["lags"])[0] - 1] \
            + list(pacf["lags"]) \
            + [list(pacf["lags"])[-1] + 1]
        fig = plt.plot(
            line_index, [0] *len(line_index),
            color= "blue"
        )
        fig = plt.show()

    def pacf_multi_fig(
            self, Y_list, start_lag_list, n_lags_list,
            data_id = None, title= "Multiple Plot of PACF",
            title_y= 0.91, cmap= plt.cm.cool, figsize= (20, 5)
            ):
        pacf_list = [
            self.pacf(y, start_lag= s, n_lags= n)
            for y, s, n in zip(Y_list, start_lag_list, n_lags_list)
        ]
        if data_id is None:
            data_id = [f"Y{k}" for k in range(len(Y_list))]
        for p, id in zip(pacf_list, data_id):
            p.columns = ["lags", f"PACF of {id}"]
        fig, axes = plt.subplots(
            nrows= 1, ncols= len(Y_list), figsize= figsize
            )
        for pacf, k in zip(pacf_list, range(len(Y_list))):
            fig = pacf.set_index("lags").plot(
                kind= "bar",
                xlabel= "Lag",
                ylim= (-1.05, 1.05),
                ax= axes[k],
                color= cmap(k / (len(Y_list) + 1)),
                legend= True
                )
            line_index = [list(pacf["lags"])[0] - 1] \
                + list(pacf["lags"]) \
                + [list(pacf["lags"])[-1] + 1]
            axes[k].plot(
                line_index, [0] *len(line_index), color= "blue"
            )
        fig = plt.suptitle(title, y= title_y)
        fig = plt.show()


    # ---------------------------------------------------------------
    # GENERAL ANN MODEL FOR MAP 'phi'
    def phi_model(
            self, J, ts, test_index, with_X= True,
            normalize= True, scaler= StandardScaler, lstm= True,
            lstm_activation= "tanh", first_layer_units= 60,
            num_hidden_layers= 0, hidden_activation= 'relu',
            hidden_units= 240, optimizer= "sgd", batch_size= 60,
            epochs= 100, dropout= 0.3, min_delta= 0.001,
            patience= 50, verbose= 1
            ):
        """
        Title       : 'phi_model'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates an ANN model representing the
        function 'phi. Please read the technical documentation for
        the description of 'phi'.
        """
        # Preparing the training and validation dataset:
        # Making lag features (kappa):
        kappa_vec = self.kappa_vec(J, ts)
        kappa_vec.index = ts.index
        kappa_vec = kappa_vec.dropna()
        if with_X == True:
            kappa_vec['X'] = list(kappa_vec.index)
        train = pd.DataFrame(
            kappa_vec,
            index= [
                i for i in kappa_vec.index
                if i not in test_index
            ]
        )
        valid = pd.DataFrame(kappa_vec, index= test_index)
        scaler = scaler()

        # If normalized:
        if normalize == True:
            X_train = scaler.fit_transform(
                train.drop("Y", axis= 1)
                )
            X_valid = scaler.transform(
                valid.drop("Y", axis= 1)
                )
        else:
            X_train = np.array(train.drop("Y", axis= 1))
            X_valid = np.array(valid.drop("Y", axis= 1))

        # If lstm layer is used:
        if lstm == True:
            X_train = X_train.reshape(
                X_train.shape[0], 1, X_train.shape[1]
            )
            X_valid = X_valid.reshape(
                X_valid.shape[0], 1, X_valid.shape[1]
            )
        else:
            # X_train = pd.DataFrame(
            #     scaler.fit_transform(train.drop("Y", axis= 1)),
            #     columns= train.columns[1:]
            # )
            X_train = pd.DataFrame(
                X_train, columns= train.columns[1:]
                )
            # X_valid = pd.DataFrame(
            #     scaler.transform(valid.drop("Y", axis= 1)),
            #     columns= valid.columns[1:]
            # )
            X_valid = pd.DataFrame(
                X_valid, columns= valid.columns[1:]
            )
        y_train = train["Y"]
        y_valid = valid["Y"]

        # Building the model:
        Sequential = self.__Sequential
        LSTM = self.__LSTM
        BatchNormalization = self.__BatchNormalization
        Dropout = self.__Dropout
        Dense = self.__Dense
        EarlyStopping = self.__EarlyStopping

        model = Sequential()

        # LSTM layer:
        if lstm == True:
            model.add(
                LSTM(
                    first_layer_units,
                    activation= lstm_activation,
                    input_shape= (X_train.shape[1], X_train.shape[2])
                )
            )
        else:
            model.add(
                Dense(
                    first_layer_units,
                    activation= "relu",
                    input_shape= [X_train.shape[1]]
                )
            )
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # Hidden layers:
        for h in range(num_hidden_layers):
            model.add(
                Dense(hidden_units, activation= hidden_activation)
            )
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # Outer layer:
        model.add(Dense(1))

        # Setting earlystopping:
        callback = EarlyStopping(
            min_delta= min_delta,
            patience= patience,
            restore_best_weights= True
        )

        # Compiling model:
        model.compile(optimizer= optimizer, loss= "mae")

        # Training model:
        history = model.fit(
            X_train, y_train,
            validation_data= (X_valid, y_valid),
            batch_size= batch_size, epochs= epochs,
            callbacks= [callback], verbose= verbose
        )
        history_df = pd.DataFrame(history.history)

        # Output:
        return model, scaler, history_df


    # ---------------------------------------------------------------
    # PREDICTION USING 'phi'
    def phi_predict(
            self, J, ts, phi_output, with_X= False,
            forecast= False, normalized= True, lstm= True,
            verbose= True
            ):
        """
        Title       : 'phi_predict'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function execute the prediction command for 'phi_model'.
        If parameter 'forecast' is true, then the prediction will be
        a forecasting.
        """
        if forecast == False:
            X = self.kappa_vec(J, ts)
            X = X.dropna()
            X = X.drop("Y", axis= 1)
        else:
            X = self.lag_fore_horizon(J, ts, with_X= with_X)

        if normalized == True:
            scaler = phi_output[1]
            X = pd.DataFrame(
                scaler.transform(X), columns= X.columns
                )
        if lstm == True:
            X = np.array(X)
            X = X.reshape(X.shape[0], 1, X.shape[1])
        y_pred = np.array(
            [
                y[0] for y in phi_output[0].predict(
                    X, verbose= verbose)
                ]
            )

        return y_pred


    # ---------------------------------------------------------------
    # GENERATING FORECAST HORIZON FOR LAG FEATURES
    def lag_fore_horizon(self, J, ts, with_X= False):
        """
        Title       : 'lag_fore_horizon'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates the lag features for the forecast.
        """
        fore_index = [
            [ts.index[j] + i for i in range(len(ts))]
            for j in J
        ]
        horizon = [
            pd.Series(list(ts), index= fi)
            for fi in fore_index
        ]
        fore_df = pd.concat(horizon, axis= 1)
        fore_df = fore_df.loc[ts.index[-1]:, :].copy()
        fore_df = fore_df.dropna()
        fore_df.columns = [f"k_{j}(Y)" for j in J]
        if with_X == True:
            fore_df["X"] = fore_df.index
        return fore_df
    

    # ---------------------------------------------------------------
    # WAVE FUNCTION ('psi') MODEL
    # WAVE TERMS (Theta) GENERATION
    def Theta(self, ts_index, n_cycle= 20, L= None):
        """
        Title       : 'Theta'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates the sinusoidal terms for the wave
        function.
        """
        if L is None:
            L = len(ts_index)
        terms = dict()
        terms['const'] = [1] *len(ts_index)
        for n in range(1, n_cycle + 1):
            terms[f"sin{n}"] = [
                np.sin(2 *np.pi *(n / L) *x) for x in ts_index
            ]
            terms[f"cos{n}"] = [
                np.cos(2 *np.pi *(n / L) *x) for x in ts_index
            ]
        terms_df = pd.DataFrame(terms)
        terms_df.index = ts_index
        return terms_df

    # WAVE MODEL ('psi')
    def psi_model(
            self, ts, max_cycle, valid_index, monitor= "val_loss"
            ):
        """
        Title       : 'wave_model'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates the wave function model using the
        sinusoidal terms and 'LinearRegression' from
        'sklearn.linear_model'.
        """
        ts_train = pd.Series(
            ts, index= [i for i in ts.index if i not in valid_index]
        )
        ts_valid = pd.Series(ts, index= valid_index)
        L = len(ts_train)
        y_train = np.array(ts_train)
        y_valid = np.array(ts_valid)

        loss = list()
        val_loss = list()
        cycle = list()
        for c in range(1, max_cycle + 1):
            X_train = self.Theta(ts_train.index, n_cycle= c, L= L)
            X_valid = self.Theta(ts_valid.index, n_cycle= c, L= L)

            model = LinearRegression(fit_intercept= False)
            _ = model.fit(X_train, y_train)
            y_pre = model.predict(X_train)
            y_val = model.predict(X_valid)

            mae = np.mean(
                [abs(a - b) for a, b in zip(list(y_pre), list(y_train))]
            )
            val_mae = np.mean(
                [abs(a - b) for a, b in zip(list(y_valid), list(y_val))]
            )
            loss.append(mae)
            val_loss.append(val_mae)
            cycle.append(c)
        loss_cycle = dict(zip(loss, cycle))
        val_loss_cycle = dict(zip(val_loss, cycle))
        if monitor == "loss":
            n_cycle = loss_cycle[min(loss)]
        elif monitor == "val_loss":
            n_cycle = val_loss_cycle[min(val_loss)]
        X_train = self.Theta(
            ts_train.index, n_cycle= n_cycle, L= L
            )
        model = LinearRegression(fit_intercept= False)
        _ = model.fit(X_train, y_train)

        return model, n_cycle, L, loss_cycle
    
    # PREDICTION COMMAND FOR 'psi'
    def psi_predict(self, ts_index, psi_output):
        """
        Title       : 'psi_predict'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function performs prediction for 'psi'.
        """
        X = self.Theta(ts_index, psi_output[1], psi_output[2])
        model = psi_output[0]
        y_pred = model.predict(X)
        return y_pred


    # ---------------------------------------------------------------
    # SEASONALITY COMPONENT
    # MODULO SEASONALITY
    def modulo_season(self, Y_index, p):
        """
        Title       : 'modulo_season'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function is an implementation of the map 'mu_p' (Please
        refer to the technical documentation).
        """
        return [x % p for x in Y_index]
    
    # FINDING SEASONAL FEATURES
    def find_modulo_season(self, Y, max_p, delta= 0.2, num_p= 5):
        """
        Title       : 'find_modulo_season'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function helps constructing the modulo seasonal features
        by finding the satisfiability of the set of all periods.

        Requirement:
        - 'Y' must be a pandas series.
        - 'max_p' is a positive integer smaller than the length of Y.
        """
        X = list(Y.index)
        Y_val = list(Y)

        # Generating maps mu_p
        mu_dict = {
            p: self.modulo_season(Y.index, p)
            for p in range(2, max_p + 1)
        }

        # Finding mu_p with best Pearson correlations:
        corr_list = list()
        for p in range(2, max_p + 1):
            corr_list.append(
                abs(self.corr(Y_val, mu_dict[p], method= "Pearson"))
            )
        corr_p_dict = dict(zip(corr_list, range(2, max_p + 1)))
        best_corr = [c for c in corr_list if c > delta]
        if len(best_corr) > num_p:
            best_corr = sorted(best_corr, reverse= True)[:num_p]
        best_p = [corr_p_dict[c] for c in best_corr]
        best_mu = [mu_dict[p] for p in best_p]

        # Making the modulo seasonal features as a dataframe:
        mod_df = pd.DataFrame(
            {f"mod {p}": mu for p, mu in zip(best_p, best_mu)}
        )
        mod_df.index = X

        # Ouput:
        return mod_df, {"p": best_p, "corr": best_corr}
    
    # GENERATING MODULO SEASONAL FEATURES:
    def modulo_seasonal_features(self, Y_index, M):
        """
        Title       : 'modulo_seasonal_features'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function generates the modulo seasonal features with
        given collection M of possible periods.
        """
        mod_df = pd.DataFrame(
            {
                f"mod {p}": self.modulo_season(Y_index, p)
                for p in M}
        )
        mod_df.index = Y_index
        return mod_df


    # ---------------------------------------------------------------
    # ANN MODEL FOR 'eta'
    def eta_ann_model(
            self, modulo_sf, y, valid_index,
            normalize= False, scaler= StandardScaler,
            first_units= 100, activation= "relu",
            num_hidden_layers= 3, hidden_units= 100,
            dropout= 0.3, optimizer= "sgd", loss= "mae",
            batch_size= 60, epochs= 200,
            min_delta= 0.001, patience= 50, verbose= 0
            ):
        """
        Title       : 'eta_ann_model'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function is an implementation of 'eta' (Please refer to
        the technical documentation) using ANN model.

        Note:
        - 'modulos_sf' is the modulo seasonal features in a pandas
          data frame.
        """
        train_index = [
            i for i in modulo_sf.index if i not in valid_index
            ]
        X_train = pd.DataFrame(modulo_sf, index= train_index)
        X_valid = pd.DataFrame(modulo_sf, index= valid_index)
        y_train = pd.Series(y, index= train_index)
        y_valid = pd.Series(y, index= valid_index)

        if normalize == True:
            scaler = scaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns= X_train.columns
            )
            X_valid = pd.DataFrame(
                scaler.transform(X_valid),
                columns= X_valid.columns
            )

        # ANN model:
        # Importance instantiations:
        Dense = self.__Dense
        BatchNormalization = self.__BatchNormalization
        Dropout = self.__Dropout
        EarlyStopping = self.__EarlyStopping
        Sequential = self.__Sequential
        
        # The model:
        model = Sequential()

        # First layer:
        model.add(
            Dense(
                first_units,
                activation= activation,
                input_shape= [X_train.shape[1]]
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # Hidden layers:
        for k in range(num_hidden_layers):
            model.add(Dense(hidden_units, activation= activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # Output layer:
        model.add(Dense(1))

        # Setting early stopping:
        callback = EarlyStopping(
            min_delta= min_delta,
            monitor= "val_loss",
            patience= patience,
            restore_best_weights= True
        )

        # Compiling the model:
        model.compile(optimizer= optimizer, loss= loss)

        # Training the model:
        history = model.fit(
            X_train, y_train,
            validation_data= (X_valid, y_valid),
            batch_size= batch_size,
            epochs= epochs,
            callbacks= [callback],
            verbose= verbose
        )
        history_df = pd.DataFrame(history.history)

        # Output:
        return model, scaler, history_df


    # ---------------------------------------------------------------
    # PREDICTION COMMAND FOR 'eta_ann_model'
    def eta_ann_predict(
            self, modulo_seasonal_features, eta_ann_output,
            normalize= False, verbose= False
            ):
        """
        Title       : 'eta_ann_predict'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function execute the prediction command for
        'eta_ann_model'.
        """
        if normalize == True:
            scaler = eta_ann_output[1]
            X = pd.DataFrame(
                scaler.transform(modulo_seasonal_features),
                columns= modulo_seasonal_features.columns
            )
        else:
            X = modulo_seasonal_features.copy()
        
        model = eta_ann_output[0]

        y_pred = np.array(
            [
                y[0] for y in model.predict(X, verbose= verbose)
                ]
        )

        return y_pred


    # ---------------------------------------------------------------
    # LINEAR MODEL FOR 'eta'
    def eta_linear(self, modulo_sf, y, valid_index, metric= "mae"):
        """
        Title       : 'eta_linear'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function is an implementation of 'eta' (Please refer to
        the technical documentation) using linear model.

        Note:
        - 'modulos_sf' is the modulo seasonal features in a pandas
          data frame.
        """
        train_index = [
            i for i in modulo_sf.index if i not in valid_index
            ]
        X_train = pd.DataFrame(modulo_sf, index= train_index)
        X_valid = pd.DataFrame(modulo_sf, index= valid_index)
        y_train = pd.Series(y, index= train_index)
        y_valid = pd.Series(y, index= valid_index)

        model = LinearRegression()
        _ = model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        if metric == "mae":
            score = self.mean_absolute_error(y_valid, y_pred)
        elif metric == "r-norm2":
            score = self.r_norm2(y_valid, y_pred)
        else:
            raise ValueError

        return model, score
    
    # ---------------------------------------------------------------
    # WHITE NOISE WEAKNESS
    def white_noise_weakness(self, nu, w):
        """
        Title       : 'white_noise_weakness'
        Type        : Instance Method
        Developer   : Rizal Purnawan

        Description
        -----------
        This function compute the weakness level of white noise based
        on the weak white noise theory presented in the technical
        framework of AbstrakTS. The function returns the triple
        (delta1, delta2, delta3).
        """
        d1 = np.mean(nu)
        d2 = (
            max(self.moving_variance(nu, w))
            - min(self.moving_variance(nu, w))
        )
        corr = list()
        nu_series = pd.Series(nu)
        # We will only sampling the autocorrelation for the first
        # 30% of the lags:
        for j in range(1, int(0.3 *len(nu))):
            lag = self.kappa(j, nu_series)
            nu_lag = pd.concat([nu_series, lag], axis= 1)
            nu_lag.columns = ["nu", "lag"]
            nu_lag = nu_lag.dropna()
            corr.append(self.corr(nu_lag["nu"], nu_lag["lag"]))
        d3 = max(corr)
        return (d1, d2, d3)
