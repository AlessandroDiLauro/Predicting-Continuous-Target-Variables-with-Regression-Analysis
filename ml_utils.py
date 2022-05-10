import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
# ================================================================================= #
# ================================================================================= #

def plot_scatter_real_pred(y_test, y_pred):
    
    x = np.linspace(0, y_test.max())
    y = x

    plt.title("Target real x target predito")
    
    plt.plot(x, y, color="red", ls=":")

    sns.scatterplot(x=y_test, y=y_pred)

    plt.xlabel("Real")
    plt.ylabel("Predito")

    plt.show()

def calc_r2_adj(r2, y, X):
    
    return 1 - (1-r2)*(len(y) - 1)/(len(y) - X.shape[1] - 1)


def calc_reg_metrics(model, X, y, label="", plot=False, dist_resids=True, print_stuff=True):
    
    y_pred = model.predict(X)

    if print_stuff:
        print(f"\nMétricas de avaliação (dados de {label}):\n")

    if plot:
        plot_scatter_real_pred(y, y_pred)

    r2 = r2_score(y, y_pred)
#     r2_adj = calc_r2_adj(r2, y, X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred)

    if print_stuff:
#         print(f"R^2: {r2:.2f} | Adj R^2: {r2_adj:.2f}")
        print(f"R^2: {r2:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2%}")

    if dist_resids:
        residuos = y - y_pred
        print(f"\nDistribuição dos resíduos de {label}:\n")
        print(pd.Series(residuos).describe())

    if plot:
        sns.histplot(residuos, kde=True)
        plt.show()
        
    # retorna um dicionário com as métricas
    metrics_dict = {"r2" : r2, 
#                     "r2_adj" : r2_adj,
                    "mae" : mae,
                    "rmse" : rmse,
                    "mape" : mape}
    
    return metrics_dict

# ================================================================================= #
# ================================================================================= #

def reg_lin_pt1_pt2(X_train, y_train, X_test, y_test, 
                    plot=True, scale_mms=False, 
                    train_metrics=True, dist_resids=False):

    if scale_mms:

        mms = MinMaxScaler().fit(X_train)

        # passamos assim, pra carregar o nome das features, e ter o .feature_names_in_
        X_train = pd.DataFrame(mms.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(mms.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # ===============================
    # passo 1 - construção do modelo

    reglin = LinearRegression()

    reglin.fit(X_train, y_train)

    # ===============================
    # passo 2 - avaliação do modelo
    
    if train_metrics:
        _ = calc_reg_metrics(reglin, X_train, y_train, label="treino", 
                             plot=plot, dist_resids=dist_resids, print_stuff=True)
        print()
        print("#"*50)
        
    _ = calc_reg_metrics(reglin, X_test, y_test, label="teste", 
                         plot=plot, dist_resids=dist_resids, print_stuff=True)

    # new: returnando o objeto do modelo treinado!
    return reglin

# ================================================================================= #
# ================================================================================= #

def plot_reglin_model(modelo, X_train, y_train, X_test, y_test): 
    
    plt.title("Modelo de regressão linear")

    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)

    x_plot_modelo = np.linspace(X_train.min(), X_train.max(), 100000)

    # f_h = b0 + b1*x + b2*x^2 + ... + bn*x^n
    # aqui começamos com f_h = b0
    y_plot_modelo = modelo.intercept_ 

    # aqui vem o resto, b1*x + b2*x^2 + ... + bn*x^n
    for n, b_n in enumerate(modelo.coef_):

        y_plot_modelo = y_plot_modelo + b_n*(x_plot_modelo**(n+1))

    plt.plot(x_plot_modelo, y_plot_modelo, color="red")
    plt.show()

def reg_lin_poly_features(X_train, y_train, X_test, y_test, 
                          deg=1,
                          plot=True, scale_mms=False, 
                          train_metrics=True, 
                          dist_resids=True,
                          plot_model=False):
    
    # dimensão dos dados de input
    data_dim = X_train.shape[1]
    
    # é importante salvar os dados originais pra plotá-los!
    # isso é importante pra caso haja transformação no espaço de features
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()
    
    if deg > 1:
        
        pf = PolynomialFeatures(degree=deg, include_bias=False).fit(X_train)
        
        X_train = pf.transform(X_train)
        X_test = pf.transform(X_test)
        
        print(f"Modelo com espaço de features transformado!\n")
        print(f"Número de features original: {pf.n_features_in_}")
        print(f"Número de features após o transformer: {pf.n_output_features_}\n")
        print("="*50)
        
    # é importante escalar depois!
    if scale_mms:

        mms = MinMaxScaler().fit(X_train)

        # não to trazendo o nome das colunas, então não vai mais ter o atributo feature_names_in_!
        # caso queira/precise, tem que alterar a função
        X_train = mms.transform(X_train)
        X_test = mms.transform(X_test)

    # ===============================
    # passo 1 - construção do modelo

    reglin = LinearRegression()

    reglin.fit(X_train, y_train)

    # ===============================
    # passo 2 - avaliação do modelo
    
    if train_metrics:
        metrics_train = calc_reg_metrics(reglin, X_train, y_train, label="treino", 
                                         plot=plot, dist_resids=dist_resids, print_stuff=True)
        print()
        print("#"*50)
    else:
        metrics_train = None
        
    metrics_test = calc_reg_metrics(reglin, X_test, y_test, label="teste",
                                     plot=plot, dist_resids=dist_resids, print_stuff=True)
    
    # só é possível quando temos uma única feature no espaço de features original!
    if plot_model and data_dim == 1:
        # note que passamos as features originais!!
        plot_reglin_model(reglin, X_train_orig, y_train, X_test_orig, y_test)

    # new: returnando o objeto do modelo treinado!
    return reglin, metrics_train, metrics_test

# ================================================================================= #
# ================================================================================= #

def reg_lin_poly_features_regularized(X_train, y_train, X_test, y_test, 
                                      deg=1,
                                      type_regularization=None, alpha=1, l1_ratio=0.5,
                                      iter_max=1000,
                                      plot=True, scale_mms=False, 
                                      train_metrics=True, 
                                      dist_resids=True,
                                      plot_model=False):
    '''
    docstring
    - type_regularization: None, "l1", "l2", "en"
    '''
    
    # dimensão dos dados de input
    data_dim = X_train.shape[1]
    
    # é importante salvar os dados originais pra plotá-los!
    # isso é importante pra caso haja transformação no espaço de features
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()
    
    if deg > 1:
        
        pf = PolynomialFeatures(degree=deg, include_bias=False).fit(X_train)
        
        X_train = pf.transform(X_train)
        X_test = pf.transform(X_test)
        
        print(f"Modelo com espaço de features transformado!\n")
        print(f"Número de features original: {pf.n_features_in_}")
        print(f"Número de features após o transformer: {pf.n_output_features_}\n")
        print("="*50)
        
    # é importante escalar depois!
    # note que o "or type_regularization" garante que haja normalização 
    # se a regulatrização for usada, mesmo que "scale_mms" seja False ;)
    if scale_mms or type_regularization:

        mms = MinMaxScaler().fit(X_train)

        # não to trazendo o nome das colunas, então não vai mais ter o atributo feature_names_in_!
        # caso queira/precise, tem que alterar a função
        X_train = mms.transform(X_train)
        X_test = mms.transform(X_test)

    # ===============================
    # passo 1 - construção do modelo
    
    if type_regularization == "l1":
        
        model = Lasso(alpha=alpha, max_iter=iter_max).fit(X_train, y_train)
        
    elif type_regularization == "l2":
        
        model = Ridge(alpha=alpha, max_iter=iter_max).fit(X_train, y_train)
        
    elif type_regularization == "en":
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=iter_max).fit(X_train, y_train)
        
    elif type_regularization == None:
    
        model = LinearRegression().fit(X_train, y_train)
        
    else:
        
        list_opcoes = ["l1", "l2", "en", None]
        raise ValueError(f"Opção de regularização indisponível!\nOpções aceitas: {list_opcoes}")


    # ===============================
    # passo 2 - avaliação do modelo
    
    if train_metrics:
        metrics_train = calc_reg_metrics(model, X_train, y_train, label="treino", 
                                         plot=plot, dist_resids=dist_resids, print_stuff=True)
        print()
        print("#"*50)
    else:
        metrics_train = None
        
    metrics_test = calc_reg_metrics(model, X_test, y_test, label="teste",
                                     plot=plot, dist_resids=dist_resids, print_stuff=True)
    
    # só é possível quando temos uma única feature no espaço de features original!
    if plot_model and data_dim == 1:
        # note que passamos as features originais!!
        plot_reglin_model(model, X_train_orig, y_train, X_test_orig, y_test)

    # new: returnando o objeto do modelo treinado!
    return model, metrics_train, metrics_test

# ================================================================================= #
# ================================================================================= #




# ================================================================================= #
# ================================================================================= #



# ================================================================================= #
# ================================================================================= #



# ================================================================================= #
# ================================================================================= #