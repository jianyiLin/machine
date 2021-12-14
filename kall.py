# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import skew
from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# from xgboost import XGBRegressor
# 导入数据
train = pd.read_csv(r'C:\Users\林剑艺\Desktop\Kaggle房价预测\data\train.csv')
test = pd.read_csv(r'C:\Users\林剑艺\Desktop\Kaggle房价预测\data\test.csv')

# Exploratory Visualization
# 去掉离群点,inplace代表直接在原来的数据修改,不创建新数据
train.drop(train[(train["GrLivArea"] > 4000) & (
    train["SalePrice"] < 300000)].index, inplace=True)
full = pd.concat([train, test], ignore_index=True)
full.drop(['Id'], axis=1, inplace=True)

# Data Cleaning
full["LotAreaCut"] = pd.qcut(full.LotArea, 10)

full['LotFrontage'] = full.groupby(['LotAreaCut'])[
    'LotFrontage'].transform(lambda x: x.fillna(x.median()))

cols = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF",
        "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)

cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
         "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None", inplace=True)

# fill in with mode
cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional",
         "Electrical", "KitchenQual", "SaleType", "Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)

full['LotFrontage'] = full.groupby(['LotAreaCut', 'Neighborhood'])[
    'LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Feature Engineering
full.groupby(['MSSubClass'])[['SalePrice']].agg(['mean', 'median', 'count'])
# Convert some numerical features into categorical features. It's better to use LabelEncoder and get_dummies for these features.
NumStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr",
          "KitchenAbvGr", "MoSold", "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
for col in NumStr:
    full[col] = full[col].astype(str)


def map_values():
    full["oMSSubClass"] = full.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    full["oMSZoning"] = full.MSZoning.map(
        {'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    full["oNeighborhood"] = full.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    full["oCondition1"] = full.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    full["oBldgType"] = full.BldgType.map(
        {'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    full["oHouseStyle"] = full.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    full["oExterior1st"] = full.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    full["oMasVnrType"] = full.MasVnrType.map(
        {'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    full["oExterQual"] = full.ExterQual.map(
        {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFoundation"] = full.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    full["oBsmtQual"] = full.BsmtQual.map(
        {'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oBsmtExposure"] = full.BsmtExposure.map(
        {'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    full["oHeating"] = full.Heating.map(
        {'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    full["oHeatingQC"] = full.HeatingQC.map(
        {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oKitchenQual"] = full.KitchenQual.map(
        {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    full["oFunctional"] = full.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    full["oFireplaceQu"] = full.FireplaceQu.map(
        {'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    full["oGarageType"] = full.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    full["oGarageFinish"] = full.GarageFinish.map(
        {'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    full["oPavedDrive"] = full.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    full["oSaleType"] = full.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    full["oSaleCondition"] = full.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    return "Done!"


map_values()
full.drop(['LotAreaCut'], axis=1, inplace=True)
full.drop(['SalePrice'], axis=1, inplace=True)

# Pipline
# Label Encoding three 'Year' features


class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lab = LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        return X

# Apply log1p to the skewed features, then get_dummies


class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self, skew):
        self.skew = skew

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_numeric = X.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= self.skew].index
        X[skewness_features] = np.log1p(X[skewness_features])
        x = pd.get_dummies(X)
        return x


# build pipeline
pipe = Pipeline([
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=1))
])

full2 = full.copy()
data_pipe = pipe.fit_transform(full2)
print(data_pipe.shape)
print(data_pipe.head())

# use robustscaler since maybe there are other outliers
scaler = RobustScaler()
n_train = train.shape[0]

X = data_pipe[:n_train]
test_X = data_pipe[n_train:]
y = train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)

# Feature Selection,简化数据和模型


class add_feature(BaseEstimator, TransformerMixin):
    def __init__(self, additional=1):
        self.additional = additional

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.additional == 1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + \
                X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
            X["TotalArea"] = X["TotalBsmtSF"] + \
                X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]

            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]

            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]

            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + \
                X["3SsnPorch"] + X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"] + X[
                "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
            return X


pipe = Pipeline([
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=1))
])

# PCA，数据降维
full_pipe = pipe.fit_transform(full)
print(full_pipe.shape)

n_train = train.shape[0]
X = full_pipe[:n_train]
test_X = full_pipe[n_train:]
y = train.SalePrice

X_scaled = scaler.fit(X).transform(X)
y_log = np.log(train.SalePrice)
test_X_scaled = scaler.transform(test_X)

pca = PCA(n_components=410)
X_scaled = pca.fit_transform(X_scaled)
test_X_scaled = pca.transform(test_X_scaled)
print(X_scaled.shape)
print(test_X_scaled.shape)

# 建立模型和评估
# k折交叉验证(k-fold)


def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                   scoring="neg_mean_squared_error", cv=5))
    return rmse

#　集成方法


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        for data in range(pred.shape[1]):
            single = [pred[model, data]*weight for model,
                      weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w


lasso = Lasso(alpha=0.0005, max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma=0.0004, kernel='rbf', C=13, epsilon=0.009)
ker = KernelRidge(alpha=0.2, kernel='polynomial', degree=3, coef0=0.8)
ela = ElasticNet(alpha=0.005, l1_ratio=0.08, max_iter=10000)
bay = BayesianRidge()
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2
weight_avg = AverageWeight(mod=[lasso, ridge, svr, ker, ela, bay], weight=[
                           w1, w2, w3, w4, w5, w6])
print(rmse_cv(weight_avg, X_scaled, y_log),
      rmse_cv(weight_avg, X_scaled, y_log).mean())
# print(score.mean())

# Stacking
# class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __int__(self, mod, meta_model):
#         self.mod = mod
#         self.meta_model = meta_model
#         self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
#     def fit(self,X,y):
#         self.saved_model = [list() for i in self.mod]
#         off_train = np.zeros((X.shape[0], len(self.mod)))
#         for i, model in enumerate(self.mod):
#             for train_index, val_index in self.kf.split(X,y):
#                 renew_model = clone(model)
#                 renew_model.fit(X[train_index], y[train_index])
#                 self.saved_model[i].append(renew_model)
#                 oof_train[val_index, i] = renew_model.predict(X[val_index])
#         self.meta_model.fit(off_train, y)
#         return self
#     def predict(self, X):
#         whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) for single_model in self.saved_model])
#         return self.meta_model.predict(whole_test)


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, mod, meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def fit(self, X, y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))

        for i, model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X, y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index, i] = renew_model.predict(X[val_index])

        self.meta_model.fit(oof_train, y)
        return self

    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)


a = SimpleImputer().fit_transform(X_scaled)
b = SimpleImputer().fit_transform(y_log.values.reshape(-1, 1)).ravel()

stack_model = stacking(mod=[lasso, ridge, svr, ker, ela, bay], meta_model=ker)
stack_model.fit(a, b)

pred = np.exp(stack_model.predict(test_X_scaled))

result = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})
result.to_csv("Submission.csv", index=False)
