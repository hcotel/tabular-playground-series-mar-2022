import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import HuberRegressor
import optuna
from constants import run_optuna, trial, run_adversarial, validation_files_index, check_val_results, optuna_trials,\
    plot_importance, monday_afternoon_validation, last_monday_validation, regressor_name
import warnings
import seaborn as sns
from math import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean, hmean
warnings.filterwarnings(action="ignore")
sns.color_palette("flare", as_cmap=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")
trend_difference = pd.read_csv("data/trend_difference_df.csv")
trend_difference_hm = pd.read_csv("data/trend_difference_roadway_hm.csv")
sub_72 = pd.read_csv("submission_files/submission_72_int.csv")
sub_72["congestion_72"] = sub_72.congestion
train["is_train"] = 1
test["is_train"] = 0
all = pd.concat([train, test], axis=0)
all["time"] = pd.to_datetime(all.time)
all["direction"] = all.direction.astype("category")
all["is_train"] = all.is_train.astype("category")

sin_vals = {
    'NB': 0.0,
    'NE': sin(1 * pi/4),
    'EB': 1.0,
    'SE': sin(3 * pi/4),
    'SB': 0.0,
    'SW': sin(5 * pi/4),
    'WB': -1.0,
    'NW': sin(7 * pi/4),
}

cos_vals = {
    'NB': 1.0,
    'NE': cos(1 * pi/4),
    'EB': 0.0,
    'SE': cos(3 * pi/4),
    'SB': -1.0,
    'SW': cos(5 * pi/4),
    'WB': 0.0,
    'NW': cos(7 * pi/4),
}

opposite_direction_map = {
    'NB': 'SB',
    'NE': 'SW',
    'EB': 'WB',
    'SE': 'NW',
    'SB': 'NB',
    'SW': 'NE',
    'WB': 'EB',
    'NW': 'SE',
}


all['direction_sin'] = all['direction'].map(sin_vals)
all['direction_cos'] = all['direction'].map(cos_vals)

all["x+y"] = all["x"].astype(str) + all["y"].astype(str)
all["x+y+direction"] = all["x+y"] + all["direction"].astype(str)
all["x+direction"] = all["x"].astype(str) + all["direction"].astype(str)
all["y+direction"] = all["y"].astype(str) + all["direction"].astype(str)
all['hour'] = all['time'].dt.hour
all['minute'] = all['time'].dt.minute
all['day'] = all['time'].dt.dayofyear
all['month'] = all['time'].dt.month
all['day_of_month'] = all['time'].dt.day
all['hour+minute'] = all['time'].dt.hour * 60 + all['time'].dt.minute
all['hour+direction'] = all['hour'].astype('str') + all['direction'].astype('str')
all["weekday"] = all["time"].dt.dayofweek
all["is_weekend"] = all["weekday"] > 4

roadways_list = all["x+y+direction"].unique().tolist()

all = all.merge(trend_difference, left_on="x+y+direction", right_on="roadway", how='left')
all = all.rename(columns={'trend': 'trend_roadway'})
all = all.merge(trend_difference_hm, on=["x+y+direction", 'hour+minute'], how='left')
all = all.rename(columns={'trend': 'trend_roadway_hm'})

medians = pd.DataFrame(all.groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.median()).reset_index()
medians = medians.rename(columns={'congestion': 'congestion_median'})
all = all.merge(medians, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')
all.loc[all.is_train == 1, 'congestion-median'] = all.loc[all.is_train == 1]['congestion'] - all.loc[all.is_train == 1]['congestion_median']

mins = pd.DataFrame(all.groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.min()).reset_index()
mins = mins.rename(columns={'congestion': 'congestion_min'})
all = all.merge(mins, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')

maxs = pd.DataFrame(all.groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.max()).reset_index()
maxs = maxs.rename(columns={'congestion': 'congestion_max'})
all = all.merge(maxs, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')

means = pd.DataFrame(all.groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.mean()).reset_index()
means = means.rename(columns={'congestion': 'congestion_mean'})
all = all.merge(means, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')

h_mean = pd.DataFrame(all[all.is_train == 1].groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.apply(hmean)).reset_index()
h_mean = h_mean.rename(columns={'congestion': 'congestion_hmean'})
all = all.merge(h_mean, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')

g_mean = pd.DataFrame(all[all.is_train == 1].groupby(["x+y+direction", 'weekday', 'hour', 'minute']).congestion.apply(gmean)).reset_index()
g_mean = g_mean.rename(columns={'congestion': 'congestion_gmean'})
all = all.merge(g_mean, on=["x+y+direction", 'weekday', 'hour', 'minute'], how='left')

mornings = all[(all.hour >= 6) & (all.hour < 12)]
morning_avgs = pd.DataFrame(mornings.groupby(["x+y+direction", 'month', 'day_of_month']).congestion.median()).reset_index()
morning_avgs = morning_avgs.rename(columns={'congestion':'congestion_morning_median'})
all = all.merge(morning_avgs, on=["x+y+direction", 'month', 'day_of_month'], how='left')

morning_avgs_allroads = pd.DataFrame(mornings.groupby(['month', 'day_of_month']).congestion.mean()).reset_index()
morning_avgs_allroads = morning_avgs_allroads.rename(columns={'congestion':'congestion_allroads_morning_median'})
all = all.merge(morning_avgs_allroads, on=['month', 'day_of_month'], how='left')

early_mornings = all[(all.hour >= 6) & (all.hour < 9)]
early_morning_avgs = pd.DataFrame(early_mornings.groupby(["x+y+direction", 'month', 'day_of_month']).congestion.median()).reset_index()
early_morning_avgs = early_morning_avgs.rename(columns={'congestion': 'congestion_emorning_median'})
all = all.merge(early_morning_avgs, on=["x+y+direction", 'month', 'day_of_month'], how='left')

all["opposite_roadway"] = all["x"].astype(str) + all["y"].astype(str) + all["direction"].map(opposite_direction_map).astype(str)
all["is_twoway"] = all["opposite_roadway"].isin(roadways_list)
opp_avgs = pd.DataFrame(all.groupby(["opposite_roadway", 'weekday', 'hour', 'minute']).congestion.median()).reset_index()
opp_avgs = opp_avgs.rename(columns={'congestion': 'congestion_opposite_median'})
all = all.merge(opp_avgs, on=["opposite_roadway", 'weekday', 'hour', 'minute'], how='left')

mornings = all[(all.hour >= 6) & (all.hour < 12)]
opp_morning_avgs = pd.DataFrame(mornings.groupby(["x+y+direction", 'month', 'day_of_month']).congestion.median()).reset_index()
opp_morning_avgs = opp_morning_avgs.rename(columns={'congestion': 'congestion_opposite_morning_median'})
all = all.merge(opp_morning_avgs, on=["x+y+direction", 'month', 'day_of_month'], how='left')

all.loc[all.congestion.isnull(), "congestion"] = all.congestion_median


all["congestion_lag1"] = all.groupby(['x+y+direction'])["congestion"].shift(1)
all.loc[all.congestion_lag1.isnull(), "congestion_lag1"] = all.congestion

all["congestion_lead1"] = all.groupby(['x+y+direction'])["congestion"].shift(-1)
all.loc[all.congestion_lead1.isnull(), "congestion_lead1"] = all.congestion

all["congestion_lag2"] = all.groupby(['x+y+direction'])["congestion"].shift(2)
all.loc[all.congestion_lag2.isnull(), "congestion_lag2"] = all.congestion_lag1

all["congestion_lag72"] = all.groupby(['x+y+direction'])["congestion"].shift(72)
all["congestion_lag72_7"] = all.groupby(['x+y+direction'])["congestion"].shift(72*7)

features = [
    "x",
    "y",
    "direction",
    "x+y",
    "x+y+direction",
    'hour+minute',
    #'hour',
    'hour+direction',
    "weekday",
    #"is_weekend",
    "direction_sin",
    "direction_cos",
    #"congestion_mean",
    "congestion_hmean",
    #"congestion_gmean",
    "congestion_median",
    "congestion_morning_median",
    "congestion_emorning_median",
    "congestion_opposite_median",
    "congestion_opposite_morning_median",
    "congestion_allroads_morning_median",
    "congestion_min",
    "congestion_max",
    "trend_roadway",
    "is_twoway",
    "congestion_lag1",
    "congestion_lag2",
    "congestion_lag72",
    "congestion_lag72_7",
    #"congestion_lead1",
    #"trend_roadway_hm",
    #"congestion_std"
]
categorical_features = [
    "x",
    "y",
    "direction",
    "x+y",
    "x+y+direction",
    'hour+minute',
    'hour',
    'hour+direction',
    "weekday",
    "is_twoway",
    "is_weekend"
]
target_feature = 'congestion'

all[categorical_features] = all[categorical_features].astype("category")
le = LabelEncoder()
all[categorical_features] = all[categorical_features].apply(le.fit_transform)

train_idxes = all[all.is_train == 1].index
# train_idxes = all.loc[train_idxes][all.loc[train_idxes].weekday == 0].index

if run_adversarial:
    y = all["is_train"]
    X = all[features]

    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    oof = np.zeros(len(X))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values, y)):
        print("fold n°{}".format(fold_))

        classifier = RFC(n_estimators=1000, n_jobs=-1, verbose=True)
        classifier.fit(all.iloc[trn_idx][features], y.iloc[trn_idx])
        oof[val_idx] = classifier.predict_proba(all.iloc[val_idx][features])[:, 1]
    print(f"ROC Adv score: {roc_auc_score(y, oof)}")

    all['adv_score'] = oof
    val_idxes = oof[train_idxes].argsort()[:20000]
    train_idx_to_drop = oof[train_idxes].argsort()[-20000:]
    train_after_dropped_idxes = train_idxes.difference(val_idxes).difference(train_idx_to_drop)
    with open(f"validation_files/val_idxes_{validation_files_index}.pickle", "wb") as handle:
        pickle.dump(val_idxes, handle)
    with open(f"validation_files/train_idxes_{validation_files_index}.pickle", "wb") as handle:
        pickle.dump(train_after_dropped_idxes, handle)
elif monday_afternoon_validation:
    val_idxes = all.iloc[train_idxes][(all.iloc[train_idxes].weekday == 0) & (all.iloc[train_idxes].hour >= 12)].index
    train_after_dropped_idxes = train_idxes.difference(val_idxes)
elif last_monday_validation:
    tst_start = pd.to_datetime('1991-09-23 12:00')
    tst_finish = pd.to_datetime('1991-09-23 23:40')
    val_idxes = all[(all['time'] >= tst_start) & (all['time'] <= tst_finish)].index
    tst_start = pd.to_datetime('1991-09-16 12:00')
    tst_finish = pd.to_datetime('1991-09-16 23:40')
    val_idxes_2 = all[(all['time'] >= tst_start) & (all['time'] <= tst_finish)].index
    tst_start = pd.to_datetime('1991-09-09 12:00')
    tst_finish = pd.to_datetime('1991-09-09 23:40')
    val_idxes_3 = all[(all['time'] >= tst_start) & (all['time'] <= tst_finish)].index
    val_idxes = val_idxes.union(val_idxes_2).union(val_idxes_3)
    train_after_dropped_idxes = train_idxes.difference(val_idxes)
else:
    with open(f"validation_files/val_idxes_{validation_files_index}.pickle", "rb") as handle:
        val_idxes = pickle.load(handle)
    with open(f"validation_files/train_idxes_{validation_files_index}.pickle", "rb") as handle:
        train_after_dropped_idxes = pickle.load(handle)

train_x = all.iloc[train_after_dropped_idxes][features]
train_y = all.iloc[train_after_dropped_idxes][target_feature]

val_x = all.iloc[val_idxes][features]
val_y = all.iloc[val_idxes][target_feature]

X_test = all[all.is_train == 0][features]

def objective_cat(trial):

    param = {}
    param['learning_rate'] = trial.suggest_discrete_uniform("learning_rate", 0.0005, 0.005, 0.0005)
    param['depth'] = trial.suggest_int('depth', 8, 12)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5)
    param['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 32)
    param['grow_policy'] = 'Depthwise'
    param['task_type'] = "GPU"
    param['iterations'] = 10000
    param['use_best_model'] = True
    param['eval_metric'] = 'MAE'
    param['od_type'] = 'IncToDec'
    param['od_wait'] = 200
    param['random_state'] = 42
    #param['logging_level'] = 'Silent'

    model = CatBoostRegressor(**param)

    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        early_stopping_rounds=50,
        verbose=1000,
    )

    preds = model.predict(val_x)

    metric = mae(val_y, preds)

    return metric

def objective_xgb(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, observation_key='validation_0-mae')
    param = {
        "tree_method": "gpu_hist",  # Use GPU acceleration
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1e2),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1e2),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 2000, 6000, 400),
        "max_depth": trial.suggest_int("max_depth", 8, 12),
        "random_state": 42,
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "objective": 'reg:squarederror'
    }
    model = XGBRegressor(**param)

    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        eval_metric="mae",
        early_stopping_rounds=200,
        callbacks=[pruning_callback],
        verbose=1000,
    )

    preds = model.predict(val_x)

    metric = mae(val_y, preds)

    return metric

def objective_huber(trial):

    param = {
        "epsilon": trial.suggest_float("epsilon", 1, 3),
        "alpha": trial.suggest_loguniform("alpha", 1e-6, 1e-2),
        "max_iter": trial.suggest_int("max_iter", 100, 2000, 100),
    }
    model = HuberRegressor(**param)

    model.fit(
        train_x,
        train_y,
    )

    preds = model.predict(val_x)

    metric = mae(val_y, preds)

    return metric

def objective_hgbt(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 250, 5),
        "max_depth": trial.suggest_int("max_depth", 10, 20, 1),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, 1),
        "loss": 'absolute_error'
    }
    model = HistGradientBoostingRegressor(**param)

    model.fit(
        train_x,
        train_y,
    )

    preds = model.predict(val_x)
    metric = mae(val_y, preds)

    return metric

if run_optuna:
    study = optuna.create_study(direction="minimize")
    if regressor_name == "xgb":
        study.optimize(objective_xgb, n_trials=optuna_trials)
    elif regressor_name == "hgbt":
        study.optimize(objective_hgbt, n_trials=optuna_trials)
    elif regressor_name == "cat":
        study.optimize(objective_cat, n_trials=optuna_trials)
    elif regressor_name == "huber":
        study.optimize(objective_huber, n_trials=optuna_trials)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)

    best_params = study.best_params
else:
    if regressor_name == "xgb":
        best_params = {'reg_lambda': 0.0010830683623988624, 'reg_alpha': 3.3840407485830766, 'colsample_bytree': 1.0, 'subsample': 0.7, 'learning_rate': 0.005096701929063882, 'n_estimators': 2800, 'max_depth': 12, 'min_child_weight': 6.019752110452971}
        best_params["tree_method"] = "gpu_hist"
        best_params["random_state"] = 42
    elif regressor_name == "hgbt":
        best_params = {'learning_rate': 0.04944492611926383, 'max_leaf_nodes': 220, 'max_depth': 17, 'min_samples_leaf': 37}
        best_params["loss"] = 'absolute_error'
    elif regressor_name == "cat":
        best_params = {'learning_rate': 0.0035, 'depth': 11, 'l2_leaf_reg': 2.0, 'min_child_samples': 23}
        best_params['grow_policy'] = 'Depthwise'
        best_params['task_type'] = "GPU"
        best_params['iterations'] = 10000
        best_params['use_best_model'] = True
        best_params['eval_metric'] = 'MAE'
        best_params['od_type'] = 'Iter'
        best_params['od_wait'] = 20
        best_params['random_state'] = 42
    elif regressor_name == "huber":
        best_params = {'epsilon': 1.1007667969037302, 'alpha': 0.00027650912904181584, 'max_iter': 800}
    else:
        best_params = {}



if regressor_name == "xgb":
    model = XGBRegressor(**best_params)
    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        eval_metric="mae",
        early_stopping_rounds=200,
        verbose=1000,
    )
elif regressor_name == "hgbt":
    best_params["loss"] = 'absolute_error'
    model = HistGradientBoostingRegressor(**best_params)
    model.fit(
        train_x,
        train_y,
    )
elif regressor_name == "cat":
    best_params['iterations'] = 10000
    best_params['grow_policy'] = 'Depthwise'
    best_params['task_type'] = "GPU"
    best_params['iterations'] = 10000
    best_params['use_best_model'] = True
    best_params['eval_metric'] = 'MAE'
    best_params['od_type'] = 'Iter'
    best_params['od_wait'] = 20
    print(best_params)
    model = CatBoostRegressor(**best_params)
    model.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        verbose=1000,
    )
elif regressor_name == "huber":
    model = HuberRegressor(**best_params)
    model.fit(
        train_x,
        train_y,
    )

if plot_importance and not regressor_name == "hgbt" and not regressor_name == "huber":
    importances = model.feature_importances_
    indices = np.argsort(importances)
    # indices = indices[-10:]

    plt.figure(figsize=(20, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

if check_val_results:
    preds = model.predict(val_x)
    metric = mae(val_y, preds)
    print(f"Validation score: {metric}")

#model.save_model(f"models/{trial}.model")
#model.load_model(f"models/{trial}.model")  # load data
preds = model.predict(X_test)
sample_submission = sample_submission.set_index('row_id', drop=False)
sample_submission["congestion"] = preds
print(f"Preds mean: {preds.mean()}")
if target_feature == "congestion-median":
    sample_submission["congestion"] = sample_submission["congestion"] + all[all.is_train == 0].congestion_median
special = pd.read_csv('data/special v2.csv', index_col="row_id")
special = special[['congestion']].rename(columns={'congestion':'special'})
sample_submission = sample_submission.merge(special, left_index=True, right_index=True, how='left')
sample_submission['special'] = sample_submission['special'].fillna(sample_submission['congestion'])
sample_submission = sample_submission.drop(['congestion'], axis=1).rename(columns={'special': 'congestion'})
sample_submission[["row_id", "congestion"]].to_csv(
    f"submission_files/submission_{trial}.csv", sep=",", index=False
)
sample_submission["congestion"] = sample_submission["congestion"].astype(int)
sample_submission[["row_id", "congestion"]].to_csv(
    f"submission_files/submission_{trial}_int.csv", sep=",", index=False
)