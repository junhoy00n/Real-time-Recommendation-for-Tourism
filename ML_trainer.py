import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

root_dir = 'D:/EVGPS/v5/'

EVGPS_DataLab_visit_place_df = pd.read_csv(root_dir + 'EVGPS&DataLab_Tour_Info&Tourist_Profile.txt', encoding='cp949')

# Top 5 near-by relation tourist attraction label concat
EVGPS_DataLab_visit_place_df['LABEL_CONCAT'] = EVGPS_DataLab_visit_place_df['Top1_Relation_Tourist_Attraction'].map(str) + ', ' + \
                                               EVGPS_DataLab_visit_place_df['Top2_Relation_Tourist_Attraction'].map(str) + ', ' + \
                                               EVGPS_DataLab_visit_place_df['Top3_Relation_Tourist_Attraction'].map(str) + ', ' + \
                                               EVGPS_DataLab_visit_place_df['Top4_Relation_Tourist_Attraction'].map(str) + ', ' + \
                                               EVGPS_DataLab_visit_place_df['Top5_Relation_Tourist_Attraction'].map(str)

# change value by label for input ML model
season_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['SEASON'])
EVGPS_DataLab_visit_place_df['SEASON'] = season_enc.transform(EVGPS_DataLab_visit_place_df['SEASON'])

sex_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['SEX'])
EVGPS_DataLab_visit_place_df['SEX'] = sex_enc.transform(EVGPS_DataLab_visit_place_df['SEX'])

companion_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['COMPANION'])
EVGPS_DataLab_visit_place_df['COMPANION'] = companion_enc.transform(EVGPS_DataLab_visit_place_df['COMPANION'])

category_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['TOUR_CATEGORY'])
EVGPS_DataLab_visit_place_df['TOUR_CATEGORY'] = category_enc.transform(EVGPS_DataLab_visit_place_df['TOUR_CATEGORY'])

place_title_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['Place_Title'])
EVGPS_DataLab_visit_place_df['Place_Title'] = place_title_enc.transform(EVGPS_DataLab_visit_place_df['Place_Title'])

target_enc = LabelEncoder().fit(EVGPS_DataLab_visit_place_df['LABEL_CONCAT'])
EVGPS_DataLab_visit_place_df['LABEL'] = target_enc.transform(EVGPS_DataLab_visit_place_df['LABEL_CONCAT'])

# split train, test for predict future
test_date_list = ['2021-10', '2021-11', '2021-12',
                  '2022-01', '2022-02', '2022-03',
                  '2022-04', '2022-05', '2022-06',
                  '2022-07', '2022-08', '2022-09']
test_date_list = '|'.join(test_date_list)

train_df_idx = EVGPS_DataLab_visit_place_df[~EVGPS_DataLab_visit_place_df['DATE'].str.contains(test_date_list)].index.values.tolist()
test_df_idx = EVGPS_DataLab_visit_place_df[EVGPS_DataLab_visit_place_df['DATE'].str.contains(test_date_list)].index.values.tolist()

train_dataframe = pd.DataFrame(EVGPS_DataLab_visit_place_df, index=train_df_idx).reset_index(drop=True)
X_train = train_dataframe[['TEMP', 'PRECIPITATION', 'SEASON', 'AGE', 'SEX', 'COMPANION', 'TOUR_CATEGORY', 'Place_Title']]
Y_train = train_dataframe['LABEL']

test_dataframe = pd.DataFrame(EVGPS_DataLab_visit_place_df, index=test_df_idx).reset_index(drop=True)
X_test = test_dataframe[['TEMP', 'PRECIPITATION', 'SEASON', 'AGE', 'SEX', 'COMPANION', 'TOUR_CATEGORY', 'Place_Title']]
Y_test = test_dataframe['LABEL']

# from xgboost import XGBClassifier
# model = XGBClassifier(max_depth=1000,
#                       n_estimators=100,
#                       nthread=4,
#                       objective=':multi:softmax',
#                       random_state=42,
#                       num_boost_around=1000,
#                       silent=True)

# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier()

from lightgbm import LGBMClassifier
model = LGBMClassifier(n_estimators=100)

# from sklearn.neural_network import MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu',
#                     early_stopping=False, learning_rate='constant',
#                     shuffle=True, random_state=42,
#                     solver='sgd', alpha=0.3, batch_size=32,
#                     learning_rate_init=0.3, max_iter=1000)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()

# from sklearn import svm
# model = svm.SVC(kernel='rbf', degree=3, C=1)

# from sklearn.ensemble import VotingClassifier
# model = VotingClassifier(estimators=[('model: A', XGBoost), ('model: B', LightGBM)], voting='soft')

model.fit(X=X_train, y=Y_train)

Y_pred = model.predict(X_test)
Y_true = Y_test

macro_f1_score = f1_score(Y_true, Y_pred, average='macro')
micro_f1_score = f1_score(Y_true, Y_pred, average='micro')
accuracy = accuracy_score(Y_true, Y_pred)

print(f'macro_f1_score: {macro_f1_score:0.3f}')
print(f'micro_f1_score: {micro_f1_score:0.3f}')
print(f'accuracy: {accuracy:0.3f}')

from tabulate import tabulate

model_predict = pd.DataFrame({'TEMP': [23.0], 'PRECIPITATION': [13.0], 'SEASON': ['AUTUMN'], 'AGE': [20], 'SEX': ['MALE'], 'COMPANION': ['FRIEND'], 'TOUR_CATEGORY': ['OTHER TOURISM'], 'Place_Title': ['이호테우해변']},
                             columns=['TEMP', 'PRECIPITATION', 'SEASON', 'AGE', 'SEX', 'COMPANION', 'TOUR_CATEGORY', 'Place_Title'])

print()
print('-'*26 + '[ In-vehicle External Factorial & Tourist Profile ]' + '-'*26)
print(tabulate(model_predict, headers='keys', tablefmt='psql', showindex=False))

model_predict['SEASON'] = season_enc.transform(model_predict['SEASON'])
model_predict['SEX'] = sex_enc.transform(model_predict['SEX'])
model_predict['COMPANION'] = companion_enc.transform(model_predict['COMPANION'])
model_predict['TOUR_CATEGORY'] = category_enc.transform(model_predict['TOUR_CATEGORY'])
model_predict['Place_Title'] = place_title_enc.transform(model_predict['Place_Title'])

rec_output_pred = model.predict(model_predict)
print()
print('-'*26 + '[ In-vehicle target advertising system for tourism ]' + '-'*26)
print(target_enc.inverse_transform(rec_output_pred))