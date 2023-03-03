import random

import pandas as pd
import os

from tqdm.auto import tqdm
from haversine import haversine

import warnings
warnings.filterwarnings('ignore')

def rmDFunnamed(dataframe):
    for col_name in dataframe.columns:
        if 'Unnamed' in col_name: del dataframe[col_name]

root_dir = 'D:/EVGPS/v5/'

EV_mRID_df = pd.DataFrame()
for f_name in tqdm(os.listdir(root_dir + 'gps/'), desc='load EV_mRID gps dataset'):
    ev = pd.read_csv(root_dir + 'gps/' + f_name)
    EV_mRID_df = pd.concat([EV_mRID_df, ev])

EV_mRID_df.drop(EV_mRID_df.columns[[0]], axis=1, inplace=True)
EV_mRID_df.to_csv(root_dir + 'EV_mRID_gps.txt')

EV_mRID_df = pd.read_csv(root_dir + 'EV_mRID_gps.txt')
EV_mRID_df['GPS'] = EV_mRID_df['LATITUDE'].map(str) + ' | ' + EV_mRID_df['LONGITUDE'].map(str)

gps = EV_mRID_df['GPS'].value_counts().index.values
gps_cnt = EV_mRID_df['GPS'].value_counts().values

# remove outlier(equal latitude/longitude > 1000)
for i in tqdm(range(len(gps_cnt)), desc='remove outlier(equal latitude/longitude > 1000)'):
    if gps_cnt[i] > 1000:
        longP_gps = EV_mRID_df[EV_mRID_df['GPS'].str.contains(gps[i])].index
        EV_mRID_df.drop(longP_gps, inplace=True)

EV_mRID_df.drop(EV_mRID_df.columns[[0]], axis=1, inplace=True)
EV_mRID_df.to_csv(root_dir + 'EV_mRID_gps.txt')

EV_mRID_df = pd.read_csv(root_dir + 'EV_mRID_gps.txt')
rmDFunnamed(EV_mRID_df)

EV_mRID_df['VISIT_DATE'] = EV_mRID_df['YEAR'].map(str) + '-' + EV_mRID_df['MONTH'].map(str) + '-' + EV_mRID_df['DATE'].map(str)
EV_mRID_df['VISIT_GPS'] = EV_mRID_df['VISIT_DATE'].map(str) + ' / ' + EV_mRID_df['GPS'].map(str)

visit_gps = EV_mRID_df['VISIT_GPS'].value_counts().index.values
visit_gps_cnt = EV_mRID_df['VISIT_GPS'].value_counts().values

# update EVGPS visit state 10 < EVGPS state < 200
visit_place_df = pd.DataFrame(columns=['DATE', 'LATITUDE', 'LONGITUDE'])
for i in tqdm(range(len(visit_gps_cnt)), desc='find Visit state'):
    if 10 < visit_gps_cnt[i] < 200:
        visit_date = visit_gps[i].split(' / ')[0]
        visit_latitude = visit_gps[i].split(' / ')[1].split(' | ')[0]
        visit_longitude = visit_gps[i].split(' / ')[1].split(' | ')[1]
        visit_state = [visit_date, visit_latitude, visit_longitude]
        visit_place_df = visit_place_df.append(pd.Series(visit_state, index=visit_place_df.columns), ignore_index=True)

visit_place_df.to_csv(root_dir + 'visit_place.txt')

visit_place_df = pd.read_csv(root_dir + 'visit_place.txt')
rmDFunnamed(visit_place_df)

# update EVGPS visit place name concat with visit korea data lab
central_tourspot = pd.DataFrame()
for task in os.listdir(root_dir + 'visitkorea_datalab/'):
    for date_list in tqdm(os.listdir(root_dir + 'visitkorea_datalab/' + task + '/'), desc='load visitkorea datalab tourspot dataset'):
        for file_name in os.listdir(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/'):
            if '중심' in file_name:
                tourspot = pd.read_csv(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/' + file_name, encoding='cp949')
                central_tourspot = pd.concat([central_tourspot, tourspot])

central_tourspot = central_tourspot.drop_duplicates(['중심관광지명'])
central_tourspot = central_tourspot.reset_index()

datalab_visit_place_df = pd.read_csv(root_dir + 'DataLab_Tour_Info&Tourist_profile.csv', encoding='cp949')
datalab_visit_place_df = datalab_visit_place_df.dropna(axis=0).reset_index()

datalab_visit_placeTitle_list = datalab_visit_place_df['Place_Title'].value_counts().index.values.tolist()
central_tourspot_placeTitle_list = central_tourspot['중심관광지명'].value_counts().index.values.tolist()

for i in range(len(central_tourspot_placeTitle_list)):
    if central_tourspot_placeTitle_list[i] not in datalab_visit_placeTitle_list:
        central_tourspot = central_tourspot.drop(central_tourspot[central_tourspot['중심관광지명'] == central_tourspot_placeTitle_list[i]].index)

rmDFunnamed(central_tourspot)
central_tourspot = central_tourspot.reset_index()

visit_place_df['CATEGORY'] = None
visit_place_df['TOURSPOT_TITLE'] = None

# update EVGPS visit place name concat with visit korea data lab based on haversine distance
for i in tqdm(range(len(visit_place_df)), desc='find visit tourspot category & tourspot title'):
    ev_visit = (visit_place_df.loc[i]['LATITUDE'], visit_place_df.loc[i]['LONGITUDE'])

    distance = []
    for j in range(len(central_tourspot)):
        tourspot_gps = (central_tourspot.loc[j]['중심 POI Y 좌표'], central_tourspot.loc[j]['중심 POI X 좌표'])
        distance.append(haversine(ev_visit, tourspot_gps, unit='m'))
    visit_place_df.loc[i, 'CATEGORY'] = central_tourspot.loc[distance.index(min(distance))]['중심카테고리 명_중']
    visit_place_df.loc[i, 'TOURSPOT_TITLE'] = central_tourspot.loc[distance.index(min(distance))]['중심관광지명']

visit_place_df.to_csv(root_dir + 'visit_place.txt')

visit_place_df = pd.read_csv(root_dir + 'visit_place.txt')
rmDFunnamed(visit_place_df)

for i in range(len(visit_place_df)):
    visit_place_df.loc[i, 'DATE'] = visit_place_df.loc[i]['DATE'].split('-')[0] + '-' + visit_place_df.loc[i]['DATE'].split('-')[1]

temp = pd.read_csv(root_dir + 'temp.csv', encoding='cp949')
precipitation = pd.read_csv(root_dir + 'precipitation.csv', encoding='cp949')
temp['DATE'] = None
precipitation['DATE'] = None
calendar_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i in range(len(temp)):
    temp.loc[i, 'DATE'] = '20' + temp.loc[i]['년월'].split('-')[1] + '-' + str(calendar_list.index(temp.loc[i]['년월'].split('-')[0]) + 1)
    precipitation.loc[i, 'DATE'] = '20' + precipitation.loc[i]['년월'].split('-')[1] + '-' + str(calendar_list.index(precipitation.loc[i]['년월'].split('-')[0]) + 1)

visit_place_df['SEASON'] = None
visit_place_df['TEMP'] = None
visit_place_df['PRECIPITATION'] = None

# add season information based on visit date
for i in tqdm(range(len(visit_place_df)), desc='concat external factor(season, temp, precipitation)'):
    if 3 <= int(visit_place_df.loc[i]['DATE'].split('-')[1]) <= 5: season = 'SPRING'
    elif 6 <= int(visit_place_df.loc[i]['DATE'].split('-')[1]) <= 8: season = 'SUMMER'
    elif 9 <= int(visit_place_df.loc[i]['DATE'].split('-')[1]) <= 11: season = 'AUTUMN'
    else: season = 'WINTER'

    visit_place_df.loc[i, 'SEASON'] = season

    # add real-time context based on National Weather Service data
    for j in range(len(temp)):
        if visit_place_df.loc[i]['DATE'] == temp.loc[j]['DATE']:
            visit_place_df.loc[i, 'TEMP'] = temp.loc[j]['평균기온(℃)']

    for j in range(len(precipitation)):
        if visit_place_df.loc[i]['DATE'] == precipitation.loc[j]['DATE']:
            visit_place_df.loc[i, 'PRECIPITATION'] = precipitation.loc[j]['강수량(mm)']

visit_place_df.to_csv(root_dir + 'visit_place.txt')

visit_place_df = pd.read_csv(root_dir + 'visit_place.txt')
rmDFunnamed(visit_place_df)

datalab_visit_place_df = pd.read_csv(root_dir + 'DataLab_Tour_Info&Tourist_profile.csv', encoding='cp949')

visit_place_df = visit_place_df.loc[:, ['DATE', 'TEMP', 'PRECIPITATION', 'SEASON', 'CATEGORY', 'TOURSPOT_TITLE']]
visit_place_df.rename(columns={'CATEGORY': 'TOUR_CATEGORY', 'TOURSPOT_TITLE': 'Place_Title'}, inplace=True)
datalab_visit_place_df = pd.concat([datalab_visit_place_df, visit_place_df]).sort_values('DATE', ascending=True, ignore_index=True)
datalab_visit_place_df.to_csv(root_dir + 'EVGPS&DataLab.txt', encoding='cp949')

EVGPS_DataLab_visit_place_df = pd.read_csv(root_dir + 'EVGPS&DataLab.txt', encoding='cp949')
rmDFunnamed(EVGPS_DataLab_visit_place_df)

EVGPS_DataLab_visit_place_df['SEASON_VisitPlace'] = EVGPS_DataLab_visit_place_df['SEASON'].map(str) + '/' + EVGPS_DataLab_visit_place_df['Place_Title'].map(str)

EVGPS_DataLab_drop_null_df = EVGPS_DataLab_visit_place_df.dropna(axis=0).reset_index()
EVGPS_DataLab_only_null_df = EVGPS_DataLab_visit_place_df[EVGPS_DataLab_visit_place_df['AGE'].isnull()].reset_index()

# add tourist profile EVGPS state based on similar real-time context (there are several types under similar context, select a random value)
for month_visit_place in EVGPS_DataLab_drop_null_df['SEASON_VisitPlace'].value_counts().index.values.tolist():
    frequency_visit_df = EVGPS_DataLab_drop_null_df[EVGPS_DataLab_drop_null_df['SEASON_VisitPlace'].str.contains(month_visit_place)].reset_index()
    EVGPS_visit_index_list = EVGPS_DataLab_only_null_df[EVGPS_DataLab_only_null_df['SEASON_VisitPlace'].str.contains(month_visit_place)].index.values.tolist()
    if len(EVGPS_visit_index_list) > 0:
        for i in EVGPS_visit_index_list:
            EVGPS_DataLab_only_null_df.loc[i, 'AGE'] = int(random.choice(frequency_visit_df['AGE'].value_counts().index.values.tolist()))
            EVGPS_DataLab_only_null_df.loc[i, 'SEX'] = random.choice(frequency_visit_df['SEX'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'COMPANION'] = random.choice(frequency_visit_df['COMPANION'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'Top1_Relation_Tourist_Attraction'] = random.choice(frequency_visit_df['Top1_Relation_Tourist_Attraction'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'Top2_Relation_Tourist_Attraction'] = random.choice(frequency_visit_df['Top2_Relation_Tourist_Attraction'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'Top3_Relation_Tourist_Attraction'] = random.choice(frequency_visit_df['Top3_Relation_Tourist_Attraction'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'Top4_Relation_Tourist_Attraction'] = random.choice(frequency_visit_df['Top4_Relation_Tourist_Attraction'].value_counts().index.values.tolist())
            EVGPS_DataLab_only_null_df.loc[i, 'Top5_Relation_Tourist_Attraction'] = random.choice(frequency_visit_df['Top5_Relation_Tourist_Attraction'].value_counts().index.values.tolist())

EVGPS_DataLab_only_null_df = EVGPS_DataLab_only_null_df.dropna(axis=0).reset_index()
EVGPS_DataLab_visit_place_df = pd.concat([EVGPS_DataLab_only_null_df, EVGPS_DataLab_drop_null_df]).sort_values('DATE', ascending=True, ignore_index=True)

EVGPS_DataLab_visit_place_df.to_csv(root_dir + 'EVGPS&DataLab_Tour_Info&Tourist_Profile.csv', encoding='cp949')