import pandas as pd
import os

from tqdm.auto import tqdm

def rmDFunnamed(dataframe):
    for col_name in dataframe.columns:
        if 'Unnamed' in col_name: del dataframe[col_name]

root_dir = 'D:/EVGPS/v5/'

for task in os.listdir(root_dir + 'visitkorea_datalab/'):
    center_tourInfo_df = pd.DataFrame()
    for date_list in tqdm(os.listdir(root_dir + 'visitkorea_datalab/' + task + '/'), desc='load visitkorea datalab tourspot dataset'):
        center_tour = pd.DataFrame()
        for file_name in os.listdir(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/'):
            if '제주시_중심' in file_name:
                jeju_center = pd.read_csv(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/' + file_name, encoding='cp949')
                jeju_center = jeju_center.iloc[:10]
            elif '서귀포시_중심' in file_name:
                sgp_center = pd.read_csv(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/' + file_name, encoding='cp949')
                sgp_center = sgp_center.iloc[:10]

        # concat jeju center tourist attraction and seogwipo center tourist attraction
        center_tour = pd.concat([jeju_center, sgp_center]).sort_values('중심성정도_SUM', ascending=False, ignore_index=True)

        MONTH = int(str(date_list)[:4])
        DAY = int(str(date_list)[-2:])
        center_tour['DATE'] = str(MONTH) + '-' + str(DAY)

        # concat near-by related tourist attraction
        for file_name in os.listdir(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/'):
            relation_tourAttraction_name = file_name.replace('_', '/')
            for i in range(len(center_tour)):
                if center_tour.loc[i]['중심관광지명'] in relation_tourAttraction_name:
                    rel_tourlist_df = pd.read_csv(root_dir + 'visitkorea_datalab/' + task + '/' + date_list + '/' + file_name, encoding='cp949')
                    center_tour.loc[i, ['연관관광지명1', '연관관광지명2', '연관관광지명3', '연관관광지명4', '연관관광지명5']] = rel_tourlist_df.iloc[:5]['연관관광지명'].values.tolist()

        center_tourInfo_df = pd.concat([center_tourInfo_df, center_tour], ignore_index=True)

    temp = pd.read_csv(root_dir + 'temp.csv', encoding='cp949')
    precipitation = pd.read_csv(root_dir + 'precipitation.csv', encoding='cp949')
    temp['DATE'] = None
    precipitation['DATE'] = None

    calendar_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i in range(len(temp)):
        temp.loc[i, 'DATE'] = '20' + temp.loc[i]['년월'].split('-')[1] + '-' + str(
            calendar_list.index(temp.loc[i]['년월'].split('-')[0]) + 1)
        precipitation.loc[i, 'DATE'] = '20' + precipitation.loc[i]['년월'].split('-')[1] + '-' + str(
            calendar_list.index(precipitation.loc[i]['년월'].split('-')[0]) + 1)

    center_tourInfo_df['SEASON'] = None
    center_tourInfo_df['TEMP'] = None
    center_tourInfo_df['PRECIPITATION'] = None

    # add season information based on visit date
    for i in tqdm(range(len(center_tourInfo_df)), desc='concat external factor(season, temp, precipitation)'):
        if 3 <= int(center_tourInfo_df.loc[i]['DATE'].split('-')[1]) <= 5:
            season = 'SPRING'
        elif 6 <= int(center_tourInfo_df.loc[i]['DATE'].split('-')[1]) <= 8:
            season = 'SUMMER'
        elif 9 <= int(center_tourInfo_df.loc[i]['DATE'].split('-')[1]) <= 11:
            season = 'AUTUMN'
        else:
            season = 'WINTER'

        center_tourInfo_df.loc[i, 'SEASON'] = season

        # add real-time context based on National Weather Service data
        for j in range(len(temp)):
            if center_tourInfo_df.loc[i]['DATE'] == temp.loc[j]['DATE']:
                center_tourInfo_df.loc[i, 'TEMP'] = temp.loc[j]['평균기온(℃)']

        for j in range(len(precipitation)):
            if center_tourInfo_df.loc[i]['DATE'] == precipitation.loc[j]['DATE']:
                center_tourInfo_df.loc[i, 'PRECIPITATION'] = precipitation.loc[j]['강수량(mm)']

    rmDFunnamed(center_tourInfo_df)

    datalab_visit_place_df = pd.DataFrame()
    datalab_visit_place_df['DATE'] = center_tourInfo_df['DATE']
    datalab_visit_place_df['TEMP'] = center_tourInfo_df['TEMP']
    datalab_visit_place_df['PRECIPITATION'] = center_tourInfo_df['PRECIPITATION']
    datalab_visit_place_df['SEASON'] = center_tourInfo_df['SEASON']
    datalab_visit_place_df['TOUR_CATEGORY'] = center_tourInfo_df['중심카테고리 명_중']
    datalab_visit_place_df['Place_Title'] = center_tourInfo_df['중심관광지명']
    datalab_visit_place_df['Top1_Relation_Tourist_Attraction'] = center_tourInfo_df['연관관광지명1']
    datalab_visit_place_df['Top2_Relation_Tourist_Attraction'] = center_tourInfo_df['연관관광지명2']
    datalab_visit_place_df['Top3_Relation_Tourist_Attraction'] = center_tourInfo_df['연관관광지명3']
    datalab_visit_place_df['Top4_Relation_Tourist_Attraction'] = center_tourInfo_df['연관관광지명4']
    datalab_visit_place_df['Top5_Relation_Tourist_Attraction'] = center_tourInfo_df['연관관광지명5']

    datalab_visit_place_df.to_csv(root_dir + f'[{task}]DataLab_Tour_Info.txt')
    print(datalab_visit_place_df)

train_datalab_visit_place_df = pd.read_csv(root_dir + '[train]DataLab_Tour_Info.txt')
rmDFunnamed(train_datalab_visit_place_df)

test_datalab_visit_place_df = pd.read_csv(root_dir + '[test]DataLab_Tour_Info.txt')
rmDFunnamed(test_datalab_visit_place_df)

datalab_visit_place_df = pd.concat([train_datalab_visit_place_df, test_datalab_visit_place_df]).sort_values('DATE', ascending=True, ignore_index=True)
datalab_visit_place_df.to_csv(root_dir + 'DataLab_Tour_Info.csv', encoding='cp949')