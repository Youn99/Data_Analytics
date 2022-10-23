import requests
import pandas as pd 
from pandas import json_normalize 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import os
from azure.storage.blob import BlobServiceClient 
from kaggle.api.kaggle_api_extended import KaggleApi
import warnings 
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler


class Data_selection:
    list_files_csv = ['hfi_cc_2021.csv', '2015.csv', '2016.csv', '2017.csv', '2018.csv', '2019.csv']
    connection_string = "connection_string"
    def __init__(self):
        pass
    def create_containers(self):
        blob_service_client = BlobServiceClient.from_connection_string(Data_selection.connection_string)
        list_containers = ["cleanedfiles", "humanfreedom","worldhappiness"]
        for container in list_containers:
            container_client = blob_service_client.get_container_client(container)
            container_client.create_container()
        return f"Containers created with success {list_containers}."

    def download_data_from_Kaggle_to_azure(self,Kaggle_user_name, Kaggle_key):
        os.environ['KAGGLE_USERNAME'] = Kaggle_user_name
        os.environ['KAGGLE_KEY'] = Kaggle_key
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('gsutters/the-human-freedom-index', path=".", unzip=True)
        api.dataset_download_files('unsdsn/world-happiness', path= ".", unzip=True)
        blob_service_client = BlobServiceClient.from_connection_string(Data_selection.connection_string)
        container_client1 = blob_service_client.get_container_client("humanfreedom")
        container_client2 = blob_service_client.get_container_client("worldhappiness")
        for i in Data_selection.list_files_csv:
            if i == 'hfi_cc_2021.csv':
                blob_client = container_client1.get_blob_client(i)
                with open(i, "rb") as data:
                    blob_client.upload_blob(data)
            else:
                blob_client = container_client2.get_blob_client(i)
                with open(i, "rb") as data:
                    blob_client.upload_blob(data)
        return "All Data are ready, you can start."
       
class Data_analytics:
    connection_string = "connection_string"
    Southern_Asia = ['Afghanistan', 'Bangladesh','Bhutan','India','Nepal','Pakistan','Sri Lanka', 'Brunei Darussalam']
    Central_and_Eastern_Europe = ['Albania','Armenia','Azerbaijan','Belarus','Bosnia and Herzegovina','Bulgaria','Croatia','Czech Republic','Estonia','Georgia', 'Hungary','Kazakhstan','Kosovo', 'Kyrgyzstan','Latvia','Lithuania','Macedonia','Moldova','Montenegro','North Macedonia','Poland','Romania','Russia','Serbia','Slovakia','Slovenia', 'Tajikistan','Turkmenistan','Ukraine','Uzbekistan']
    Middle_East_and_Northern_Africa = ['Algeria','Bahrain','Egypt', 'Iran','Iraq','Israel','Jordan','Kuwait','Lebanon','Libya','Morocco','Palestinian Territories','Qatar','Saudi Arabia','Syria','Tunisia','Turkey','United Arab Emirates','Yemen', 'Syrian Arab Republic', 'Oman', 'Eswatini']
    Sub_Saharan_Africa = ['Angola', 'Benin','Botswana','Burkina Faso','Burundi','Cameroon','Central African Republic','Chad','Comoros','Congo','Ethiopia','Gabon','Gambia','Ghana','Guinea','Ivory Coast','Kenya','Lesotho','Liberia','Madagascar','Malawi','Mali','Mauritania','Mauritius','Mozambique','Niger','Nigeria','Rwanda','Senegal','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Swaziland','Tanzania','Togo','Uganda','Zambia','Zimbabwe', 'Namibia', 'Seychelles', 'Guinea-Bissau', 'Djibouti', 'Cabo Verde', 'Gambia, The', 'Congo (Brazzaville)','Congo (Kinshasa)','Congo, Rep.', 'Congo, Dem. Rep.']
    Latin_America_and_Caribbean = ['Argentina','Belize','Bolivia','Brazil', 'Chile', 'Colombia', 'Costa Rica','Dominican Republic','Ecuador', 'El Salvador','Guatemala', 'Haiti','Honduras','Jamaica','Mexico','Nicaragua','Panama','Paraguay','Peru','Trinidad and Tobago','Uruguay','Venezuela', 'Trinidad & Tobago', 'Suriname', 'Guyana', 'Barbados', 'Bahamas']
    Australia_and_New_Zealand = ['Australia','New Zealand', 'Papua New Guinea','Fiji' ]
    Western_Europe = ['Austria','Belgium','Cyprus','Denmark','Finland','France','Germany','Greece','Iceland','Ireland', 'Italy','Luxembourg','Malta','Netherlands','North Cyprus','Northern Cyprus','Norway','Portugal','Spain','Sweden','Switzerland','United Kingdom']
    Southeastern_Asia = ['Cambodia','Indonesia','Laos','Malaysia','Myanmar','Philippines','Singapore','Thailand','Vietnam', 'Timor-Leste']
    North_America = ['Canada','United States']
    Eastern_Asia = ['China','Hong Kong','Hong Kong S.A.R., China','Japan','Mongolia', 'South Korea','Taiwan','Taiwan Province of China']
    def __init__(self):

        pass

    def upload_from_API(self, year):
        self.year = year
        self.Whreport_list = []
        self.Freedomindex_list = []
        for anno in self.year: 
                Whreport = f"https://happiness08.azurewebsites.net/api/WHReport?anno={anno}" # it still working you can try by inserting a year from 2016 to 2019, you need just to copy and paste this url in your browser by adding a year.
                Freedomindex = f"https://happiness08.azurewebsites.net/api/FreedomIndex?anno={anno}"
                Whreport_request = requests.get(Whreport).json()
                Freedomindex_request = requests.get(Freedomindex).json()
                self.Whreport_list.append(json_normalize(Whreport_request[str(anno)]))
                self.Freedomindex_list.append(json_normalize(Freedomindex_request[str(anno)]))
    def Get_FreedomIndex_dataset(self):
        Fred_2015 = self.Freedomindex_list[0]
        Fred_2016 = self.Freedomindex_list[1]
        Fred_2017 = self.Freedomindex_list[2]
        Fred_2018 = self.Freedomindex_list[3]
        Fred_2019 = self.Freedomindex_list[4]
        fir_append = Fred_2015.append(Fred_2016)
        sec_append = fir_append.append(Fred_2017)
        thir_append = sec_append.append(Fred_2018)
        Fred_index = thir_append.append(Fred_2019)
        Fred_index.loc[Fred_index['countries'] == 'Venezuela, RB', 'countries'] = 'Venezuela'
        Fred_index.loc[Fred_index['countries'] == 'Iran, Islamic Rep.', 'countries'] = 'Iran'
        Fred_index.loc[Fred_index['countries'] == 'Slovak Republic', 'countries'] = 'Slovakia'
        Fred_index.loc[Fred_index['countries'] == 'Russian Federation', 'countries'] = 'Russia'
        Fred_index.loc[Fred_index['countries'] == 'Lao PDR', 'countries'] = 'Laos'
        Fred_index.loc[Fred_index['countries'] == 'Kyrgyz Republic', 'countries'] = 'Kyrgyzstan'
        Fred_index.loc[Fred_index['countries'] == 'Korea, Rep.', 'countries'] = 'South Korea'
        Fred_index.loc[Fred_index['countries'] == 'Hong Kong SAR, China', 'countries'] = 'Hong Kong'
        Fred_index.loc[Fred_index['countries'] == 'Gambia, The ', 'countries'] = 'Gambia'
        Fred_index.loc[Fred_index['countries'] == 'Egypt, Arab Rep.', 'countries'] = 'Egypt'
        Fred_index.loc[Fred_index['countries'] == "Cote d'Ivoire", 'countries'] = 'Ivory Coast'
        Fred_index.loc[Fred_index['countries'] == 'Bahamas, The', 'countries'] = 'Bahamas'
        Fred_index.loc[Fred_index['countries'] == 'Yemen, Rep.', 'countries'] = 'Yemen'
        Fred_index.loc[Fred_index['countries'] == 'Gambia, The', 'countries'] = 'Gambia'
        Fred_index.loc[Fred_index['countries'] == 'North Macedonia', 'countries'] = 'Macedonia'
        Fred_index.loc[Fred_index['countries'] == 'Syrian Arab Republic', 'countries'] = 'Syria'
        Fred_index['Region'] = Fred_index['countries']
        for i in Fred_index.Region:
            if i in Data_analytics.Southern_Asia:
                Fred_index["Region"].replace({i: "Southern Asia"}, inplace=True)
            elif i in Data_analytics.Central_and_Eastern_Europe:
                Fred_index["Region"].replace({i: "Central and Eastern Europe"}, inplace=True)
            elif i in Data_analytics.Middle_East_and_Northern_Africa:
                Fred_index["Region"].replace({i: "Middle East and Northern Africa"}, inplace=True)
            elif i in Data_analytics.Sub_Saharan_Africa:
                Fred_index["Region"].replace({i: "Sub-Saharan Africa"}, inplace=True)
            elif i in Data_analytics.Latin_America_and_Caribbean:
                Fred_index["Region"].replace({i: "Latin America and Caribbean"}, inplace=True)
            elif i in Data_analytics.Australia_and_New_Zealand:
                Fred_index["Region"].replace({i: "Australia and New Zealand"}, inplace=True)
            elif i in Data_analytics.Western_Europe:
                Fred_index["Region"].replace({i: "Western Europe"}, inplace=True)
            elif i in Data_analytics.Southeastern_Asia:
                Fred_index["Region"].replace({i: "Southeastern Asia"}, inplace=True)
            elif i in Data_analytics.North_America:
                Fred_index["Region"].replace({i: "North America"}, inplace=True)
            elif i in Data_analytics.Eastern_Asia:
                Fred_index["Region"].replace({i: "Eastern Asia"}, inplace=True)
        Freedom_index = Fred_index.drop(['region'], axis = 1)
        Freedom_index = Freedom_index.rename(columns={'countries': 'Country'})
        Freedom_index = Freedom_index.rename(columns={'year': 'Year'})
        Freedom_index.sort_values(by=['Country', 'Year'])

        return Freedom_index

    def Get_WordHappiness_dataset(self):

        Data_2015 = self.Whreport_list[0]
        Data_2016 = self.Whreport_list[1]
        Data_2017 = self.Whreport_list[2]
        Data_2018 = self.Whreport_list[3]
        Data_2019 = self.Whreport_list[4]
        Data_2015['Year'] = 2015
        Data_2016['Year'] = 2016
        Data_2017['Year'] = 2017
        Data_2018['Year'] = 2018
        Data_2019['Year'] = 2019
        Data_2015_new = Data_2015.drop(['Standard Error', 'Dystopia Residual'], axis = 1)
        Data_2016_new = Data_2016.drop(['Lower Confidence Interval', 'Upper Confidence Interval', 'Dystopia Residual'], axis = 1)
        Data_1516 = Data_2015_new.append(Data_2016_new)
        Data_1516.loc[Data_1516['Country'] == 'Somaliland region', 'Country'] = 'Somaliland Region'
        Data_1516.loc[Data_1516['Country'] == 'Congo (Brazzaville)', 'Country'] = 'Congo, Rep.'
        Data_1516.loc[Data_1516['Country'] == 'Congo (Kinshasa)', 'Country'] = 'Congo, Dem. Rep.'
        Data_2017_new = Data_2017.drop(['Whisker.high','Whisker.low', 'Dystopia.Residual'], axis = 1)
        Data_2017_new.columns = ['Country', 'Happiness Rank','Happiness Score', 'Economy (GDP per Capita)','Family', 'Health (Life Expectancy)','Freedom','Generosity', 'Trust (Government Corruption)', 'Year']
        Data_2018.columns = ['Happiness Rank', 'Country','Happiness Score', 'Economy (GDP per Capita)','Family','Health (Life Expectancy)', 'Freedom', 'Generosity','Trust (Government Corruption)', 'Year']
        Data_2019.columns = ['Happiness Rank', 'Country','Happiness Score', 'Economy (GDP per Capita)','Family','Health (Life Expectancy)', 'Freedom', 'Generosity','Trust (Government Corruption)', 'Year']
        app_data = Data_2017_new.append(Data_2018)
        Data_171819 = app_data.append(Data_2019)
        Data_171819.loc[Data_171819['Country'] == 'Hong Kong S.A.R., China', 'Country'] = 'Hong Kong'
        Data_171819.loc[Data_171819['Country'] == 'Taiwan Province of China', 'Country'] = 'Taiwan'
        Data_171819.loc[Data_171819['Country'] == 'Trinidad & Tobago', 'Country'] = 'Trinidad and Tobago'
        Data_171819.loc[Data_171819['Country'] == 'North Macedonia', 'Country'] = 'Macedonia'
        Data_171819.loc[Data_171819['Country'] == 'Northern Cyprus', 'Country'] = 'North Cyprus'
        Data_171819.loc[Data_171819['Country'] == 'Congo (Brazzaville)', 'Country'] = 'Congo, Rep.'
        Data_171819.loc[Data_171819['Country'] == 'Congo (Kinshasa)', 'Country'] = 'Congo, Dem. Rep.'
        Data_171819['region'] = Data_171819['Country']

        for i in Data_171819.region:
            if i in Data_analytics.Southern_Asia:
                Data_171819["region"].replace({i: "Southern Asia"}, inplace=True)
            elif i in Data_analytics.Central_and_Eastern_Europe:
                Data_171819["region"].replace({i: "Central and Eastern Europe"}, inplace=True)
            elif i in Data_analytics.Middle_East_and_Northern_Africa:
                Data_171819["region"].replace({i: "Middle East and Northern Africa"}, inplace=True)
            elif i in Data_analytics.Sub_Saharan_Africa:
                Data_171819["region"].replace({i: "Sub-Saharan Africa"}, inplace=True)
            elif i in Data_analytics.Latin_America_and_Caribbean:
                Data_171819["region"].replace({i: "Latin America and Caribbean"}, inplace=True)
            elif i in Data_analytics.Australia_and_New_Zealand:
                Data_171819["region"].replace({i: "Australia and New Zealand"}, inplace=True)
            elif i in Data_analytics.Western_Europe:
                Data_171819["region"].replace({i: "Western Europe"}, inplace=True)
            elif i in Data_analytics.Southeastern_Asia:
                Data_171819["region"].replace({i: "Southeastern Asia"}, inplace=True)
            elif i in Data_analytics.North_America:
                Data_171819["region"].replace({i: "North America"}, inplace=True)
            elif i in Data_analytics.Eastern_Asia:
                Data_171819["region"].replace({i: "Eastern Asia"}, inplace=True)
        Data_171819 = Data_171819.rename(columns={'region': 'Region'})
        World_happi = Data_1516.append(Data_171819)
        World_happi.loc[World_happi['Country'] == 'Swaziland', 'Country'] = 'Eswatini'
        World_happi.sort_values(by=['Country','Year'], inplace = True)

        return World_happi

    def Get_column_with_nan(self, Dataframe):

        self.Dataframe = Dataframe
        column_with_nan = []
        column_with_nan1 = []
        column_with_nan3 = []
        column_with_nan4 = []
        Df = ['Freedom Index', 'World Happiness', 'Full Dataset', 'Dataset']
        if self.Dataframe in Df:
            if self.Dataframe == 'Freedom Index':
                col = self.Get_FreedomIndex_dataset()
                column_dict1 = dict(col.isnull().sum())
                for i in column_dict1.items():
                    if i[1] > 0:
                        column_with_nan.append(i)
                print(column_with_nan)
            if self.Dataframe == 'World Happiness':
                col2 = self.Get_WordHappiness_dataset()
                column_dict2 = dict(col2.isnull().sum())
                for ii in column_dict2.items():
                    if ii[1] > 0:
                        column_with_nan1.append(ii)
                print(column_with_nan1)
            if self.Dataframe == 'Full Dataset':
                col3 = self.Merge_our_datasets()
                column_dict3 = dict(col3.isnull().sum())
                for a in column_dict3.items():
                    if a[1] > 0:
                        column_with_nan3.append(a)
                print(column_with_nan3)
            if self.Dataframe == 'Dataset':
                col4 = self.clean_with_mean_nan()
                column_dict4 = dict(col4.isnull().sum())
                for b in column_dict4.items():
                    if b[1] > 0:
                        column_with_nan4.append(b)
                print(column_with_nan4)
            
        else:
            print(f"Dataframe not disponibile reprove with: [{Df}]")

    def Get_nan_of_Fred_index(self):

        fred_index = self.Get_FreedomIndex_dataset()
        nan_Fred_index = fred_index[fred_index.isna().any(axis=1)]

        return nan_Fred_index

    def Get_nan_of_World_Happiness(self):

        world_happiness= self.Get_WordHappiness_dataset()
        nan_world_happiness = world_happiness[world_happiness.isna().any(axis=1)]

        return nan_world_happiness

    def Get_nan_of_fulldataset(self):

        full_dataset= self.Merge_our_datasets()
        nan_full_dataset = full_dataset[full_dataset.isna().any(axis=1)]

        return nan_full_dataset

    def Get_info_about_countries_ofworldhappiness(self):

        World_happi = self.Get_WordHappiness_dataset()

        return World_happi['Country'].value_counts()[World_happi['Country'].value_counts()!=5]

    def Merge_our_datasets(self):

        Happ = self.Get_WordHappiness_dataset()
        Fred = self.Get_FreedomIndex_dataset()
        Full_dataset = pd.merge(Happ, Fred, how='inner', on = ['Year', 'Country', 'Region'])

        return Full_dataset

    def set_index(self):

        full_dataset = self.Merge_our_datasets()
        full_dataset.set_index('Country', inplace=True)

        return full_dataset

    def nan_(self,state):

            if np.size(self.full_dataset.loc[state].Year)>1:
                self.full_dataset.loc[state] = self.full_dataset.loc[state].fillna(value=self.full_dataset.loc[state].mean())
            else:
                self.full_dataset.loc[str(state)] = self.full_dataset.loc[str(state)].fillna(self.full_dataset.mean())

    def list_nan(self):

        full_data = self.Merge_our_datasets()
        lista_nan = list(full_data[full_data.isna().any(axis=1)].Country.unique())
        return lista_nan

    def clean_with_mean_nan(self):

        lista_nan = self.list_nan()
        self.full_dataset = self.Merge_our_datasets()
        self.full_dataset.set_index('Country', inplace=True)
        for element in lista_nan:
            self.nan_(element)
        full_dataset_ = self.full_dataset.fillna(self.full_dataset.mean())
        full_dataset_.reset_index(inplace=True)

        return full_dataset_

    def Generate_datasets_for_powerBI(self):

        hf = ['ISO','Year','hf_score','hf_quartile', 'hf_rank']
        pf_rol = ['ISO', 'Year','pf_rol_procedural','pf_rol_civil','pf_rol_criminal','pf_rol']
        pf_ss =['ISO', 'Year','pf_ss_homicide','pf_ss_disappearances_disap','pf_ss_disappearances_violent','pf_ss_disappearances_organized','pf_ss_disappearances_fatalities','pf_ss_disappearances_injuries','pf_ss_disappearances_torture','pf_ss_killings','pf_ss_disappearances','pf_ss']
        pf_movement = ['ISO','Year','pf_movement_vdem_foreign','pf_movement_vdem_men','pf_movement_vdem_women','pf_movement_vdem','pf_movement_cld','pf_movement']
        pf_religion = ['ISO','Year','pf_religion_suppression','pf_religion_freedom_vdem','pf_religion_freedom_cld','pf_religion_freedom','pf_religion']
        pf_assembly = ['ISO','Year','pf_assembly_entry','pf_assembly_freedom_house','pf_assembly_freedom_bti','pf_assembly_freedom_cld','pf_assembly_freedom','pf_assembly_parties_barriers','pf_assembly_parties_bans','pf_assembly_parties_auton','pf_assembly_parties','pf_assembly_civil','pf_assembly','pf_assembly_rank']
        pf_expression = ['ISO','Year','pf_assembly_rank','pf_expression_killed','pf_expression_jailed','pf_expression_media','pf_expression_cultural','pf_expression_gov','pf_expression_internet','pf_expression_harass','pf_expression_selfcens','pf_expression_freedom_bti','pf_expression_freedom_cld','pf_expression_freedom','pf_expression','pf_expression_rank']
        pf_identity = ['ISO','Year','pf_identity_same_m','pf_identity_same_f','pf_identity_same','pf_identity_divorce','pf_identity_fgm','pf_identity_inheritance_widows','pf_identity_inheritance_daughters','pf_identity_inheritance','pf_identity']
        pf = ['ISO','Year','pf_score','pf_womens', 'pf_rank']
        ef_government = ['ISO','Year','ef_government_consumption','ef_government_transfers','ef_government_enterprises','ef_government_tax_income','ef_government_tax_payroll','ef_government_tax','ef_government_soa','ef_government']
        ef_legal = ['ISO','Year','ef_legal_judicial','ef_legal_courts','ef_legal_protection','ef_legal_military','ef_legal_integrity','ef_legal_enforcement','ef_legal_regulatory','ef_legal_police','ef_legal']
        ef_money = ['ISO','Year','ef_money_growth','ef_money_sd','ef_money_inflation','ef_money_currency','ef_money']
        ef_trade = ['ISO','Year','ef_trade_tariffs_revenue','ef_trade_tariffs_mean','ef_trade_tariffs_sd','ef_trade_tariffs','ef_trade_regulatory_nontariff','ef_trade_regulatory_compliance','ef_trade_regulatory','ef_trade_black','ef_trade_movement_foreign','ef_trade_movement_capital','ef_trade_movement_visit','ef_trade_movement','ef_trade']
        ef_regulation = ['ISO','Year','ef_regulation_credit_ownership','ef_regulation_credit_private','ef_regulation_credit_interest','ef_regulation_credit','ef_regulation_labor_minwage','ef_regulation_labor_firing','ef_regulation_labor_bargain','ef_regulation_labor_hours','ef_regulation_labor_dismissal','ef_regulation_labor_conscription','ef_regulation_labor','ef_regulation_business_adm','ef_regulation_business_bureaucracy','ef_regulation_business_start','ef_regulation_business_bribes','ef_regulation_business_licensing','ef_regulation_business_compliance','ef_regulation_business','ef_regulation', 'ef_rank','ef_score']
        Dim = ['Region','Country','Year','ISO']
        full_dataset = self.clean_with_mean_nan()
        hf_ = full_dataset[hf]
        pf_rol_ = full_dataset[pf_rol]
        pf_ss_ =full_dataset[pf_ss]
        pf_movement_ = full_dataset[pf_movement]
        pf_religion_ = full_dataset[pf_religion]
        pf_assembly_ = full_dataset[pf_assembly]
        pf_expression_ = full_dataset[pf_expression]
        pf_identity_ = full_dataset[pf_identity]
        pf_ = full_dataset[pf]
        ef_government_ = full_dataset[ef_government]
        ef_legal_ = full_dataset[ef_legal]
        ef_money_ = full_dataset[ef_money]
        ef_trade_ = full_dataset[ef_trade]
        ef_regulation_ = full_dataset[ef_regulation]
        Dim_ = full_dataset[Dim]
        hf_.to_csv('hf.csv', index=False)
        pf_rol_.to_csv('pf_rol.csv', index=False)
        pf_ss_.to_csv('pf_ss.csv', index= False)
        pf_movement_.to_csv('pf_movement.csv', index = False)
        pf_religion_.to_csv('pf_religion.csv', index = False)
        pf_assembly_.to_csv('pf_assembly.csv', index = False)
        pf_expression_.to_csv('pf_expression.csv', index = False)
        pf_identity_.to_csv('pf_identity.csv', index = False)
        pf_.to_csv('pf.csv', index = False)
        ef_government_.to_csv('ef_government.csv', index = False)
        ef_legal_.to_csv('ef_legal.csv', index= False)
        ef_money_.to_csv('ef_money.csv', index= False)
        ef_trade_.to_csv('ef_trade.csv', index = False )
        ef_regulation_.to_csv('ef_regulation.csv', index = False )
        Dim_.to_csv('Dim.csv', index = False)

        return "Done!"

    def Upload_in_azure_blob(self):

        blob_service_client = BlobServiceClient.from_connection_string(Data_analytics.connection_string)
        container_client = blob_service_client.get_container_client("cleanedfiles")
        lista_csv = ['hf.csv','pf_rol.csv', 'pf_ss.csv', 'pf_movement.csv', 'pf_religion.csv','pf_assembly.csv','pf_expression.csv','pf_identity.csv', 'pf.csv', 'ef_government.csv','ef_legal.csv','ef_money.csv','ef_trade.csv','ef_regulation.csv','Dim.csv']
        for i in lista_csv:
            blob_client = container_client.get_blob_client(i)
            with open(i, "rb") as data:
                blob_client.upload_blob(data)

    def generate_data_for_Tableau(self):

        Word_happi = self.Get_WordHappiness_dataset()
        Word_happi.to_csv('World_happiness.csv', index = False)

        return "Done"

    def generate_scaled_data_for_tableau(self):

        World_happi = self.Get_WordHappiness_dataset()
        scaler = MinMaxScaler()
        World_happi.iloc[:,4:10] = scaler.fit_transform(World_happi.iloc[:,4:10])
        h_cols = ['Country', 'Year','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Generosity','Trust (Government Corruption)']
        rank_df = World_happi[h_cols[1:]].rank(axis=0,numeric_only=True, method='dense', ascending=False)
        rank_df['Country'] = World_happi['Country']
        rank_df['True Influence'] = World_happi[h_cols[2:5]].rank(axis=0,numeric_only=True, method='dense').idxmax(axis=1)
        for n in rank_df.Year:
            if n == 5:
                rank_df.Year.replace({n: 2015}, inplace= True)
            elif n == 4:
                rank_df.Year.replace({n: 2016}, inplace= True)
            elif n == 3:
                rank_df.Year.replace({n: 2017}, inplace= True)
            elif n == 2:
                rank_df.Year.replace({n: 2018}, inplace= True)
            elif n == 1:
                rank_df.Year.replace({n: 2019}, inplace= True)
        rank_df.to_csv('Ranked_WorldHappiness.csv')

        return 'Done'

class Machine_learning(Data_analytics):

    def __init__(self):
        pass

    def load_dataset(self):

        inherit = Data_analytics()
        years = [year for year in range(2015, 2020)]
        api =inherit.upload_from_API(years)
        FulldataML = inherit.clean_with_mean_nan()
        lista_rank = ['hf_rank','pf_rank', 'ef_rank']
        for i in lista_rank:
            new_full_dataML =FulldataML.drop(i, axis = 1)

        return new_full_dataML

    def filter_by_correlation(self, number, percent):
        self.number = number
        self.percent = percent
        data = self.load_dataset()
        lista_columns = list(data.columns)
        drp_v = []
        for n in lista_columns:
            if n.count('_') > number:
                drp_v.append(n)
        new_fulldata = data.drop(drp_v, axis=1)
        dicto = dict(new_fulldata.corr().loc['Happiness Score'])
        lista_features = []
        for i in dicto.items():
            if i[1] > percent:
                lista_features.append(i[0])

        return lista_features

    def get_features(self):

        fullDataML = self.load_dataset()
        list_features = self.filter_by_correlation(self.number, self.percent)
        feat = fullDataML[list_features]
        features = feat.drop(['Happiness Score'], axis= 1)

        return features

    def get_target(self, target):

        self.target = target
        fullDataML = self.load_dataset()
        target = fullDataML[target]

        return target

    def get_data_for_classification(self):

        fullDataML = self.load_dataset()
        list_features = self.filter_by_correlation(self.number, self.percent)
        feat = fullDataML[list_features]
        bins = [2, 3.5, 4, 5.5, 7, 8.5]
        fasce = ['2-3.5', '3.5-4', '4-5.5', '5.5-7', '7-8.5']
        d = dict(enumerate(fasce, 1))
        feat['New_target'] = np.vectorize(d.get)(np.digitize((feat['Happiness Score'].astype(int)), bins))
        Data_ML = feat.drop(['Happiness Score'], axis=1)
        Data_ML= Data_ML.rename(columns={'New_target': 'Happiness_Score'})

        return Data_ML

    def metric(self, y_true, y_pred):

        true_positive = np.sum(np.logical_and(y_true == 1, y_pred == y_true))
        true_negative = np.sum(np.logical_and(y_pred == 0, y_pred == y_true))
        false_positive = np.sum(np.logical_and(y_pred == 1, y_pred != y_true))
        false_negative = np.sum(np.logical_and(y_pred == 0, y_pred != y_true))
        Accuracy = (true_positive + true_negative) / (true_positive+true_negative+false_positive+false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        err_rate = (false_positive + false_negative) / (true_positive+true_negative+false_positive+false_negative)
        specificity = true_negative / (true_negative + false_positive)# True Negative Rate
         
        print(f'The True positive are: {true_positive}\nThe True negative are: {true_negative}\nThe False positive are {false_positive}\nThe False negative are {false_negative}\nThe Accurancy is {Accuracy}\nThe precision is {precision}\nThe recall is {recall}\nThe error rate is {err_rate}\nThe True negative rate is {specificity}')