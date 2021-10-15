import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

warnings.filterwarnings('ignore')
import joblib

df = pd.read_excel('readyForModelisation2.xlsx', index_col=0)

from sklearn.compose import make_column_selector

categorical = make_column_selector(dtype_exclude=np.number)
numerical = make_column_selector(dtype_include=np.number)
x = df[categorical]

df.replace(999, np.nan, inplace=True)
df.replace(11, np.nan, inplace=True)

StoreEval = df[['Info_Facility_Understand_numeric', 'Visit_Eval_numeric',
                'Store_Staff_numeric', 'Request_Comprehension_numeric',
                'Proposed_solution_in_Store_numeric',
                'privileged welcome as a business customer_numeric', 'Waiting_Time_in_Store',
                'Commercial_Eval', 'Time_Before_talk', 'commercial_understanding', 'commercial_solution',
                'Visit_Eval.1',
                ]]

AppelEval = df[['Network_Quality', 'Rate_SOS',
                'Call_efficiency', 'Network_Coverage', 'Call_Voice_Quality',
                'Communication_Quality', 'Voice_Comm_Inside', 'Voice_Comm_Outside',
                'Rate_Conf_Call', 'Rate_Voice_Message']]

InternetEval = df[['Quality_Internet_Connection',
                   'Mob_Internet_Accessibility', 'Navigation_Speed_Mobile',
                   'Price_Quality_Mob_Int', 'Rate_Trans_Internet']]

RoamingEval = df[['Rate_Roaming_Service', 'Network_Accessibility', 'Quality_Voice',
                  'Quality_Mobile_Internet', 'Info_cost_Roaming_Service',
                  'Cost_Roaming_Service']]

df_clean = df.drop(['Info_Facility_Understand_numeric', 'Visit_Eval_numeric',
                    'Store_Staff_numeric', 'Request_Comprehension_numeric',
                    'Proposed_solution_in_Store_numeric',
                    'privileged welcome as a business customer_numeric', 'Waiting_Time_in_Store',
                    'Commercial_Eval', 'Time_Before_talk', 'commercial_understanding', 'commercial_solution',
                    'Visit_Eval.1', 'Network_Quality', 'Rate_SOS',
                    'Call_efficiency', 'Network_Coverage', 'Call_Voice_Quality',
                    'Communication_Quality', 'Voice_Comm_Inside', 'Voice_Comm_Outside',
                    'Rate_Conf_Call', 'Rate_Voice_Message', 'Quality_Internet_Connection',
                    'Mob_Internet_Accessibility', 'Navigation_Speed_Mobile',
                    'Price_Quality_Mob_Int', 'Rate_Trans_Internet', 'Rate_Roaming_Service', 'Network_Accessibility',
                    'Quality_Voice',
                    'Quality_Mobile_Internet', 'Info_cost_Roaming_Service',
                    'Cost_Roaming_Service'], axis=1, errors='ignore')

df_clean['Appel_score'] = AppelEval.mean(axis=1)
df_clean['Roamnig_score'] = RoamingEval.mean(axis=1)
df_clean['Internet_score'] = InternetEval.mean(axis=1)
df_clean['Store_score'] = StoreEval.mean(axis=1)
AppelEval['Appel_score'] = AppelEval.mean(axis=1)
RoamingEval['Roamnig_score'] = RoamingEval.mean(axis=1)
InternetEval['Internet_score'] = InternetEval.mean(axis=1)
StoreEval['store_score'] = StoreEval.mean(axis=1)
df_clean['note'] = df_clean[['Appel_score', 'Roamnig_score', 'Internet_score', 'Store_score']].mean(axis=1)

y_appel = df_clean['Appel_score']
y_roaming = df_clean['Roamnig_score']
y_internet = df_clean['Internet_score']
y_store = df_clean['Store_score']
y_note = df_clean['note']
status = pd.get_dummies(x, drop_first=False)
df_clean = pd.concat([y_appel, y_roaming, y_internet, y_store, status], axis=1)

AppelEval = pd.concat([AppelEval['Appel_score'], status], axis=1)
RoamingEval = pd.concat([RoamingEval['Roamnig_score'], status], axis=1)
InternetEval = pd.concat([InternetEval['Internet_score'], status], axis=1)
StoreEval.dropna(inplace=True)

# Putting feature variable to X
X = df_clean[['city_GABES', 'city_SFAX', 'city_TUNIS', 'Sect_Comp_Administration',
              'Sect_Comp_Assurances', 'Sect_Comp_Banque', 'Sect_Comp_Commerce',
              'Sect_Comp_INFORMATIQUE', 'Sect_Comp_Industrie', 'Sect_Comp_Santé',
              'Sect_Comp_Services', 'Sect_Comp_Tourisme', 'Opert_Princ_Ooredoo',
              'Opert_Princ_Orange', 'Opert_Princ_Tunisie Télécom', 'Opert_Period_1 an à moins de 2 ans',
              'Opert_Period_2 ans à moins de 3 ans', 'Opert_Period_3 ans ou plus',
              'Opert_Period_3 mois à moins de 6 mois',
              'Opert_Period_6 mois à moins de 12 mois',
              'Opert_Period_Il y a moins de 3 mois',
              'Abonnement_Orange_ Certains ou tous les employés ont leur propre abonnement  mobile et ne se font pas rembourser du tout leur facture par votre entreprise',
              'Abonnement_Orange_ Certains ou tous les employés ont souscrit leur propre abonnement  mobile mais se font rembourser en partie ou en totalité leur facture par l’entreprise',
              'Abonnement_Orange_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en intégralité par l’entreprise.',
              'Abonnement_Orange_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en partie par l’entreprise',
              'Abonnement_Ooredoo_ Certains ou tous les employés ont leur propre abonnement  mobile et ne se font pas rembourser du tout leur facture par votre entreprise',
              'Abonnement_Ooredoo_ Certains ou tous les employés ont souscrit leur propre abonnement  mobile mais se font rembourser en partie ou en totalité leur facture par l’entreprise',
              'Abonnement_Ooredoo_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en intégralité par l’entreprise.',
              'Abonnement_Ooredoo_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en partie par l’entreprise',
              'Abonnement_Telecom_ Certains ou tous les employés ont leur propre abonnement  mobile et ne se font pas rembourser du tout leur facture par votre entreprise',
              'Abonnement_Telecom_ Certains ou tous les employés ont souscrit leur propre abonnement  mobile mais se font rembourser en partie ou en totalité leur facture par l’entreprise',
              'Abonnement_Telecom_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en intégralité par l’entreprise.',
              'Abonnement_Telecom_ Les abonnements mobiles sont fournis par l’entreprise à certains ou à tous les employés et les factures sont payées en partie par l’entreprise',
              'Opert_Princ_Type_Offer_Offre hybride',
              'Opert_Princ_Type_Offer_Offre mobile postpayé',
              'Opert_Princ_Type_Offer_Offre mobile postpayée avec tarif préférentiel intra-flotte',
              'Opert_Princ_Type_Offer_Offre mobile prépayée',
              'Opert_Princ_Type_Offer_Offre mobile prépayée avec tarif préférentiel intra-flotte',
              ]]

X.rename(columns={'city_GABES': 'Gabes',
                  'city_SFAX': 'Sfax', 'city_TUNIS': 'Tunis', 'Sect_Comp_Administration': 'Administration',
                  'Sect_Comp_Assurances': 'Assurance', 'Sect_Comp_Banque': 'Banque',
                  'Sect_Comp_Commerce': 'Commerce', 'Sect_Comp_INFORMATIQUE': 'Informatique',
                  'Sect_Comp_Industrie': 'Industrie', 'Sect_Comp_Santé': 'Santé', 'Sect_Comp_Services': 'Services',
                  'Sect_Comp_Tourisme': 'Tourisme', 'Opert_Princ_Orange': 'Orange Tunisie',
                  'Opert_Princ_Tunisie Télécom': 'Tunisie Télécom', 'Opert_Princ_Ooredoo': 'Ooredoo',
                  'Opert_Period_1 an à moins de 2 ans': 'Entre 1 et 2 ans',
                  'Opert_Period_2 ans à moins de 3 ans': 'Entre 2 et 3 ans', 'Opert_Period_3 ans ou plus': '3 ans plus',
                  'Opert_Period_3 mois à moins de 6 mois': 'Entre 3 et 6 mois',
                  'Opert_Period_6 mois à moins de 12 mois': 'Entre 6 et 12 mois',
                  'Opert_Period_Il y a moins de 3 mois': '3 mois',
                  'Opert_Princ_Type_Offer_Offre hybride': 'Hybride',
                  'Opert_Princ_Type_Offer_Offre mobile postpayé': 'Mobile Postpayé',
                  'Opert_Princ_Type_Offer_Offre mobile postpayée avec tarif préférentiel intra-flotte': 'Mobile postpayée avec tarif préférentiel intra-flotte',
                  'Opert_Princ_Type_Offer_Offre mobile prépayée': 'Mobile prépayée',
                  'Opert_Princ_Type_Offer_Offre mobile prépayée avec tarif préférentiel intra-flotte': 'Mobile prépayée avec tarif préférentiel intra-flotte'

                  }, inplace=True)
X = pd.read_excel("features.xlsx")

X_train_note, X_test_note, y_train_note, y_test_note = train_test_split(X, y_note, train_size=0.8, random_state=43)
extNote = ExtraTreesRegressor(random_state=42)
extNote.fit(X_train_note, y_train_note)

X_train_roaming, X_test_roaming, y_train_roaming, y_test_roaming = train_test_split(X, y_roaming, train_size=0.8,
                                                                                    random_state=43)
bagRoaming = BaggingRegressor(random_state=42)
bagRoaming.fit(X_train_roaming, y_train_roaming)

X_train_internet, X_test_internet, y_train_internet, y_test_internet = train_test_split(X, y_internet, train_size=0.8,
                                                                                        random_state=43)
extInternet = ExtraTreesRegressor(random_state=42)
extInternet.fit(X_train_internet, y_train_internet)

X_train_appel, X_test_appel, y_train_appel, y_test_appel = train_test_split(X, y_appel, train_size=0.8, random_state=43)
bagAppel = BaggingRegressor(random_state=42)
bagAppel.fit(X_train_appel, y_train_appel)

note = "Deploiement-project/note.sav"
appel = "Deploiement-project/appel.sav"
roaming = "Deploiement-project/roaming.sav"
internet = "Deploiement-project/internet.sav"
features = "Deploiement-project/features.sav"
joblib.dump(extNote, note)
joblib.dump(bagAppel, appel)
joblib.dump(bagRoaming, roaming)
joblib.dump(extInternet, internet)
joblib.dump(X, features)
