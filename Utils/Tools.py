import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns






def transform_numerical_to_categorical(df):
    category_features = []
    threshold = 2
    for each in df.columns:
        if df[each].nunique() <= threshold:
            category_features.append(each)

    print("transformed from numerical to categorical : ", category_features)

    for each in category_features:
        df[each] = df[each].astype('category')

    obj_col = df.select_dtypes("object").columns
    for each in obj_col:
        df[each] = df[each].astype('category')

    return df


def clean_map(string):
    '''nettoyage des caractères de liste en sortie de LIME as_list'''
    signes = [' => ', ' <= ', '<', '>', "="]
    for signe in signes:
        if signe in string:
            signe_confirme = signe
        string = string.replace(signe, '____')
    string = string.split('____')
    if string[0][-1] == ' ':
        string[0] = string[0][:-1]

    return (string, signe_confirme)



def model_local_interpretation(explanation):
    
    df_map = pd.DataFrame(explanation.as_list())
    print("df map", df_map)
    df_map['feature'] = df_map[0].apply(lambda x: clean_map(x)[0][0])
    df_map['signe'] = df_map[0].apply(lambda x: clean_map(x)[1])
    df_map['val_lim'] = df_map[0].apply(lambda x: clean_map(x)[0][-1]).astype('float').astype('int')
    df_map['ecart'] = df_map[1]

    df_map = df_map[['feature', 'signe', 'val_lim', 'ecart']]


    df_map_filtered_plus = df_map.sort_values(by='ecart', ascending=False).head(6)
    print("df_map_filtered_plus", df_map_filtered_plus)
    df_map_filtered_neg = df_map.sort_values(by='ecart', ascending=True).head(6)
    print("df_map_filtered_neg", df_map_filtered_neg)

    return df_map_filtered_plus, df_map_filtered_neg


def fonction_comparaison(df_map_filtered, df, ID, df_train):
    # global
    #df need to containon scaled info
    ID = int(ID)
    x = df[df['SK_ID_CURR'].isin([ID])]

    df_map_filtered['customer_values'] = [int(x[feature].values) for feature in
                                          df_map_filtered['feature'].values.tolist()]

    df_map_filtered['moy_global'] = [int(df[feature].mean()) for feature in df_map_filtered['feature'].values.tolist()]
    # clients en règle
    df_map_filtered['moy_en_regle'] = [int(df_train[df_train['TARGET'] == 0][feature].mean()) for feature in
                                       df_map_filtered['feature'].values.tolist()]
    # clients en règle
    df_map_filtered['moy_defaut'] = [int(df_train[df_train['TARGET'] == 1][feature].mean()) for feature in
                                     df_map_filtered['feature'].values.tolist()]

    return df_map_filtered


def df_chain_explain(df, cat):
    '''Ecrit une chaine de caractéres permettant d\'expliquer l\'influence des features dans le résultat de l\'algorithme '''
    if cat == 'positif':
        chaine = '### Principales caractéristiques contribuantes ###  \n'
    else:
        chaine = '### Principales caractéristiques discriminantes ###  \n'

    for feature in df['feature'].values:

        chaine += '### Caractéristique : ' + str(feature) + '###  \n'
        chaine += '* **Prospect : **' + str(df[df['feature'] == feature]['customer_values'].values[0])
        chaine_discrim = ' (seuil de pénalisation : ' + str(df[df['feature'] == feature]['signe'].values[0]) + ' '
        chaine_discrim += str(df[df['feature'] == feature]['val_lim'].values[0]) + ')'

        if cat == 'negatif':
            chaine += '<span style=\'color:red\'>' + chaine_discrim + '</span>  \n'
        else:
            chaine += '<span style=\'color:green\'>' + chaine_discrim + '</span>  \n'

    return chaine


def graphes_streamlit(df, cat):
    '''A partir du df, affichage un subplot de 6 graphes représentatif du client comparé à d'autres clients sur 6 features'''
    f, ax = plt.subplots(2, 3, figsize=(10, 10), sharex=False)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    i = 0
    j = 0
    liste_cols = ['Client', 'global', 'En Règle', 'En défaut']
    for feature in df['feature'].values:

        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(
            y=df[df['feature'] == feature][['customer_values', 'moy_global', 'moy_en_regle', 'moy_defaut']].values[0],
            x=liste_cols,
            ax=ax[i, j])
        sns.axes_style("white")

        if len(feature) >= 18:
            chaine = feature[:18] + '\n' + feature[18:]
        else:
            chaine = feature
        if cat == 'negatif':
            chaine += '\n(pénalise le score)'
            ax[i, j].set_facecolor('#ffe3e3')  # contribue négativement
            ax[i, j].set_title(chaine, color='#990024')
        else:
            chaine += '\n(améliore le score)'
            ax[i, j].set_facecolor('#e3ffec')
            ax[i, j].set_title(chaine, color='#017320')

        if j == 2:
            i += 1
            j = 0
        else:
            j += 1
        if i == 2:
            break
    for ax in f.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
    if i != 2:  # cas où on a pas assez de features à expliquer (ex : 445260)
        True
    st.pyplot()

    return True

