import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             r2_score, f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import statsmodels.api as sm
from scipy.spatial import ConvexHull
import matplotlib.ticker as mtick
import os
import warnings
from dmi_utils import hent_vejr

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Machine Learning Exam", layout="wide")

st.title("Machine Learning Exam - Football Matches")

# ==========================================
# STAGE 2: Data Preparation (KØRER USYNLIGT)
# ==========================================
rawdf = pd.read_csv("data/matchdataraw.csv")
cleaneddf = rawdf.copy()

cleaneddf.drop(columns=["XGPERSHOT"], inplace=True)

matches_to_remove = [1699867, 1699952, 2497370, 2720158, 2818322, 1699964]
cleaneddf['MATCH_WYID'] = cleaneddf['MATCH_WYID'].astype(int)
cleaneddf = cleaneddf[~cleaneddf['MATCH_WYID'].isin(matches_to_remove)]

cleaneddf['DATE'] = pd.to_datetime(cleaneddf['DATE'])
cleaneddf['DATEUTC'] = pd.to_datetime(cleaneddf['DATEUTC'])
cleaneddf['weekday'] = cleaneddf['DATE'].dt.day_name() 

team_mapping = {
    7458: 'Nordsjælland', 7499: 'SønderjyskE', 7456: 'Viborg', 
    7455: 'Midtjylland', 7460: 'OB', 7510: 'Hobro', 
    7452: 'København', 7462: 'Randers', 7453: 'Brøndby', 
    7457: 'AGF', 7454: 'AaB', 7451: 'Esbjerg', 
    7484: 'Lyngby', 7461: 'Silkeborg', 7465: 'Horsens', 
    7566: 'FC Helsingør', 7488: 'Vendsyssel', 7473: 'Vejle', 
    7490: 'Hvidovre', 7469: 'Fredericia'
}

cleaneddf['TEAM_WYID'] = pd.to_numeric(cleaneddf['TEAM_WYID'], errors='coerce').fillna(0).astype(int)
cleaneddf['Hold'] = cleaneddf['TEAM_WYID'].map(team_mapping)
cleaneddf['Total Cards'] = pd.to_numeric(cleaneddf['REDCARDS'], errors='coerce').fillna(0) + \
                           pd.to_numeric(cleaneddf['YELLOWCARDS'], errors='coerce').fillna(0)

if 'MATCHLABEL' in cleaneddf.columns:
    matchlabel_data = cleaneddf.pop('MATCHLABEL')
    cleaneddf['MATCHLABEL'] = matchlabel_data

cols_to_sum = {
    'SHOTS': 'Total Shots', 'FOULS': 'Total Fouls', 'CORNERS': 'Total Corners',
    'REDCARDS': 'Total Red cards', 'YELLOWCARDS': 'Total Yellow cards', 'OFFSIDES': 'Total Off-side',
    'DRIBBLES': 'Total Dribbles', 'GOALS': 'Total Goals', 'PROGRESSIVERUNS': 'Total Progressive runs',
    'FREEKICKS': 'Total Free kicks', 'TOTALTHROWINS': 'Total Throw ins'
}

match_df = cleaneddf.groupby(['MATCH_WYID', 'MATCHLABEL']).agg({
    **{col: 'sum' for col in cols_to_sum.keys()}
}).reset_index()

match_df.rename(columns=cols_to_sum, inplace=True)
match_df['Total Cards'] = match_df['Total Red cards'] + match_df['Total Yellow cards']

cleaneddf = cleaneddf.replace("SønderjyskE", "Sønderjyske")
match_df = match_df.replace("SønderjyskE", "Sønderjyske")

match_df[['Teams', 'Score']] = match_df['MATCHLABEL'].str.rsplit(', ', n=1, expand=True)
match_df[['Home Team', 'Away Team']] = match_df['Teams'].str.split(' - ', expand=True)
match_df[['Home Goals', 'Away Goals']] = match_df['Score'].str.split('-', expand=True)

match_df["Home Goals"] = match_df["Home Goals"].astype(str).str.extract(r'(\d+)').astype(int)
match_df["Away Goals"] = match_df["Away Goals"].astype(str).str.extract(r'(\d+)').astype(int)

cols = [
    'MATCH_WYID','MATCHLABEL','Teams','Score','Home Team','Away Team','Home Goals','Away Goals',
    'Total Shots','Total Fouls','Total Corners','Total Red cards','Total Yellow cards',
    'Total Off-side','Total Dribbles','Total Goals','Total Progressive runs',
    'Total Free kicks','Total Throw ins','Total Cards'
]
match_df = match_df[cols]

cleaneddf = cleaneddf.sort_values("MATCH_WYID")
cleaneddf["ROW"] = cleaneddf.groupby("MATCH_WYID").cumcount()
cleaneddf["VENUE"] = cleaneddf["ROW"].map({0: "Home", 1: "Away"})

# ==========================================
# NAVIGATION / TABS
# ==========================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Oversigt",
    "Vejranalyse", 
    "Hjemmebanefordel & Spilstrategi", 
    "Disciplin & Kampresultater", 
    "Samlet Konklusion"
])

# ==========================================
# TAB 0: Landing Page
# ==========================================
with tab0:
    st.header("Velkommen til Fodbold Data Dashboardet")
    
    st.write("""
    Dette dashboard er udviklet som en del af et Business Intelligence og Machine Learning eksamensprojekt. 
    Formålet med projektet er at undersøge og præsentere de faktorer, der har størst indflydelse på 
    udfaldet af fodboldkampe. Vi anvender både statistiske metoder og maskinlæring (herunder regression, 
    klassificering og clustering) for at skabe en dybere forståelse af kampdataen.
    """)

    st.subheader("How to use this dashboard")
    st.write("""
    * **Brug fanerne (tabs) øverst:** Skærmen er opdelt i overordnede analytiske faner.
    * **Udforsk hypoteser:** Hver fane repræsenterer et bestemt analysemne og indeholder underliggende hypoteser.
    * **Visualiseringer:** Under hver hypotese finder du relevante grafer (boksplots, heatmaps, scatterplots osv.) og konklusioner, der be- eller afkræfter hypotesen.
    """)

    st.subheader("Categories (Stage 1: Problem Foundation)")
    st.markdown("""
    Analysen er opdelt i tre primære undersøgelseskategorier samt en samlet konklusion:

    **1. Vejranalyse**
    * *Undersøger relationen mellem regnvejr og kampens udspil.*
    * Hypoteser: Scorer holdene flere mål i regnvejr? Bliver der uddelt flere kort eller lavet flere frispark?

    **2. Hjemmebanefordel & Spilstrategi**
    * *Undersøger fordelen ved at spille på hjemmebane og holdenes skududnyttelse.*
    * Hypoteser: Vinder hjemmeholdet oftere? Får udeholdet flere kort pr. foul? Fører en højere skudprocent direkte til flere mål?

    **3. Disciplin & Kampresultater**
    * *Undersøger relationen mellem frispark, kort, hjørnespark og chancen for at vinde.*
    * Hypoteser: Har antallet af kort eller hjørnespark en direkte effekt på kampens resultat? Kan taktiske frispark betale sig?
    
    **4. Samlet Konklusion**
    * *Afslutning og opsummering af de vigtigste fund på tværs af alle modeller.*
    """)

# ==========================================
# TAB 1: Vejranalyse
# ==========================================
with tab1:
    st.header("Vejranalyse")
    st.markdown("""
    ### Hypoteser: Der er en relation mellem regnvejr og en kamps udspil
    * Der scores flere mål i regnvejrskampe
    * Der bliver uddelt flere kort i regnvejrskampe
    * Der bliver generelt lavet flere fouls i regnvejrskampe
    * Der bliver dømt flere frispark i regnvejrskampe
    """)

    TEAM_MUNICIPALITY = {
        "København": "0101", "Brøndby": "0153", "Nordsjælland": "0217", "Lyngby": "0173",
        "FC Helsingør": "0217", "Hvidovre": "0167", "OB": "0461", "Esbjerg": "0561",
        "SønderjyskE": "0540", "Sønderjyske": "0540", "AaB": "0851", "Vendsyssel": "0813",
        "Hobro": "0846", "Midtjylland": "0657", "AGF": "0751", "Viborg": "0791",
        "Horsens": "0615", "Fredericia": "0607", "Vejle": "0630", "Randers": "0730", "Silkeborg": "0740"
    }

    dato_map = cleaneddf.groupby('MATCH_WYID').agg(DATE=('DATE', 'first'), MATCHLABEL=('MATCHLABEL', 'first')).reset_index()
    dato_map['home_team'] = dato_map['MATCHLABEL'].str.split(',').str[0].str.split(' - ').str[0].str.strip()
    dato_map['municipality_id'] = dato_map['home_team'].map(TEAM_MUNICIPALITY)
    dato_map['dato_str'] = dato_map['DATE'].dt.strftime('%Y-%m-%d')

    unikke = dato_map[dato_map['municipality_id'].notna()][['dato_str', 'municipality_id']].drop_duplicates().reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    CACHE_PATH = "data/vejr_cache.csv"
    vejr_df = hent_vejr(unikke, cache_path=CACHE_PATH)

    match_df_vejr = match_df.copy()
    dato_map['municipality_id'] = dato_map['municipality_id'].astype(str).str.zfill(4)
    vejr_df['municipality_id'] = vejr_df['municipality_id'].astype(str).str.zfill(4)

    dato_map2 = dato_map[['MATCH_WYID', 'dato_str', 'municipality_id']].merge(vejr_df, on=['dato_str', 'municipality_id'], how='left')
    match_df_vejr = match_df_vejr.merge(dato_map2[['MATCH_WYID', 'nedbor_mm']], on='MATCH_WYID', how='left')
    match_df_vejr['Total Regn'] = match_df_vejr['nedbor_mm']
    match_df_vejr['Regnede'] = match_df_vejr['Total Regn'].apply(lambda x: pd.notna(x) and x > 0)
    match_df_vejr.drop(columns=['nedbor_mm'], inplace=True)

    st.write("**Descriptive statistics:**")
    st.dataframe(match_df_vejr.groupby('Regnede')[['Total Goals', 'Total Yellow cards', 'Total Red cards', 'Total Fouls', 'Total Free kicks']].mean().round(2))

    # H1
    st.subheader("H1: Der scores flere mål i regnvejrskampe")
    fig, ax = plt.subplots(figsize=(6, 4))
    match_df_vejr.boxplot(column='Total Goals', by='Regnede', ax=ax)
    plt.title('H1: — er der flere mål i regnvejr')
    plt.suptitle('')
    plt.xlabel('Regnede')
    plt.ylabel('Antal mål')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    regn = match_df_vejr[match_df_vejr['Regnede'] == True]['Total Goals']
    torr = match_df_vejr[match_df_vejr['Regnede'] == False]['Total Goals']
    t, p = stats.ttest_ind(regn, torr)
    st.write(f"Gennemsnit regn: {regn.mean():.2f} mål | Gennemsnit tørvejr: {torr.mean():.2f} mål")
    st.write(f"T-test p-værdi: **{p:.4f}** (Konklusion: {'SIGNIFIKANT forskel' if p < 0.05 else 'INGEN signifikant forskel'})")

    # H2
    st.subheader("H2: Der bliver uddelt flere kort i regnvejrskampe")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    match_df_vejr.boxplot(column='Total Yellow cards', by='Regnede', ax=ax)
    plt.title('H2: Flere gule kort i regnvejr?')
    plt.suptitle('')
    plt.xlabel('Regnede')
    plt.ylabel('Antal gule kort')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    match_df_vejr.boxplot(column='Total Red cards', by='Regnede', ax=ax)
    plt.title('H2: Flere røde kort i regnvejr?')
    plt.suptitle('')
    plt.xlabel('Regnede')
    plt.ylabel('Antal røde kort')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    regn_gule = match_df_vejr[match_df_vejr['Regnede'] == True]['Total Yellow cards']
    torr_gule = match_df_vejr[match_df_vejr['Regnede'] == False]['Total Yellow cards']
    t, p_gule = stats.ttest_ind(regn_gule, torr_gule)
    regn_rode = match_df_vejr[match_df_vejr['Regnede'] == True]['Total Red cards']
    torr_rode = match_df_vejr[match_df_vejr['Regnede'] == False]['Total Red cards']
    t, p_rode = stats.ttest_ind(regn_rode, torr_rode)
    st.write(f"Gule kort p-værdi: **{p_gule:.4f}** | Røde kort p-værdi: **{p_rode:.4f}**")

    # H3
    st.subheader("H3: Der bliver generelt lavet flere fouls i regnvejrskampe")
    fig, ax = plt.subplots(figsize=(6, 4))
    match_df_vejr.boxplot(column='Total Fouls', by='Regnede', ax=ax)
    plt.title('H3: Flere frispark i regnvejr?')
    plt.suptitle('')
    plt.xlabel('Regnede')
    plt.ylabel('Antal frispark')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    
    regn_fouls = match_df_vejr[match_df_vejr['Regnede'] == True]['Total Fouls']
    torr_fouls = match_df_vejr[match_df_vejr['Regnede'] == False]['Total Fouls']
    t, p_fouls = stats.ttest_ind(regn_fouls, torr_fouls)
    st.write(f"Fouls p-værdi: **{p_fouls:.4f}**")

    # H4
    st.subheader("H4: Der bliver dømt flere frispark i regnvejrskampe")
    fig, ax = plt.subplots(figsize=(6, 4))
    match_df_vejr.boxplot(column='Total Free kicks', by='Regnede', ax=ax)
    plt.title('H4: Flere frispark i regnvejr?')
    plt.suptitle('')
    plt.xlabel('Regnede')
    plt.ylabel('Antal frispark')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    
    regn_frispark = match_df_vejr[match_df_vejr['Regnede'] == True]['Total Free kicks']
    torr_frispark = match_df_vejr[match_df_vejr['Regnede'] == False]['Total Free kicks']
    t, p_frispark = stats.ttest_ind(regn_frispark, torr_frispark)
    st.write(f"Frispark p-værdi: **{p_frispark:.4f}**")

    # CM
    st.subheader("Confusion Matrix: Logistisk regression (Beskidt kamp vs Regn)")
    match_df_vejr['Beskidthedsværdi'] = (match_df_vejr['Total Fouls'] * 0.5 + match_df_vejr['Total Yellow cards'] * 2 + match_df_vejr['Total Red cards'] * 4 + match_df_vejr['Total Free kicks'])
    graense = match_df_vejr['Beskidthedsværdi'].median()
    match_df_vejr['BeskidtKamp'] = (match_df_vejr['Beskidthedsværdi'] > graense).astype(int)

    X = match_df_vejr[['Total Regn']].fillna(0)
    y = match_df_vejr['BeskidtKamp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ikke beskidt', 'Beskidt'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix — Regn som forklaring')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.4f}**")

    # PCA Clustering
    st.subheader("Klyngedannelse (Clustering) med vejrdata (PCA)")
    team_df = cleaneddf.copy()
    cols_to_drop = [c for c in ['Regnede', 'Regnede_x', 'Regnede_y'] if c in team_df.columns]
    team_df = team_df.drop(columns=cols_to_drop)
    team_df = team_df.merge(match_df_vejr[['MATCH_WYID', 'Regnede']], on='MATCH_WYID', how='left')
    team_df['Regnede'] = team_df['Regnede'].apply(lambda x: 1 if x is True or x == 1 else 0)

    features = ['SHOTS', 'FOULS', 'CORNERS', 'YELLOWCARDS', 'REDCARDS', 'GOALS', 'Regnede']
    df_analysis = team_df[features].copy().dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_analysis)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_analysis['Cluster'] = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_features)
    df_analysis['PC1'] = pca_components[:, 0]
    df_analysis['PC2'] = pca_components[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = sns.scatterplot(data=df_analysis, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.1, ax=ax)
    sns.scatterplot(data=df_analysis[df_analysis['Regnede'] == 0], x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.5, marker='o', s=50, legend=False, ax=ax)
    sns.scatterplot(data=df_analysis[df_analysis['Regnede'] == 1], x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.7, marker='X', s=100, edgecolor='black', legend=False, ax=ax)

    dry_circle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='Tørvejr')
    rain_cross = mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='Regnvejr')

    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles=handles + [dry_circle, rain_cross], title='Cluster & Vejrtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('PCA-visualisering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Heatmap
    st.subheader("Heatmap")
    heatmap_df = match_df_vejr[['Total Regn', 'Total Goals', 'Total Yellow cards', 'Total Red cards', 'Total Fouls', 'Total Free kicks']].copy()
    corr = heatmap_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title('Korrelation mellem regn og statistik')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ==========================================
# TAB 2: Hjemmebanefordel & Spilstrategi
# ==========================================
with tab2:
    st.header("Hjemmebanefordel & Spilstrategi")
    st.markdown("""
    ### Hypoteser
    * Man vinder oftere med hjemmebanefordel
    * Hjemmebanehold får færre “fouls” stemt imod sig end udebanehold
    * Udebanehold har flere kort pr. foul end hjemmebanehold
    * Højere skud procent fører til flere mål
    """)

    # Result distribution
    st.subheader("Match Results Distribution")
    match_df["Result"] = np.select(
        [match_df["Home Goals"] > match_df["Away Goals"], match_df["Home Goals"] < match_df["Away Goals"]],
        ["Home Win", "Away Win"], default="Draw"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    result_pct = match_df["Result"].value_counts(normalize=True) * 100
    result_pct.plot(kind="bar", ax=ax)
    plt.title("Match Results Distribution (%)")
    plt.xlabel("Result")
    plt.ylabel("Percentage of Matches")
    for i, v in enumerate(result_pct):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Fouls Share
    st.subheader("Fouls: Hjemme vs Ude")
    fouls_pct = cleaneddf.groupby("VENUE")["FOULS"].sum()
    fouls_pct = fouls_pct / fouls_pct.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    fouls_pct.plot(kind="bar", ax=ax)
    plt.title("Share of Fouls by Venue (%)")
    plt.xlabel("Venue")
    plt.ylabel("Percentage of Fouls (%)")
    for i, v in enumerate(fouls_pct):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    plt.ylim(0,100)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    home_fouls = team_df[team_df["VENUE"] == "Home"]["FOULS"].mean()
    away_fouls = team_df[team_df["VENUE"] == "Away"]["FOULS"].mean()
    st.write(f"Average fouls home team: **{home_fouls:.2f}** | Average fouls away team: **{away_fouls:.2f}**")

    # Cards per foul
    st.subheader("Kort pr. Foul: Hjemme vs Ude")
    cleaneddf["TOTAL_CARDS"] = cleaneddf["YELLOWCARDS"] + cleaneddf["REDCARDS"]
    cleaneddf["CARDS_PER_FOUL"] = cleaneddf["TOTAL_CARDS"] / cleaneddf["FOULS"].replace(0,1)
    cards_pct = cleaneddf.groupby("VENUE")["CARDS_PER_FOUL"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    cards_pct.plot(kind="bar", ax=ax)
    plt.title("Cards per Foul by Venue (%)")
    plt.xlabel("Venue")
    plt.ylabel("Cards per Foul (%)")
    for i, v in enumerate(cards_pct):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Shot percentage
    st.subheader("Spilstrategi: Skudprocent vs. Mål")
    team_df["SHOT_PERCENTAGE"] = team_df["SHOTSONTARGET"] / team_df["SHOTS"].replace(0,1)
    team_df = team_df.sort_values("MATCH_WYID")
    team_df["OPP_GOALS"] = team_df.groupby("MATCH_WYID")["GOALS"].transform(lambda x: x.iloc[::-1].values)

    features = ["SHOT_PERCENTAGE", "SHOTS", "XG"]
    X = team_df[features]
    y = team_df["GOALS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=team_df["SHOT_PERCENTAGE"] * 100, y=team_df["GOALS"], ax=ax)
    plt.xlabel("Shot Percentage (%)")
    plt.ylabel("Goals")
    plt.title("Relationship Between Shot Percentage and Goals")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    st.write(f"Regression R2 score: **{r2_score(y_test, pred):.4f}**")
    st.write(f"Coefficients: {model.coef_}")

    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    st.text(str(model_sm.summary()))

    # Random Forest
    st.subheader("Random Forest Feature Importance")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

    importance = rf_model.feature_importances_ * 100
    feature_importance = pd.Series(importance, index=features)

    fig, ax = plt.subplots(figsize=(6, 4))
    feature_importance.sort_values().plot(kind="barh", ax=ax)
    plt.title("Random Forest Feature Importance (%)")
    plt.xlabel("Importance (%)")
    for i, v in enumerate(feature_importance.sort_values()):
        ax.text(v + 0.2, i, f"{v:.1f}%")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    st.write(f"Random Forest R2 score: **{r2_score(y_test, pred_rf):.4f}**")

    # Classification Models
    st.subheader("Classification Models (Forudsig Sejr)")
    team_df["WIN"] = (team_df["GOALS"] > team_df["OPP_GOALS"]).astype(int)
    y_win = team_df["WIN"]

    features_xg = ["SHOT_PERCENTAGE", "SHOTSONTARGET", "SHOTS", "XG", "FOULS", "CORNERS", "FREEKICKS"]
    features_rain = ["SHOT_PERCENTAGE", "SHOTSONTARGET", "SHOTS", "Regnede", "FOULS", "CORNERS", "FREEKICKS"]
    features_base = ["SHOT_PERCENTAGE", "SHOTSONTARGET", "SHOTS", "FOULS", "CORNERS", "FREEKICKS"]

    train_idx, test_idx = train_test_split(team_df.index, test_size=0.2, random_state=42)
    X_train_a, X_test_a = team_df[features_xg].loc[train_idx], team_df[features_xg].loc[test_idx]
    X_train_b, X_test_b = team_df[features_rain].loc[train_idx], team_df[features_rain].loc[test_idx]
    X_train_c, X_test_c = team_df[features_base].loc[train_idx], team_df[features_base].loc[test_idx]
    y_train_win, y_test_win = y_win.loc[train_idx], y_win.loc[test_idx]

    clf_a = LogisticRegression(max_iter=2000).fit(X_train_a, y_train_win)
    clf_b = LogisticRegression(max_iter=2000).fit(X_train_b, y_train_win)
    clf_c = LogisticRegression(max_iter=2000).fit(X_train_c, y_train_win)

    pred_a, pred_b, pred_c = clf_a.predict(X_test_a), clf_b.predict(X_test_b), clf_c.predict(X_test_c)
    cm_a, cm_b, cm_c = confusion_matrix(y_test_win, pred_a), confusion_matrix(y_test_win, pred_b), confusion_matrix(y_test_win, pred_c)

    # CM A
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Reds", xticklabels=["Tabt", "Vundet"], yticklabels=["Tabt", "Vundet"], ax=ax)
    plt.title("Model med XG")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
        
    st.write("**MODEL A - XG**")
    st.write(f"Accuracy: {accuracy_score(y_test_win, pred_a):.4f} | F1: {f1_score(y_test_win, pred_a):.4f} | Precision: {precision_score(y_test_win, pred_a):.4f} | Recall: {recall_score(y_test_win, pred_a):.4f}")

    # CM B
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Blues", xticklabels=["Tabt", "Vundet"], yticklabels=["Tabt", "Vundet"], ax=ax)
    plt.title("Model med Regn (uden XG)")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
        
    st.write("**MODEL B - RAIN**")
    st.write(f"Accuracy: {accuracy_score(y_test_win, pred_b):.4f} | F1: {f1_score(y_test_win, pred_b):.4f} | Precision: {precision_score(y_test_win, pred_b):.4f} | Recall: {recall_score(y_test_win, pred_b):.4f}")

    # CM C
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_c, annot=True, fmt="d", cmap="Greens", xticklabels=["Tabt", "Vundet"], yticklabels=["Tabt", "Vundet"], ax=ax)
    plt.title("Basismodel uden XG og Regn")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
        
    st.write("**MODEL C - BASE**")
    st.write(f"Accuracy: {accuracy_score(y_test_win, pred_c):.4f} | F1: {f1_score(y_test_win, pred_c):.4f} | Precision: {precision_score(y_test_win, pred_c):.4f} | Recall: {recall_score(y_test_win, pred_c):.4f}")

    # Coefficients A
    coef_a = pd.Series(clf_a.coef_[0], index=features_xg).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    coef_a.plot(kind="barh", ax=ax)
    plt.title("Coefficients - XG")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Coefficients B
    coef_b = pd.Series(clf_b.coef_[0], index=features_rain).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    coef_b.plot(kind="barh", ax=ax)
    plt.title("Coefficients - Rain")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Coefficients C
    coef_c = pd.Series(clf_c.coef_[0], index=features_base).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    coef_c.plot(kind="barh", ax=ax)
    plt.title("Coefficients - Base")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Clustering Shots vs Goals
    st.subheader("Clustering: Hold efter spillestil (Skud vs Mål)")
    cluster_features = ["SHOTS", "SHOTSONTARGET", "SHOT_PERCENTAGE", "XG", "FOULS", "CORNERS"]
    X_cluster = team_df[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=3, random_state=42)
    team_df["Cluster"] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    team_df['SHOTS_plot'] = team_df['SHOTS'] + np.random.uniform(-0.6, 0.6, len(team_df))
    team_df['GOALS_plot'] = team_df['GOALS'] + np.random.uniform(-0.4, 0.4, len(team_df))
    colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}

    sns.scatterplot(data=team_df, x='SHOTS_plot', y='GOALS_plot', hue='Cluster', palette=colors, alpha=0.5, s=40, ax=ax)

    for cluster_id in team_df['Cluster'].unique():
        points = team_df[team_df['Cluster'] == cluster_id][['SHOTS_plot', 'GOALS_plot']].values
        if len(points) >= 3:
            hull = ConvexHull(points)
            ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.2, color=colors.get(cluster_id, 'gray'), edgecolor=colors.get(cluster_id, 'gray'), lw=2)

    cards_avg = team_df.groupby('Cluster')['SHOTS'].mean().sort_values().reset_index()
    cluster_labels = {
        cards_avg.iloc[0]['Cluster']: 'Defensivt',
        cards_avg.iloc[1]['Cluster']: 'Moderat',
        cards_avg.iloc[2]['Cluster']: 'Aggressivt'
    }
    legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', label=cluster_labels[c], markerfacecolor=colors[c], markersize=8) for c in sorted(colors.keys())]
    ax.legend(handles=legend_elements, title='Spillestil', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("Shots")
    plt.ylabel("Goals")
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ==========================================
# TAB 3: Disciplin & Kampresultater
# ==========================================
with tab3:
    st.header("Disciplin & Kampresultater")
    st.markdown("""
    ### Hypoteser
    * Mængden af kort et hold modtager har en korrelation til kampens resultat
    * Mængden af hjørnespark har en korrelation til at vinde kampe
    * Man vinder flere kampe, jo flere frispark et hold får dømt til deres fordel
    """)

    # Clustering Fouls vs Cards
    st.subheader("Clustering: Fouls vs Kort")
    cardsdf_cluster = cleaneddf[['TOTAL_CARDS', 'FOULS']].copy().dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(cardsdf_cluster)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cardsdf_cluster['Gruppe_Nummer'] = kmeans.fit_predict(data_scaled)

    cards_avg = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['TOTAL_CARDS', 'FOULS'])
    cards_avg['Gruppe_Nummer'] = [0, 1, 2]
    cards_avg = cards_avg.sort_values(by=['TOTAL_CARDS'])

    navne = {
        cards_avg.iloc[0]['Gruppe_Nummer']: 'Fairplay',
        cards_avg.iloc[1]['Gruppe_Nummer']: 'Taktisk',
        cards_avg.iloc[2]['Gruppe_Nummer']: 'Grov Kamp'
    }
    cardsdf_cluster['Spillestil'] = cardsdf_cluster['Gruppe_Nummer'].map(navne)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    cardsdf_cluster['FOULS_plot'] = cardsdf_cluster['FOULS'] + np.random.uniform(-0.6, 0.6, len(cardsdf_cluster))
    cardsdf_cluster['Cards_plot'] = cardsdf_cluster['TOTAL_CARDS'] + np.random.uniform(-0.4, 0.4, len(cardsdf_cluster))
    farver = {'Fairplay': '#4daf4a', 'Taktisk': '#377eb8', 'Grov Kamp': '#e41a1c'}

    sns.scatterplot(data=cardsdf_cluster, x='FOULS_plot', y='Cards_plot', hue='Spillestil', palette=farver, alpha=0.5, s=40, ax=ax)

    for gruppe_navn in cardsdf_cluster['Spillestil'].unique():
        prikker = cardsdf_cluster[cardsdf_cluster['Spillestil'] == gruppe_navn][['FOULS_plot', 'Cards_plot']].values
        if len(prikker) >= 3:
            hull = ConvexHull(prikker)
            ax.fill(prikker[hull.vertices, 0], prikker[hull.vertices, 1], alpha=0.2, color=farver[gruppe_navn], edgecolor=farver[gruppe_navn], lw=2)

    plt.title('Klynger af Hold')
    plt.xlabel('Antal Frispark (FOULS)')
    plt.ylabel('Antal Kort (Total Cards)')
    plt.legend(title='Spillestil', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Stacked Bar Chart (Kampudfald pr. Spillestil)
    st.subheader("Kampudfald pr. Spillestil")
    cardsdf_copy = cleaneddf.copy()
    cardsdf_copy['Total Cards'] = cardsdf_copy['YELLOWCARDS'] + cardsdf_copy['REDCARDS']

    def find_udfald(kamp):
        if len(kamp) == 2:
            maal = kamp['GOALS'].values
            if maal[0] > maal[1]:
                kamp['Resultat'] = ['Sejr', 'Nederlag']
            elif maal[0] < maal[1]:
                kamp['Resultat'] = ['Nederlag', 'Sejr']
            else:
                kamp['Resultat'] = ['Uafgjort', 'Uafgjort']
        else:
            kamp['Resultat'] = np.nan
        return kamp

    df_udfald = cardsdf_copy.groupby('MATCH_WYID', group_keys=False).apply(find_udfald).dropna(subset=['Resultat'])
    cardsdf_cluster['Resultat'] = df_udfald['Resultat']

    fordeling = cardsdf_cluster.groupby(['Spillestil', 'Resultat']).size().unstack(fill_value=0)
    fordeling_pct = fordeling.div(fordeling.sum(axis=1), axis=0)
    fordeling_pct = fordeling_pct[['Sejr', 'Uafgjort', 'Nederlag']]

    fig, ax = plt.subplots(figsize=(6, 4))
    farver_resultat = ['#4daf4a', '#999999', '#e41a1c']
    fordeling_pct.plot(kind='bar', stacked=True, color=farver_resultat, ax=ax, edgecolor='white', width=0.7)

    plt.title('Kampudfald pr. Spillestil')
    plt.xlabel('Spillestil (Cluster)')
    plt.ylabel('Procentdel af kampe')
    plt.xticks(rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for c in ax.containers:
        labels = [f'{v.get_height()*100:.1f}%' if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)
    plt.legend(title='Kampens Udfald', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Logistic Regression (Card Diff vs Win)
    st.subheader("Supervised Learning: Vinderchance vs. Kort-forskel")
    def klargor_regression(kamp):
        if len(kamp) == 2:
            maal = kamp['GOALS'].values
            if maal[0] > maal[1]:
                kamp['Vandt'] = [1, 0]
            elif maal[0] < maal[1]:
                kamp['Vandt'] = [0, 1]
            else:
                kamp['Vandt'] = [0, 0]
            kort = kamp['Total Cards'].values
            kamp['Card_Diff'] = [kort[0] - kort[1], kort[1] - kort[0]]
        else:
            kamp['Vandt'] = np.nan
            kamp['Card_Diff'] = np.nan
        return kamp

    cardsdf_reg = cardsdf_copy.groupby('MATCH_WYID', group_keys=False).apply(klargor_regression).dropna(subset=['Vandt', 'Card_Diff'])
    cardsdf_reg['Card_Diff'] = cardsdf_reg['Card_Diff'].astype(float)
    cardsdf_reg['Vandt'] = cardsdf_reg['Vandt'].astype(int)

    X = cardsdf_reg[['Card_Diff']]
    y = cardsdf_reg['Vandt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=cardsdf_reg, x='Card_Diff', y='Vandt', logistic=True, ci=None, y_jitter=0.04, scatter_kws={'alpha': 0.1, 'color': 'black'}, line_kws={'color': '#e41a1c', 'linewidth': 3}, ax=ax)
    plt.title('Vinderchance vs. Kort-forskel')
    plt.xlabel('Kort-forskel (Egne kort minus modstanderens)')
    plt.ylabel('Sandsynlighed for Sejr')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-5, 5)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Random Forest Corners
    st.subheader("Random Forest Prædiktion af Sejr (Hjørnespark mv.)")
    cols_to_drop = ['OPPONENT_TEAM_WYID', 'OPPONENT_GOALS', 'WIN']
    df_safe = cleaneddf.drop(columns=[col for col in cols_to_drop if col in cleaneddf.columns])
    opp_df = df_safe[['MATCH_WYID', 'TEAM_WYID', 'GOALS']].copy()
    opp_df.columns = ['MATCH_WYID', 'OPPONENT_TEAM_WYID', 'OPPONENT_GOALS']
    df_merged = pd.merge(df_safe, opp_df, on='MATCH_WYID')
    df_merged = df_merged[df_merged['TEAM_WYID'] != df_merged['OPPONENT_TEAM_WYID']]
    df_merged['WIN'] = (df_merged['GOALS'] > df_merged['OPPONENT_GOALS']).astype(int)

    features = ['CORNERS', 'XG', 'TOUCHESINBOX', 'SHOTS']
    X = df_merged[features].dropna()
    y = df_merged.loc[X.index, 'WIN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pointtab', 'Sejr'])
    disp.plot(cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relativ Vigtighed')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    # Convex Hull Corners vs xG
    st.subheader("Taktiske Profiler (Hjørnespark vs xG)")
    cluster_features = ['CORNERS', 'XG']
    df_merged_clean = df_merged.dropna(subset=cluster_features).copy()
    X_cluster = df_merged_clean[cluster_features]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_merged_clean['Cluster'] = kmeans.fit_predict(X_cluster)

    centroids = df_merged_clean.groupby('Cluster')['CORNERS'].mean().sort_values()
    cluster_mapping = {
        centroids.index[0]: 'Afventende',
        centroids.index[1]: 'Effektiv',
        centroids.index[2]: 'Tung Dominans'
    }
    df_merged_clean['Taktisk Profil'] = df_merged_clean['Cluster'].map(cluster_mapping)

    farver = {
        'Afventende': '#4daf4a', 
        'Effektiv': '#377eb8', 
        'Tung Dominans': '#e41a1c'
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_merged_clean, x='CORNERS', y='XG', hue='Taktisk Profil', palette=farver, alpha=0.5, s=40, ax=ax)

    for gruppe_navn in df_merged_clean['Taktisk Profil'].unique():
        prikker = df_merged_clean[df_merged_clean['Taktisk Profil'] == gruppe_navn][['CORNERS', 'XG']].values
        if len(prikker) >= 3:
            hull = ConvexHull(prikker)
            ax.fill(prikker[hull.vertices, 0], prikker[hull.vertices, 1], alpha=0.2, color=farver[gruppe_navn], edgecolor=farver[gruppe_navn], lw=2)

    plt.title('Taktiske Profiler (Hjørnespark vs xG)')
    plt.xlabel('Antal Hjørnespark (CORNERS)')
    plt.ylabel('Expected Goals (XG)')
    plt.legend(title='Taktisk Profil', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ==========================================
# TAB 4: Samlet Konklusion
# ==========================================
with tab4:
    st.header("Samlet Afsluttende Konklusion")
    st.markdown("""
    * Denne opgave har med både statistiske analyser og machine learning undersøgt, hvilke faktorer der har størst betydning for udfaldet af fodboldkampe. På tværs af analyserne ses det tydeligt, at de mest afgørende variable er relateret til produktion af chancer og offensiv kvalitet, såsom xG, skud og effektivitet foran mål. Desuden viser analysen, at holdets disciplin spiller en væsentlig rolle, da flere disciplinære sanktioner i form af kort mindsker vinderchancerne. Samtidig viser resultaterne, at faktorer som vejr og antallet hjørnespark ikke i sig selv har en stærk eller afgørende effekt på kampens udfald.

    * Vores analyser peger derfor på, at det ikke er de ydre forhold alene, men derimod holdenes evne til at skabe og udnytte chancer, der bedst forklarer forskellen mellem sejr og nederlag. De supervised learning-modeller, vi anvendte, viste især, at nogle variable har langt større forklaringskraft end andre, mens de unsupervised modeller gav et visuelt og taktisk overblik over forskellige mønstre i dataen. Dette gjorde det muligt både at teste konkrete hypoteser og samtidig opdage underliggende strukturer i kampdataen.

    * Samlet set viser opgaven, at machine learning er et nyttigt redskab til at analysere sportsdata, fordi metoderne både kan bruges til forudsigelse, forklaring og mønstergenkendelse. Resultaterne understreger samtidig, at fodbold er et komplekst spil, hvor enkelte variable sjældent kan stå alene, men hvor flere faktorer tilsammen skaber det samlede kampbillede.
    """)