import requests
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

# Kommune-ID for hvert hjemmehold
TEAM_MUNICIPALITY = {
    "København":    "0101",
    "Brøndby":      "0153",
    "Nordsjælland": "0217",
    "Lyngby":       "0173",
    "FC Helsingør": "0217",
    "Hvidovre":     "0167",
    "OB":           "0461",
    "Esbjerg":      "0561",
    "SønderjyskE":  "0540",
    "Sønderjyske":  "0540",
    "AaB":          "0851",
    "Vendsyssel":   "0813",
    "Hobro":        "0846",
    "Midtjylland":  "0657",
    "AGF":          "0751",
    "Viborg":       "0791",
    "Horsens":      "0615",
    "Fredericia":   "0607",
    "Vejle":        "0630",
    "Randers":      "0730",
    "Silkeborg":    "0740",
}

def hent_vejr(unikke, cache_path="data/vejr_cache.csv"):
    """
    Henter nedbørsdata fra DMI municipalityValue endpoint.
    Meget bedre dækning end stationsdata da DMI interpolerer til alle kommuner.
    Kun tilgængeligt for Danmark fra 2011 og frem.

    Args:
        unikke:     DataFrame med kolonner 'dato_str' og 'municipality_id'
        cache_path: Sti til cache CSV-fil (gemmer så du ikke henter igen)

    Returns:
        DataFrame med kolonner: dato_str, municipality_id, nedbor_mm
    """

    if os.path.exists(cache_path):
        print(f"Cache fundet — loader fra {cache_path}")
        df = pd.read_csv(cache_path, dtype={"municipality_id": str})
        df["municipality_id"] = df["municipality_id"].str.zfill(4)
        return df

    print(f"Henter {len(unikke)} kombinationer fra DMI...")

    def hent_en(row):
        try:
            r = requests.get(
                "https://opendataapi.dmi.dk/v2/climateData/collections/municipalityValue/items",
                params={
                    "parameterId":    "acc_precip",
                    "timeResolution": "day",
                    "datetime":       f"{row['dato_str']}T00:00:00Z/{row['dato_str']}T23:59:59Z",
                    "municipalityId": row["municipality_id"],
                    "limit":          1,
                },
                timeout=10
            )
            features = r.json().get("features", []) if r.status_code == 200 else []
            mm = round(float(features[0]["properties"]["value"]), 1) if features else None
        except Exception:
            mm = None
        return {"dato_str": row["dato_str"], "municipality_id": row["municipality_id"], "nedbor_mm": mm}

    rows = [row for _, row in unikke.iterrows()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        resultater = list(executor.map(hent_en, rows))

    vejr_df = pd.DataFrame(resultater)
    vejr_df["municipality_id"] = vejr_df["municipality_id"].astype(str).str.zfill(4)
    vejr_df.to_csv(cache_path, index=False)
    print(f"Gemt — {vejr_df['nedbor_mm'].notna().sum()} ud af {len(vejr_df)} dage fik data")
    return vejr_df