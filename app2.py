# ============================
# ADS PMF SCALING WEB APP
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import os
from datetime import date
import re

st.set_page_config(
    page_title="ADS PMF Scaling Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📊 ADS PMF Scaling Web Application")
st.write("Upload your ADS, PMF, Granular Spec, and Main Spec files to perform PMF scaling.")

# ---------------------------------------------------------
# ✅ Collapsible Upload Section
# ---------------------------------------------------------
with st.expander("📁 Upload Required Files", expanded=True):
    ads_file = st.file_uploader("1. Upload ADS File (CSV or XLSX)", type=["csv", "xlsx"])
    pmf_file = st.file_uploader("2. Upload PMF File (XLSX)", type=["xlsx"])
    gran_file = st.file_uploader("3. Upload Granular Spec File (XLSX)", type=["xlsx"])
    main_spec_file = st.file_uploader("4. Upload Main Spec File (XLSX)", type=["xlsx"]) 

if ads_file and pmf_file and gran_file and main_spec_file: 

    st.success("✅ All files uploaded. Processing will start automatically.")

    progress = st.progress(0)
    status = st.empty()

    # --------------------------------------------------
    # STEP 1 — Load ADS
    # --------------------------------------------------
    status.write("Loading ADS file...")
    if ads_file.name.endswith(".csv"):
        ads_df = pd.read_csv(ads_file, dtype=str)
    else:
        ads_df = pd.read_excel(ads_file, dtype=str)

    progress.progress(10)

    ads_filename = os.path.splitext(ads_file.name)[0]
    today_str = date.today().isoformat()

    # --------------------------------------------------
    # NEW STEP — Load Main Spec (WITH SMART HEADER DETECTION)
    # --------------------------------------------------
    status.write("Reading Main Spec definitions...")
    
    try:
        # 1. Load the Excel File Object
        xl = pd.ExcelFile(main_spec_file)
        
        # 2. Find the correct sheet (looks for "model" in name, or defaults to first visible)
        sheet_name = next((s for s in xl.sheet_names if "model" in s.lower()), None)
        if not sheet_name:
            # Fallback to the first sheet if no sheet has "model" in the name
            sheet_name = xl.sheet_names[0]

        # 3. Read first 50 rows to find the header
        preview = pd.read_excel(main_spec_file, sheet_name=sheet_name, nrows=50, header=None)
        header_row_idx = None

        # 4. Scan rows for keywords "Variable" and "Type"
        for i, row in preview.iterrows():
            cells = [str(x).strip().lower() for x in row.fillna("")]
            # Look for cells containing both "variable" and "type"
            if any("variable" in c for c in cells) and any("type" in c for c in cells):
                header_row_idx = i
                break
        
        if header_row_idx is None:
            st.error("❌ Could not auto-detect header row (with 'Variable' and 'Type') in Main Spec.")
            st.stop()

        # 5. Load the actual dataframe using the found header row
        ms_df = pd.read_excel(main_spec_file, sheet_name=sheet_name, header=header_row_idx, dtype=str)
        ms_df.columns = [str(c).strip().upper() for c in ms_df.columns]

        # 6. Validate Columns
        type_col = next((c for c in ms_df.columns if c in ["TYPE", "TYPES", "VARIABLE TYPE"]), None)
        var_col = next((c for c in ms_df.columns if c in ["VARIABLE", "VARIABLES", "VARIABLE NAME", "VAR NAME", "VARIABLE_NAME"]), None)

        if not type_col or not var_col:
            st.error(f"❌ Found header at row {header_row_idx}, but 'Type' or 'Variable' columns are missing. Found: {ms_df.columns.tolist()}")
            st.stop()
        
        # 7. Setup Selection UI
        available_types = sorted(ms_df[type_col].dropna().unique().tolist())
        
        st.divider()
        st.subheader("⚙️ Configuration")
        selected_type_category = st.radio(
            "Select Variable Type to Process:",
            options=["All"] + available_types,
            index=0,
            horizontal=True
        )
        st.divider()

        # 8. Create Filter List
        if selected_type_category == "All":
            allowed_vars = set(ms_df[var_col].str.upper().str.strip().tolist())
        else:
            allowed_vars = set(ms_df[ms_df[type_col] == selected_type_category][var_col].str.upper().str.strip().tolist())

        # Update filename based on selection
        type_suffix = "All" if selected_type_category == "All" else selected_type_category
        scaled_ads_filename = f"{ads_filename}_Scaled_{type_suffix}_{today_str}.csv"
        log_filename = f"{ads_filename}_Logs_{type_suffix}_{today_str}.xlsx"

    except Exception as e:
        st.error(f"Error processing Main Spec: {e}")
        st.stop()

    # Resume original logic
    ads_df.columns = [c.strip() for c in ads_df.columns]

    geo_col = next((c for c in ads_df.columns if c.upper() == "GEOGRAPHY"), None)
    season_candidates = ["SEASON", "PERIOD_DEFINITION", "TIME_PERIODS"]
    season_col = next((c for c in ads_df.columns if c.upper() in season_candidates), None)

    if geo_col is None or season_col is None:
        st.error("❌ ADS must contain Geography and Season column.")
        st.stop()

    ads_work = pd.DataFrame()
    ads_work["_G"] = ads_df[geo_col].str.upper().str.strip()
    ads_work["_S"] = ads_df[season_col].str.upper().str.strip()

    progress.progress(20)
    status.write("Loading PMF multipliers...")

    # --------------------------------------------------
    # 🌍 Geography Normalization Helper
    # --------------------------------------------------
    def normalize_geo(name: str):
        """Treat BOSE.COM, BOSE_COM, and BOSE COM as identical."""
        return str(name).strip().upper().replace(".", "").replace("_", "").replace(" ", "")

    # --------------------------------------------------
    # STEP 2 — PMF
    # --------------------------------------------------
    pmf = pd.read_excel(pmf_file, sheet_name="PMF", dtype=str)
    pmf.columns = [c.upper() for c in pmf.columns]

    if "SEASON" not in pmf.columns:
        if "PERIOD MAPPING" in pmf.columns:
            pmf.rename(columns={"PERIOD MAPPING": "SEASON"}, inplace=True)
        else:
            st.error("❌ PMF must contain SEASON.")
            st.stop()

    pmf["GEOGRAPHY"] = pmf["GEOGRAPHY"].str.upper().str.strip()
    pmf["SEASON"] = pmf["SEASON"].str.upper().str.strip()

    ads_pmf_cols = [c for c in ads_df.columns if "_PMF" in c.upper()]
    pmf_vars = [c for c in pmf.columns if "_PMF" in c]
    common = [c for c in ads_pmf_cols if c.upper() in pmf_vars]

    pmf_long = pmf[["GEOGRAPHY", "SEASON"] + common].melt(
        id_vars=["GEOGRAPHY", "SEASON"],
        var_name="VARIABLE",
        value_name="PMF_MULT"
    )
    pmf_long["VARIABLE"] = pmf_long["VARIABLE"].str.upper()
    pmf_long["PMF_MULT"] = pd.to_numeric(pmf_long["PMF_MULT"], errors="coerce")

    pmf_dict = {
       (normalize_geo(g), str(s).strip().upper(), str(v).strip().upper()): m
       for g, s, v, m in pmf_long[["GEOGRAPHY", "SEASON", "VARIABLE", "PMF_MULT"]].itertuples(index=False)
    }

    progress.progress(40)
    status.write("Loading Granular Spec skip logic...")

    # --------------------------------------------------
    # STEP 2.5 — Skip Rules
    # --------------------------------------------------
    gran_xl = pd.ExcelFile(gran_file)
    map_df = pd.read_excel(gran_file, sheet_name="MAP", dtype=str)
    map_df.columns = [c.upper() for c in map_df.columns]

    map_df["GEOGRAPHY"] = map_df["GEOGRAPHY"].str.upper().str.strip()
    map_df["MAP"] = map_df["MAP"].str.upper().str.strip()

    geo2map = map_df.set_index("GEOGRAPHY")["MAP"].to_dict()
    ads_work["_MAP"] = ads_work["_G"].apply(normalize_geo).map({normalize_geo(k): v for k, v in geo2map.items()})

    skip_triples = set()
    season_pattern = re.compile(r"^S\d\s20\d{2}$")

    for code in map_df["MAP"].unique():
        sheet = str(code)
        if sheet not in gran_xl.sheet_names:
            continue

        gdf = pd.read_excel(gran_file, sheet_name=sheet, dtype=str)
        gdf = gdf.iloc[:, :4]
        gdf.columns = [c.upper() for c in gdf.columns]

        if "VARIABLE" not in gdf.columns or "CONTRIBUTION" not in gdf.columns:
            continue

        gdf["VARIABLE"] = gdf["VARIABLE"].str.upper().str.strip()
        gdf["CONTRIBUTION"] = gdf["CONTRIBUTION"].str.upper().str.strip()

        valid = gdf["CONTRIBUTION"].str.match(season_pattern, na=False)
        gdf_valid = gdf[valid]

        for _, row in gdf_valid.iterrows():
            skip_triples.add((f"{row['VARIABLE']}_PMF", row["CONTRIBUTION"], sheet))

    progress.progress(60)
    status.write("Applying PMF multipliers...")

    # --------------------------------------------------
    # STEP 3 — Apply Multipliers
    # --------------------------------------------------
    result_ads = ads_df.copy()
    skipped_rows = []
    multiplied_rows = []

    G = ads_work["_G"].values
    S = ads_work["_S"].values
    M = ads_work["_MAP"].values

    def get_base_pmf(c):
        name = c.upper()
        i = name.find("_PMF")
        return name if i == -1 else name[: i]

    for col in common:
        col_u = col.upper()
        col_base = get_base_pmf(col_u)

        # <--- Check if variable matches selected Type
        # If it doesn't match, we continue loop (Silent Skip)
        # We do NOT log these, to prevent the Log file from crashing with millions of rows.
        if selected_type_category != "All":
            if col_base not in allowed_vars:
                continue 
        # --------------------------------------------

        column_vals = pd.to_numeric(result_ads[col], errors="coerce")

        for i in range(len(result_ads)):

            if (col_base + "_PMF", S[i], M[i]) in skip_triples:
                # Logged because it WAS selected, but skipped due to Granular Rule
                skipped_rows.append((i, G[i], S[i], M[i], col, "Granular Skip Rule"))
                continue

            mult = pmf_dict.get((normalize_geo(G[i]), S[i], col_u))
            if mult is None or pd.isna(column_vals.iat[i]):
                continue

            updated = column_vals.iat[i] * mult
            result_ads.at[i, col] = updated

            multiplied_rows.append(
                (i, G[i], S[i], M[i], col, column_vals.iat[i], mult, updated)
            )

    progress.progress(90)
    status.write("Preparing logs...")

    # Log file creation
    skipped_df = pd.DataFrame(skipped_rows, columns=["Row", "Geo", "Season", "MAP", "Variable", "Reason"])
    multiplied_df = pd.DataFrame(multiplied_rows,
                                 columns=["Row", "Geo", "Season", "MAP", "Variable", "Original", "Multiplier", "Updated"])

    log_output = io.BytesIO()
    with pd.ExcelWriter(log_output, engine="openpyxl") as writer:
        skipped_df.to_excel(writer, sheet_name="Skipped", index=False)
        multiplied_df.to_excel(writer, sheet_name="Multiplied", index=False)

    progress.progress(100)
    status.write("✅ Completed!")

    st.success(f"✅ PMF Scaling Completed! (Type: {selected_type_category})")

    # --------------------------------------------------
    # ✅ DOWNLOAD SECTION
    # --------------------------------------------------
    with st.expander("📥 Download Outputs", expanded=True):

        st.download_button(
            label="⬇️ Download Scaled ADS CSV",
            data=result_ads.to_csv(index=False).encode(),
            file_name=scaled_ads_filename,
            mime="text/csv"
        )

        st.download_button(
            label="⬇️ Download Logs (Excel)",
            data=log_output.getvalue(),
            file_name=log_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )