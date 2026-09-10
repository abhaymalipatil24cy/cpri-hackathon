import pdfplumber
import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path

PDF_PATH = r'C:\Users\ROG\Downloads\CPRI_Hackathon_Screening_Dataset_PARTICIPANT.pdf'

def parse_pdf_training(pdf_path):
    rows_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(3, 21): # Pages 4 to 21
            page = pdf.pages[page_idx]
            words = page.extract_words()
            
            lines_dict = {}
            for w in words:
                if w['top'] < 70 and page_idx == 3:
                    continue
                if w['top'] < 25 and page_idx > 3:
                    continue

                top_key = round(w['top'], 1)
                matched_key = None
                for k in lines_dict:
                    if abs(k - top_key) < 2.5:
                        matched_key = k
                        break
                if matched_key is None:
                    matched_key = top_key
                    lines_dict[matched_key] = []
                lines_dict[matched_key].append(w)
            
            for k in sorted(lines_dict.keys()):
                line_words = sorted(lines_dict[k], key=lambda x: x['x0'])
                if not line_words or not line_words[0]['text'].startswith('TRN-'):
                    continue
                
                test_id = line_words[0]['text'].strip()
                cols = {c: None for c in range(11)}
                cols[0] = test_id
                
                for w in line_words[1:]:
                    x0 = w['x0']
                    text = w['text']
                    
                    if 100 <= x0 < 175:
                        cols[1] = text
                    elif 175 <= x0 < 240:
                        cols[2] = text
                    elif 240 <= x0 < 310:
                        cols[3] = text
                    elif 310 <= x0 < 385:
                        cols[4] = text
                    elif 385 <= x0 < 450:
                        cols[5] = text
                    elif 450 <= x0 < 520:
                        cols[6] = text
                    elif 520 <= x0 < 590:
                        cols[7] = text
                    elif 590 <= x0 < 655:
                        cols[8] = text
                    elif x0 >= 655:
                        if text.endswith('Valid') and not text.endswith('Invalid'):
                            cols[9] = text[:-5]
                            cols[10] = 'Valid'
                        elif text.endswith('Invalid'):
                            cols[9] = text[:-7]
                            cols[10] = 'Invalid'
                        elif text in ['Valid', 'Invalid']:
                            cols[10] = text
                        else:
                            cols[9] = text

                rows_data.append({
                    'Test_ID': cols[0],
                    'Applied_Voltage_kV': cols[1],
                    'Load_Current_A': cols[2],
                    'Ambient_Temperature_C': cols[3],
                    'Test_Duration_min': cols[4],
                    'Sensor_S1': cols[5],
                    'Sensor_S2': cols[6],
                    'Sensor_S3': cols[7],
                    'Sensor_S4': cols[8],
                    'Reference_Parameter': cols[9],
                    'Validity_Label': cols[10]
                })

    df = pd.DataFrame(rows_data)
    return df

def parse_pdf_test(pdf_path):
    all_test_rows = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for p_offset in range(7):
            p1_idx = 21 + p_offset
            p2_idx = 28 + p_offset
            p3_idx = 35 + p_offset
            
            p1 = pdf.pages[p1_idx]
            p2 = pdf.pages[p2_idx]
            p3 = pdf.pages[p3_idx]
            
            def get_lines(page, is_p1_first):
                words = page.extract_words()
                words_sorted = sorted(words, key=lambda w: (w['top'], w['x0']))
                line_groups = []
                curr_line = []
                curr_y = None
                for w in words_sorted:
                    if is_p1_first and w['top'] < 70:
                        continue
                    if not is_p1_first and w['top'] < 25:
                        continue
                    if curr_y is None or abs(w['top'] - curr_y) < 5.0:
                        curr_line.append(w)
                        if curr_y is None:
                            curr_y = w['top']
                    else:
                        line_groups.append((curr_y, sorted(curr_line, key=lambda x: x['x0'])))
                        curr_line = [w]
                        curr_y = w['top']
                if curr_line:
                    line_groups.append((curr_y, sorted(curr_line, key=lambda x: x['x0'])))
                return line_groups

            l1 = get_lines(p1, p1_idx == 21)
            l2 = get_lines(p2, p2_idx == 28)
            l3 = get_lines(p3, p3_idx == 35)

            l1 = [g for g in l1 if g[1] and g[1][0]['text'].startswith('TST-')]
            l2 = [g for g in l2 if g[1] and not g[1][0]['text'].startswith('Test_Duration')]
            l3 = [g for g in l3 if g[1] and not g[1][0]['text'].startswith('Sensor_S4')]

            for i, (y1, w1) in enumerate(l1):
                test_id = w1[0]['text'].strip()
                
                cols_p1 = [None, None, None]
                for w in w1[1:]:
                    x0 = w['x0']
                    if 120 <= x0 < 250:
                        cols_p1[0] = w['text']
                    elif 250 <= x0 < 350:
                        cols_p1[1] = w['text']
                    elif x0 >= 350:
                        cols_p1[2] = w['text']

                cols_p2 = [None, None, None, None]
                matched_l2 = None
                for y2, w2 in l2:
                    if abs(y2 - y1) < 4.0:
                        matched_l2 = w2
                        break
                if matched_l2 is not None:
                    for w in matched_l2:
                        x0 = w['x0']
                        if x0 < 200:
                            cols_p2[0] = w['text']
                        elif 200 <= x0 < 310:
                            cols_p2[1] = w['text']
                        elif 310 <= x0 < 420:
                            cols_p2[2] = w['text']
                        elif x0 >= 420:
                            cols_p2[3] = w['text']

                matched_s4 = None
                for y3, w3 in l3:
                    if abs(y3 - y1) < 4.0:
                        matched_s4 = w3[0]['text']
                        break

                all_test_rows.append({
                    'Test_ID': test_id,
                    'Applied_Voltage_kV': cols_p1[0],
                    'Load_Current_A': cols_p1[1],
                    'Ambient_Temperature_C': cols_p1[2],
                    'Test_Duration_min': cols_p2[0],
                    'Sensor_S1': cols_p2[1],
                    'Sensor_S2': cols_p2[2],
                    'Sensor_S3': cols_p2[3],
                    'Sensor_S4': matched_s4
                })

    return pd.DataFrame(all_test_rows)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def run_verification():
    print("=== CP12.1 FULL GROUND-TRUTH VERIFICATION ===")
    
    df_pdf_trn = parse_pdf_training(PDF_PATH)
    df_pdf_tst = parse_pdf_test(PDF_PATH)
    
    df_csv_trn = pd.read_csv('data/training_data.csv')
    df_csv_tst = pd.read_csv('data/test_data.csv')
    df_sub = pd.read_csv('outputs/TeamName.csv')

    print(f"PDF Training rows: {len(df_pdf_trn)}, CSV Training rows: {len(df_csv_trn)}")
    print(f"PDF Test rows: {len(df_pdf_tst)}, CSV Test rows: {len(df_csv_tst)}")

    # 1. ID checks
    trn_pdf_ids = set(df_pdf_trn['Test_ID'])
    trn_csv_ids = set(df_csv_trn['Test_ID'])
    tst_pdf_ids = set(df_pdf_tst['Test_ID'])
    tst_csv_ids = set(df_csv_tst['Test_ID'])
    sub_ids = set(df_sub['Test_ID'])

    missing_trn_ids = trn_pdf_ids - trn_csv_ids
    extra_trn_ids = trn_csv_ids - trn_pdf_ids
    missing_tst_ids = tst_pdf_ids - tst_csv_ids
    extra_tst_ids = tst_csv_ids - tst_pdf_ids
    overlap = trn_csv_ids.intersection(tst_csv_ids)

    # 2. Value comparisons by joining on Test_ID
    # Convert CSV numeric columns to string / float for exact comparison
    num_cols_trn = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min', 
                    'Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4', 'Reference_Parameter']
    num_cols_tst = ['Applied_Voltage_kV', 'Load_Current_A', 'Ambient_Temperature_C', 'Test_Duration_min', 
                    'Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']

    merged_trn = pd.merge(df_pdf_trn, df_csv_trn, on='Test_ID', suffixes=('_pdf', '_csv'))
    merged_tst = pd.merge(df_pdf_tst, df_csv_tst, on='Test_ID', suffixes=('_pdf', '_csv'))

    trn_text_mismatches = 0
    trn_num_mismatches = 0
    ref_diffs = []
    label_mismatches = 0

    for idx, row in merged_trn.iterrows():
        # Label
        if str(row['Validity_Label_pdf']).strip() != str(row['Validity_Label_csv']).strip():
            label_mismatches += 1

        for col in num_cols_trn:
            val_pdf_str = str(row[f'{col}_pdf']).strip() if pd.notnull(row[f'{col}_pdf']) else 'nan'
            val_csv = row[f'{col}_csv']
            
            # Numeric conversion
            try:
                v_pdf = float(val_pdf_str)
            except ValueError:
                v_pdf = np.nan
            
            try:
                v_csv = float(val_csv)
            except (ValueError, TypeError):
                v_csv = np.nan

            if pd.isna(v_pdf) and pd.isna(v_csv):
                continue
            elif pd.isna(v_pdf) != pd.isna(v_csv):
                trn_num_mismatches += 1
                if col == 'Reference_Parameter':
                    ref_diffs.append((row['Test_ID'], v_pdf, v_csv))
            else:
                diff = abs(v_pdf - v_csv)
                if diff > 1e-4:
                    trn_num_mismatches += 1
                if col == 'Reference_Parameter':
                    ref_diffs.append((row['Test_ID'], v_pdf, v_csv, diff))

    tst_num_mismatches = 0
    for idx, row in merged_tst.iterrows():
        for col in num_cols_tst:
            val_pdf_str = str(row[f'{col}_pdf']).strip() if pd.notnull(row[f'{col}_pdf']) else 'nan'
            val_csv = row[f'{col}_csv']
            
            try:
                v_pdf = float(val_pdf_str)
            except ValueError:
                v_pdf = np.nan
            
            try:
                v_csv = float(val_csv)
            except (ValueError, TypeError):
                v_csv = np.nan

            if pd.isna(v_pdf) and pd.isna(v_csv):
                continue
            elif pd.isna(v_pdf) != pd.isna(v_csv):
                tst_num_mismatches += 1
            else:
                if abs(v_pdf - v_csv) > 1e-4:
                    tst_num_mismatches += 1

    # 3. Label Counts
    val_counts = df_csv_trn['Validity_Label'].value_counts().to_dict()

    # 4. Missing counts
    pdf_trn_nulls = {c: int(df_pdf_trn[c].isnull().sum()) for c in ['Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']}
    csv_trn_nulls = {c: int(df_csv_trn[c].isnull().sum()) for c in ['Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']}
    
    pdf_tst_nulls = {c: int(df_pdf_tst[c].isnull().sum()) for c in ['Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']}
    csv_tst_nulls = {c: int(df_csv_tst[c].isnull().sum()) for c in ['Sensor_S1', 'Sensor_S2', 'Sensor_S3', 'Sensor_S4']}

    # 5. Hashes
    submission_hash = sha256_file('outputs/TeamName.csv')
    cp5_hash = sha256_file('artifacts/validity/s3_consistency_model.pkl')
    cp6_hash = sha256_file('artifacts/validity/validity_model.pkl')
    cp7_hash = sha256_file('artifacts/regression/final_model_metadata.json')
    cp8_hash = sha256_file('artifacts/validity/attention_score_spec.json')

    results = {
        'df_pdf_trn_shape': list(df_pdf_trn.shape),
        'df_csv_trn_shape': list(df_csv_trn.shape),
        'df_pdf_tst_shape': list(df_pdf_tst.shape),
        'df_csv_tst_shape': list(df_csv_tst.shape),
        'missing_trn_ids': list(missing_trn_ids),
        'extra_trn_ids': list(extra_trn_ids),
        'missing_tst_ids': list(missing_tst_ids),
        'extra_tst_ids': list(extra_tst_ids),
        'overlap_count': len(overlap),
        'trn_num_mismatches': trn_num_mismatches,
        'tst_num_mismatches': tst_num_mismatches,
        'label_mismatches': label_mismatches,
        'ref_param_mismatches': len([r for r in ref_diffs if len(r) > 3 and r[3] > 1e-4]),
        'val_counts': val_counts,
        'pdf_trn_nulls': pdf_trn_nulls,
        'csv_trn_nulls': csv_trn_nulls,
        'pdf_tst_nulls': pdf_tst_nulls,
        'csv_tst_nulls': csv_tst_nulls,
        'submission_hash': submission_hash,
        'cp5_hash': cp5_hash,
        'cp6_hash': cp6_hash,
        'cp7_hash': cp7_hash,
        'cp8_hash': cp8_hash
    }

    print("\n--- VERIFICATION SUMMARY ---")
    print(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    run_verification()
