from pathlib import Path
import pandas as pd

# =========================================================
# PATHS
# =========================================================
source_dir = Path("scripts/figures/v8/source_data")
output_file = source_dir / "Supplementary_Data_1.xlsx"

MAX_EXCEL_DATA_ROWS = 1_048_575

# =========================================================
# FIND REAL CSV FILES
# Ignore macOS ._xxx.csv metadata files
# =========================================================
csv_files = sorted(
    [
        f for f in source_dir.glob("*.csv")
        if not f.name.startswith("._")
    ],
    key=lambda p: p.stem.lower()
)

print(f"Found {len(csv_files)} CSV files:")

for csv_file in csv_files:
    print(f"  {csv_file.name}")


# =========================================================
# WRITE XLSX
# =========================================================
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

    workbook = writer.book

    header_format = workbook.add_format({
        "bold": True,
        "valign": "top"
    })

    for csv_file in csv_files:

        sheet_name = csv_file.stem[:31]

        print(f"\nReading {csv_file.name} ...")

        df = pd.read_csv(csv_file)

        print(f"  rows    = {len(df):,}")
        print(f"  columns = {len(df.columns):,}")
        print(f"  sheet   = {sheet_name}")

        # =================================================
        # CHECK EXCEL LIMIT
        # =================================================
        if len(df) > MAX_EXCEL_DATA_ROWS:
            raise ValueError(
                f"{csv_file.name} contains {len(df):,} rows, "
                f"which exceeds Excel's maximum of "
                f"{MAX_EXCEL_DATA_ROWS:,} data rows per sheet."
            )

        if len(df.columns) > 16_384:
            raise ValueError(
                f"{csv_file.name} contains {len(df.columns):,} columns, "
                f"which exceeds Excel's maximum of 16,384 columns."
            )

        # =================================================
        # WRITE DATA
        # =================================================
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        worksheet = writer.sheets[sheet_name]

        # =================================================
        # HEADER FORMAT
        # =================================================
        for col_num, column_name in enumerate(df.columns):
            worksheet.write(0, col_num, column_name, header_format)

        # Freeze header
        worksheet.freeze_panes(1, 0)

        # Autofilter
        if len(df.columns) > 0:
            worksheet.autofilter(
                0,
                0,
                len(df),
                len(df.columns) - 1
            )

        # =================================================
        # COLUMN WIDTH
        # Only inspect first 1000 rows for speed
        # =================================================
        for col_num, column_name in enumerate(df.columns):

            sample = df[column_name].dropna().astype(str).head(1000)

            if len(sample) > 0:
                max_value_length = sample.str.len().max()
            else:
                max_value_length = 0

            width = max(
                len(str(column_name)),
                max_value_length
            ) + 2

            width = min(width, 30)

            worksheet.set_column(
                col_num,
                col_num,
                width
            )

        print(f"  Written: {sheet_name}")


# =========================================================
# DONE
# =========================================================
print("\n========================================")
print("Done.")
print(f"Saved to: {output_file}")
print(f"Total sheets: {len(csv_files)}")
print("========================================")