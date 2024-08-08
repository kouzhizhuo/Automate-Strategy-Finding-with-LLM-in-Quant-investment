import pandas as pd


def read_alpha_formula_from_excel(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        alpha_formulas = []
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            target_columns = df.columns[
                df.columns.str.contains("Alpha Formula", case=False)
            ].tolist()

            if not target_columns:
                print(f"Cannot find 'Alpha Formula' in {sheet_name}")
            else:
                alpha_formula = df[target_columns[0]].values.tolist()
                alpha_formulas.extend(alpha_formula)

        return alpha_formulas

    except Exception as e:
        print(f"Error reading excel file: {str(e)}")
        return []


if __name__ == "__main__":
    file_path = "./data/Seed Alpha.xlsx"
    alpha_formulas = read_alpha_formula_from_excel(file_path)
    print("Alpha Formulas:")
    print(alpha_formulas)
