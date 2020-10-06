# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
from openpyxl import load_workbook

def FnF(excel, col, row):
    
    if row == len(excel) - 1:
        
        return excel[col][row - 1]
    else:
        _next = row + 1
        while excel[col][_next] == float('-inf'):
            if _next == len(excel) - 1:
                
                return excel[col][row - 1]
            else:
                _next += 1
    
    if row == 0:

        return excel[col][_next]
    else:
        _prev = row - 1
    
    return (excel[col][_next] + excel[col][_prev]) / 2

def Fix_Cell_Width(fixed_path):
    wb = load_workbook(fixed_path)
    sheet = wb.active
    cols = sheet.columns
    for col in cols:
        sheet.column_dimensions[col[0].column_letter].width = 23
    wb.save(fixed_path)

    
kisiler_path = "Kişiler (fixed)"
#kisiler_path = "Kişiler"
veriler_path = "/data2"
excel_path = "/body_joints2.xlsx"
fixed_excel_path = "/body_joints_fixed.xlsx"


kisiler = os.listdir(kisiler_path)

for person in kisiler:
    if person != "Template" and person != "13" and person != "14":
        tryouts_path = kisiler_path + "/" + person
        tryouts = os.listdir(tryouts_path)
        
        for tryout in tryouts:
            full_path = tryouts_path + "/" + tryout + veriler_path + excel_path
            excel = pd.read_excel(full_path)
            leng = len(excel)

            for col in excel.columns:
                for row in range(leng):
                    if excel[col][row] == float('-inf'):
                        excel[col][row] = FnF(excel, col, row)
            
            fixed_path = tryouts_path + "/" + tryout + veriler_path + fixed_excel_path
            excel.to_excel(fixed_path, index=False, sheet_name='Sheet')
            
            Fix_Cell_Width(fixed_path)

