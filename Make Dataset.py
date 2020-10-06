# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:23:57 2020

@author: berke
"""

import os
import pandas as pd
from openpyxl import load_workbook

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

frames = []
for person in kisiler:
    
    if person != "Template" and person != "13" and person != "14":
        tryouts_path = kisiler_path + "/" + person
        tryouts = os.listdir(tryouts_path)
        
        for tryout in tryouts:
            full_path = tryouts_path + "/" + tryout + veriler_path + fixed_excel_path
            excel = pd.read_excel(full_path)
            
            frames.append(excel)
            
full_dataset = "datasets/Dataset.xlsx"
all_excel = pd.concat(frames)
all_excel.to_excel(full_dataset, index=False, sheet_name='Sheet')

Fix_Cell_Width(full_dataset)
