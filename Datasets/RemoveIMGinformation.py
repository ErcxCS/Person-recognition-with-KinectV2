# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:52:22 2020

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

dataset_path = "Dataset.xlsx"
without_IMG_information = "Dataset without IMG info.xlsx"
dataset = pd.read_excel(dataset_path)

cols = dataset.columns
sayac = 0
for col in cols:
    if 'IMG' in col:
        del dataset[col]
        sayac += 1

print(len(dataset.columns))
print(sayac)

print(dataset.head())

dataset.to_excel(without_IMG_information, index=False, sheet_name='Sheet')
Fix_Cell_Width(without_IMG_information)