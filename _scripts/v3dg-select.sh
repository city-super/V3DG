# [test]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_SingleDonut.json --TAU 2048 --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

# [composed scenes]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/DonutSea.json --TAU 2048

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 2048

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Crowd.json --TAU 2048

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Temples.json --TAU 8192

# [single large scenes]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Matrixcity.json --TAU 32768

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Downtown.json --TAU 32768
