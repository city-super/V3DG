# [different iters]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-iter0prime.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-iter0.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-iter40.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-iter160.json --TAU 2048
# python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-iter640.json --TAU 2048

# [different ns]

# python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-n4096x2.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-n2048x4.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-n1024x8.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-n512x16.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest-n256x32.json --TAU 2048

# [different taus]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 512
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 1024
# python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 2048
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 4096
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/Forest.json --TAU 8192
