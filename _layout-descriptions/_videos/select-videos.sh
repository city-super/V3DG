# [composed scenes]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/DonutSea-blender-trajectory-zoominout.json --TAU 2048 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/Forest-blender-trajectory-zoominout.json --TAU 2048 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/Crowd-blender-trajectory-zoominout.json --TAU 2048 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/Temples-blender-trajectory-zoominout.json --TAU 8192 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

# [individual scenes]

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/Matrixcity-blender-trajectory-zoominout-up.json --TAU 32768 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f

python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_videos/Downtown-blender-trajectory-zoominout-yx.json --TAU 32768 --SAVE_IMAGES_CONTINUOUS t --SAVE_IMAGES_COMPARISON f --SAVE_METRICS_AT_DIFFERENT_DISTANCES f
