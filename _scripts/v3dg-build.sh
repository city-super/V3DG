# [test]

python v3dg-build.py --ASSET_KIND donut --ASSET_NAME donut --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION 16

# [build objects]

python v3dg-build.py --ASSET_KIND donut --ASSET_NAME donut --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply

for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply
done

for ASSET_NAME in 100831 100833 100835 100837 100839 100841 100843 100845 100847 100849 100851 100853 100855 100857 100859 100861; do
    python v3dg-build.py --ASSET_KIND mvhumannet --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_8000/fused.ply
done

# [build rectified scenes]

python v3dg-build.py --ASSET_KIND matrixcity --ASSET_NAME matrixcity-12w-blackbg-rectified --ASSET_GSPLY_FILENAME point_cloud/iteration_120000/fused.ply

python v3dg-build.py --ASSET_KIND scenes --ASSET_NAME downtown-12w-blackbg-rectified --ASSET_GSPLY_FILENAME point_cloud/iteration_120000/fused.ply

python v3dg-build.py --ASSET_KIND scenes --ASSET_NAME temple-12w-blackbg-rectified --ASSET_GSPLY_FILENAME point_cloud/iteration_120000/fused.ply
