# [ablation - different iters (Forest)]

ITER=0 # iter0prime
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER --SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION f
done

ITER=0
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER
done

ITER=40
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER
done

ITER=160
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER
done

# [ablation - different ns (Forest)]

ITER=640
N_G_IN_C=256
N_C_IN_CG=32
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER  --BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER $N_G_IN_C --BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP $N_C_IN_CG --BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER $N_C_IN_CG
done

ITER=640
N_G_IN_C=512
N_C_IN_CG=16
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER  --BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER $N_G_IN_C --BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP $N_C_IN_CG --BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER $N_C_IN_CG
done

ITER=640
N_G_IN_C=1024
N_C_IN_CG=8
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER  --BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER $N_G_IN_C --BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP $N_C_IN_CG --BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER $N_C_IN_CG
done

ITER=640
N_G_IN_C=2048
N_C_IN_CG=4
for ASSET_NAME in bonsai crispy desk dinosaur fall house oak pine; do
    python v3dg-build.py --ASSET_KIND trees --ASSET_NAME $ASSET_NAME --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION $ITER  --BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER $N_G_IN_C --BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP $N_C_IN_CG --BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER $N_C_IN_CG
done
