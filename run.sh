#!/bin/bash
# K=50
# LOG="../train_results/logs_dropout_cluster"
# NUM_EPOCHS=400
# SAVE_MODEL_PATH="./results/saved_models/resnet34_dropout_cluster.pth"

# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --use_tensorboard

TEST_CSV_PATH="./csv_files/tcga/test.csv"
# TEST_CSV_PATH="./tcga_csv_50/test.csv"
K=50
NUM_EPOCHS=300
BS=16


# =====train 100 sample, k=50=====
echo "========Fold1========"
LOG="./result/tcga_res_100_50/fold1/logs_fold1"
SAVE_MODEL_PATH="./result/tcga_res_100_50/fold1/resnet34.pth"
TRAIN_CSV_PATH="./csv_files/tcga/fold1/train.csv"
VAL_CSV_PATH="./csv_files/tcga/fold1/val.csv"
python run.py \
    --k $K \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --batch_size $BS \
    --use_tensorboard


echo "========Fold2========"
LOG="./result/tcga_res_100_50/fold2/logs_fold2"
SAVE_MODEL_PATH="./result/tcga_res_100_50/fold2/resnet34.pth"
TRAIN_CSV_PATH="./csv_files/tcga/fold2/train.csv"
VAL_CSV_PATH="./csv_files/tcga/fold2/val.csv"
python run.py \
    --k $K \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --batch_size $BS \
    --use_tensorboard


echo "========Fold3========"
LOG="./result/tcga_res_100_50/fold3/logs_fold3"
SAVE_MODEL_PATH="./result/tcga_res_100_50/fold3/resnet34.pth"
TRAIN_CSV_PATH="./csv_files/tcga/fold3/train.csv"
VAL_CSV_PATH="./csv_files/tcga/fold3/val.csv"
python run.py \
    --k $K \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --batch_size $BS \
    --use_tensorboard


echo "========Fold4========"
LOG="./result/tcga_res_100_50/fold4/logs_fold4"
SAVE_MODEL_PATH="./result/tcga_res_100_50/fold4/resnet34.pth"
TRAIN_CSV_PATH="./csv_files/tcga/fold4/train.csv"
VAL_CSV_PATH="./csv_files/tcga/fold4/val.csv"
python run.py \
    --k $K \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --batch_size $BS \
    --use_tensorboard


echo "========Fold5========"
LOG="./result/tcga_res_100_50/fold5/logs_fold5"
SAVE_MODEL_PATH="./result/tcga_res_100_50/fold5/resnet34.pth"
TRAIN_CSV_PATH="./csv_files/tcga/fold5/train.csv"
VAL_CSV_PATH="./csv_files/tcga/fold5/val.csv"
python run.py \
    --k $K \
    --logdir $LOG \
    --num_epochs $NUM_EPOCHS \
    --save_model_path $SAVE_MODEL_PATH \
    --train_csv_path $TRAIN_CSV_PATH \
    --val_csv_path $VAL_CSV_PATH \
    --test_csv_path $TEST_CSV_PATH \
    --batch_size $BS \
    --use_tensorboard

echo "Done!"

# =============ResNet50 backbone=================
# BATCH_SIZE=8
# echo "========Fold1========"
# LOG="./tcga_csv_50/fold1/logs_dropout_cluster_fold1"
# SAVE_MODEL_PATH="./tcga_csv_50/fold1/resnet50_dropout_cluster.pth"
# TRAIN_CSV_PATH="./tcga_csv_50/fold1/train.csv"
# VAL_CSV_PATH="./tcga_csv_50/fold1/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BATCH_SIZE \
#     --use_tensorboard


# echo "========Fold2========"
# LOG="./tcga_csv_50/fold2/logs_dropout_cluster_fold2"
# # LOG="./tcga_cluster_csv/fold2/logs_dropout_cluster_fold2"
# SAVE_MODEL_PATH="./tcga_csv_50/fold2/resnet50_dropout_cluster.pth"
# # SAVE_MODEL_PATH="./tcga_cluster_csv/fold2/resnet34_dropout_cluster.pth"
# TRAIN_CSV_PATH="./tcga_csv_50/fold2/train.csv"
# # TRAIN_CSV_PATH="./tcga_cluster_csv/fold2/train.csv"
# VAL_CSV_PATH="./tcga_csv_50/fold2/val.csv"
# # VAL_CSV_PATH="./tcga_cluster_csv/fold2/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BATCH_SIZE \
#     --use_tensorboard


# echo "========Fold3========"
# LOG="./tcga_csv_50/fold3/logs_dropout_cluster_fold3"
# # LOG="./tcga_cluster_csv/fold3/logs_dropout_cluster_fold3"
# SAVE_MODEL_PATH="./tcga_csv_50/fold3/resnet50_dropout_cluster.pth"
# # SAVE_MODEL_PATH="./tcga_cluster_csv/fold3/resnet34_dropout_cluster.pth"
# TRAIN_CSV_PATH="./tcga_csv_50/fold3/train.csv"
# # TRAIN_CSV_PATH="./tcga_cluster_csv/fold3/train.csv"
# VAL_CSV_PATH="./tcga_csv_50/fold3/val.csv"
# # VAL_CSV_PATH="./tcga_cluster_csv/fold3/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BATCH_SIZE \
#     --use_tensorboard


# echo "========Fold4========"
# LOG="./tcga_csv_50/fold4/logs_dropout_cluster_fold4"
# # LOG="./tcga_cluster_csv/fold4/logs_dropout_cluster_fold4"
# SAVE_MODEL_PATH="./tcga_csv_50/fold4/resnet50_dropout_cluster.pth"
# # SAVE_MODEL_PATH="./tcga_cluster_csv/fold4/resnet34_dropout_cluster.pth"
# TRAIN_CSV_PATH="./tcga_csv_50/fold4/train.csv"
# # TRAIN_CSV_PATH="./tcga_cluster_csv/fold4/train.csv"
# VAL_CSV_PATH="./tcga_csv_50/fold4/val.csv"
# # VAL_CSV_PATH="./tcga_cluster_csv/fold4/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BATCH_SIZE \
#     --use_tensorboard


# echo "========Fold5========"
# LOG="./tcga_csv_50/fold5/logs_dropout_cluster_fold5"
# # LOG="./tcga_cluster_csv/fold5/logs_dropout_cluster_fold5"
# SAVE_MODEL_PATH="./tcga_csv_50/fold5/resnet50_dropout_cluster.pth"
# # SAVE_MODEL_PATH="./tcga_cluster_csv/fold5/resnet34_dropout_cluster.pth"
# TRAIN_CSV_PATH="./tcga_csv_50/fold5/train.csv"
# # TRAIN_CSV_PATH="./tcga_cluster_csv/fold5/train.csv"
# VAL_CSV_PATH="./tcga_csv_50/fold5/val.csv"
# # VAL_CSV_PATH="./tcga_cluster_csv/fold5/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BATCH_SIZE \
#     --use_tensorboard

# echo "Done!"
# echo "========Fold1========"
# LOG="./tcga_nll/fold1/logs_fold1"
# SAVE_MODEL_PATH="./tcga_nll/fold1/resnet34.pth"
# TRAIN_CSV_PATH="./tcga_csv/fold1/train.csv"
# VAL_CSV_PATH="./tcga_csv/fold1/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BS \
#     --use_tensorboard

# echo "========Fold2========"
# LOG="./tcga_nll/fold2/logs_fold2"
# SAVE_MODEL_PATH="./tcga_nll/fold2/resnet34.pth"
# TRAIN_CSV_PATH="./tcga_csv/fold2/train.csv"
# VAL_CSV_PATH="./tcga_csv/fold2/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BS \
#     --use_tensorboard

# echo "========Fold3========"
# LOG="./tcga_nll/fold3/logs_fold3"
# SAVE_MODEL_PATH="./tcga_nll/fold3/resnet34.pth"
# TRAIN_CSV_PATH="./tcga_csv/fold3/train.csv"
# VAL_CSV_PATH="./tcga_csv/fold3/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BS \
#     --use_tensorboard

# echo "========Fold4========"
# LOG="./tcga_nll/fold4/logs_fold4"
# SAVE_MODEL_PATH="./tcga_nll/fold4/resnet34.pth"
# TRAIN_CSV_PATH="./tcga_csv/fold4/train.csv"
# VAL_CSV_PATH="./tcga_csv/fold4/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BS \
#     --use_tensorboard

# echo "========Fold5========"
# LOG="./tcga_nll/fold5/logs_fold5"
# SAVE_MODEL_PATH="./tcga_nll/fold5/resnet34.pth"
# TRAIN_CSV_PATH="./tcga_csv/fold5/train.csv"
# VAL_CSV_PATH="./tcga_csv/fold5/val.csv"
# python run.py \
#     --k $K \
#     --logdir $LOG \
#     --num_epochs $NUM_EPOCHS \
#     --save_model_path $SAVE_MODEL_PATH \
#     --train_csv_path $TRAIN_CSV_PATH \
#     --val_csv_path $VAL_CSV_PATH \
#     --test_csv_path $TEST_CSV_PATH \
#     --batch_size $BS \
#     --use_tensorboard

# echo "Done!"
