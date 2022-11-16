
# python main.py \
# --datasetType assemble \
# --covidxTrainImagePath [the path of COVIDx train dataset] \
# --covidxTestImagePath [the path of COVIDx test dataset] \
# --chestImagePath [the path of ChestXray14 dataset] \
# --chestFilePath [the list file of ChestXray14 dataset] \
# --covidxTrainFilePath [the train list file of COVIDx dataset] \
# --covidxTestFilePath [the test list file of COVIDx dataset] \
# --numClass 14 \
# --mode train \
# --covidxRatio 1 \
# --isTrain True \
# --chestRatio 0.4 --saveDir all 

rm -rf .work_dir/30000

python main.py \
--datasetType assemble \
--covidxTrainImagePath ../../data/COVIDX/train \
--covidxTestImagePath ../../data/COVIDX/test \
--chestImagePath ../../data/chestXray14/images \
--chestFilePath ../../data/chestXray14/train_official.txt \
--covidxTrainFilePath ../../data/COVIDX/train.txt \
--covidxTestFilePath ../../data/COVIDX/test.txt \
--extraNumClass 1 \
--mode train \
--covidxNum 30000 \
--chestNum 5000 \
--batchSize 16 \
--epochs 64 \
--numWorkers 12 \
--isTrain \
--workDir 30000 \
--loss fully \
--saveDir all  &
