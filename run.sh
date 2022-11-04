
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


python main.py \
--datasetType assemble \
--covidxTrainImagePath /Users/zenglezhu/code/dataset/COVIDX/train \
--covidxTestImagePath /Users/zenglezhu/code/dataset/COVIDX/test \
--chestImagePath /Users/zenglezhu/code/dataset/chestxray14/train \
--chestFilePath /Users/zenglezhu/code/dataset/chestxray14/train_official.txt \
--covidxTrainFilePath /Users/zenglezhu/code/dataset/COVIDX/train.txt \
--covidxTestFilePath /Users/zenglezhu/code/dataset/COVIDX/test.txt \
--extraNumClass 14 \
--mode train \
--covidxNum 10000 \
--chestNum 10000 \
--isTrain True \
--saveDir all  &
