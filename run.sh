
python main.py \
--datasetType assemble \
--covidxTrainImagePath [the path of COVIDx train dataset] \
--covidxTestImagePath [the path of COVIDx test dataset] \
--chestImagePath [the path of ChestXray14 dataset] \
--chestFilePath [the list file of ChestXray14 dataset] \
--covidxTrainFilePath [the train list file of COVIDx dataset] \
--covidxTestFilePath [the test list file of COVIDx dataset] \
--numClass 14 \
--mode train \
--covidxRatio 1 \
--isTrain True \
--chestRatio 0.4 --saveDir all 