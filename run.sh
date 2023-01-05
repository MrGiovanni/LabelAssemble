rm -rf .work_dir/30000

python main.py \
--datasetType assemble \
--mode train \
--batchSize 16 \
--epochs 64 \
--numWorkers 12 \
--isTrain \
--workDir 30000 \
--loss fully \
--saveDir all  &
