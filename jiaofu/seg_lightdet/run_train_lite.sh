GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-28523}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --batch 32 \
    --imgsz 416 \
    --data ./data/animal.yaml \
    --weights /root/project/pretrains/yolov5lEfficientLite.pt \
    --cfg ./models/lightModels/yolov5lEfficientLite.yaml \
    --hyp ./data/hyps/hyp.scratch-low.yaml \
    --device 0 \
    --name yolov5lEfficientLite/exp \
    --epochs 200 \
    --cache

