# python val.py \
# --data ./data/animal.yaml \
# --weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5s/exp/weights/best.pt \
# --img-size 416 \
# --batch 16 \
# --device 0 

# python val.py \
# --data ./data/animal.yaml \
# --weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5l/exp/weights/best.pt \
# --img-size 416 \
# --batch 16 \
# --device 0 

python val.py \
--data ./data/animal.yaml \
--weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5lEfficientLite/exp/weights/best.pt \
--img-size 416 \
--batch 16 \
--device 0 

