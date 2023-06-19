python detect.py \
--source /root/project/Datas/otherDatas/dataset_n/detect/images/824-3.jpg \
--weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5s/exp/weights/best.pt \
--img-size 416 \
--device 0 

python detect.py \
--source /root/project/Datas/otherDatas/dataset_n/orgs/images/824-3.jpg \
--weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5l/exp/weights/best.pt \
--img-size 416 \
--device 0 

python detect.py \
--source /root/project/Datas/otherDatas/dataset_n/orgs/images/824-3.jpg \
--weights /root/project/Modules/TrackAnomalyTask/jiaofu/seg_lightdet/runs/train/yolov5lEfficientLite/exp/weights/best.pt \
--img-size 416 \
--device 0 



