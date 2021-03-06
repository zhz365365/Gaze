/usr/bin/python3 ../src/train.py \
--GPU='3' \
--data_type='MPIIGaze' \
--net_name='resnet_v2_101' \
--image_height=144 \
--image_width=240 \
--trainable='1' \
--eyeball='left' \
--mission='phi' \
--batch_size=16 \
--server=29 \
--classes_theta=90 \
--classes_phi=90 \
--build_pyramid=0 \
--epoch=200 \
--lr_boundaries='16,32,48,64,80,96,112,140,170' \
--lr_values='0.0004,0.00004,0.000004,0.0000004,0.00000004,0.000000004,0.0000000004,0.00000000004,0.000000000004,0.0000000000004'