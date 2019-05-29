/usr/bin/python3 ../src/train.py \
--GPU='1' \
--data_type='LOPO_MPIIGaze' \
--net_name='hourglasses' \
--trainable='1' \
--eyeball='left' \
--batch_size=16 \
--server=29 \
--build_pyramid=0 \
--build_pyramid_layers=1 \
--epoch=400 \
--reduce_mean=0 \
--leave_one=14 \
--net_head='stem' \
--lr_boundaries='100,200,300' \
--lr_values='0.00001,0.000001,0.0000001,0.00000001'