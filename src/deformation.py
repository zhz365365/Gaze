import tensorflow as tf

def _to_c_h_w_1(x):
    x = tf.reduce_mean(x, axis=0, keep_dims=True)
    return tf.transpose(x, [3, 1, 2, 0])

def _to_bc_h_w_2(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, 2c] -> [b*c, h, w, 2]
    return tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1], x_shape[2], 2])

def _to_bc_h_w(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, c] -> [b*c, h, w]
    return tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1], x_shape[2]])

def _to_bc_hw(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b, h, w, c] -> [b*c, h*w]
    return  tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1, x_shape[1] * x_shape[2]])

def _to_b_h_w_c(x, x_shape):
    #x_shape = [b, h, w, c]
    #[b*c, h, w] -> [b, h, w, c]
    #[b*c, h*w] -> [b, h, w, c]
    #[b, c, h*w] -> [b, h, w, c]
    return tf.transpose(tf.reshape(x, [-1, x_shape[3], x_shape[1], x_shape[2]]), [0, 2, 3, 1])

def tf_flatten(a):
    return tf.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    assert len(a.get_shape()) == 1
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def tf_repeat_2d(a, repeats):
    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

def tf_batch_map_offsets(x, offsets, order=1):
    input_shape = tf.shape(x)
    batch_channel_size = input_shape[0]
    height_size = input_shape[1]
    width_size = input_shape[2]

    offsets = tf.reshape(offsets, [batch_channel_size, -1, 2])

    grid = tf.meshgrid(tf.range(height_size), tf.range(width_size), indexing='ij')
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_channel_size)
    coords = offsets + grid
    mapped_vals = tf_batch_map_coordinates(x, coords)
    return mapped_vals

def tf_batch_map_coordinates(x, coords, order=1):
    input_shape = tf.shape(x)
    batch_channel_size = input_shape[0]
    height_size = input_shape[1]
    width_size = input_shape[2]
    h_plus_w_size = tf.shape(coords)[1]

    coords = tf.stack([tf.clip_by_value(coords[:,:,0], 0, tf.cast(height_size, tf.float32) - 1), 
                       tf.clip_by_value(coords[:,:,1], 0, tf.cast(width_size, tf.float32) - 1)], axis=-1)

    coords_lt = tf.cast(tf.floor(coords), tf.int32)
    coords_rb = tf.cast(tf.ceil(coords), tf.int32)
    coords_lb = tf.stack([coords_lt[:,:,0], coords_rb[:,:,1]], axis=-1)
    coords_rt = tf.stack([coords_rb[:,:,0], coords_lt[:,:,1]], axis=-1)

    idx = tf_repeat(tf.range(batch_channel_size), h_plus_w_size)

    def _get_vals_by_coords(feature_map, coords):
        indices = tf.stack([idx, tf_flatten(coords[:,:,0]), tf_flatten(coords[:,:,1])], axis=-1)
        vals = tf.gather_nd(feature_map, indices)
        vals = tf.reshape(vals, (batch_channel_size, h_plus_w_size))
        return vals

    vals_lt = _get_vals_by_coords(x, coords_lt)
    vals_rb = _get_vals_by_coords(x, coords_rb)
    vals_lb = _get_vals_by_coords(x, coords_lb)
    vals_rt = _get_vals_by_coords(x, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, tf.float32)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:,:,0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:,:,0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:,:,1]

    return mapped_vals


