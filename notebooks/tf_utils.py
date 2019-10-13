import tensorflow as tf
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_fmri(img, p_from=None, p_to=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    img_data = img.get_data()
    img_shape = img_data.shape
    p_from = p_from if p_from is not None else 0
    p_to = p_to if p_to is not None else img_data.shape[-1]-1
    
    img_data = img.get_data()[:,:,:,p_from:p_to]
    img_shape = img_data.shape    
    img_raw = img_data.tostring()
    print(img_shape)
    
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    # Create a dictionary describing the features.
    features = tf.train.Features(feature={'height': _int64_feature(img_shape[0]),
                                          'width': _int64_feature(img_shape[1]),
                                          'depth': _int64_feature(img_shape[2]),
                                          'vols': _int64_feature(img_shape[3]),
                                          'img_raw': _bytes_feature(img_raw)})
    example_proto = tf.train.Example(features=features)
    return example_proto.SerializeToString()


def _fmri_parse_function(example_proto, image_description = None):
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'vols': tf.io.FixedLenFeature([], tf.int64),
        'img_raw': tf.io.FixedLenFeature([], tf.string),
    } if image_description is None else image_description
    return tf.io.parse_single_example(example_proto, image_feature_description)


# method to retrive a volume using the filename queue
def read_and_decode(filename_queue, volume_shape):
	reader = tf.io.TFRecordReader()
	key , serialized_example = reader.read(filename_queue)
	features = _fmri_parse_function(serialized_example)
	vol_str = tf.io.decode_raw(features['img_raw'], tf.float32)
	volume = tf.io.reshape(vol_str, volume_shape)
	return volume