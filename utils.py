import tensorflow as tf
import argparse
import os

# Input dir of images
image_dir = os.getcwd() + '/small_dataset'
# Percentage for test
test_percentage = 10
# Percentage for validation
validation_percentage = 10                          


""" Check images format (jpg, png allowed)
Args:
     image_dir:  String - path to the dataset directories.
"""
def check_image_format(image_dir):

    for root, dirs, files in os.walk(image_dir):
        for image in files:
            ext = ih.what(os.path.join(root, image))
            if (ext != 'jpeg' and ext != 'png'):
                return image
            
    return 0


""" Load the absolute path of an image and convert it in a numpy ndarray
Args:
     filename: Absolute path of image
Return:
     a numpy ndarray
"""
def load_image(file_name):
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data



""" IMAGE RESIZE WITH CROP """
def resize_image_with_crop_or_pad(image, target_height, target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    zero = tf.constant(0)
    half = tf.constant(2)

    offset_crop_width = tf.python.control_flow_ops.cond(
        tf.less(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.sub(original_width, target_width), half),
        lambda: zero)

    offset_pad_width = tf.python.control_flow_ops.cond(
        tf.greater(
            target_width,
            original_width),
        lambda: tf.floordiv(tf.sub(target_width, original_width), half),
        lambda: zero)

    offset_crop_height = tf.python.control_flow_ops.cond(
        tf.less(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.sub(original_height, target_height), half),
        lambda: zero)

    offset_pad_height = tf.python.control_flow_ops.cond(
        tf.greater(
            target_height,
            original_height),
        lambda: tf.floordiv(tf.sub(target_height, original_height), half),
        lambda: zero)

    cropped = crop_to_bounding_box(
        image, offset_crop_height, offset_crop_width,
        tf.minimum(target_height, original_height),
        tf.minimum(target_width, original_width))

    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                  target_height, target_width)

    return resized


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width):
    cropped = tf.slice(
        image,
        tf.pack([offset_height, offset_width, 0]),
        tf.pack([target_height, target_width, -1]))

    return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width):
    image_shape = tf.shape(image)
    original_height = image_shape[0]
    original_width = image_shape[1]

    after_padding_width = tf.sub(
        tf.sub(target_width, offset_width),  original_width)
    after_padding_height = tf.sub(
        tf.sub(target_height, offset_height), original_height)

    paddings = tf.reshape(
        tf.pack(
            [offset_height, after_padding_height,
             offset_width, after_padding_width,
             0, 0]), [3, 2])

    padded = tf.pad(image, paddings)

    return padded


def pad_and_crop_image_dimensions(target_height, target_width, image_dir):
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(
            "{image_dir}/*.jpg".format(image_dir=image_dir)))

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)

    resized_padding = resize_image_with_crop_or_pad(image, target_height + 1,
                                                    target_width + 1)
    resized_cropping = resize_image_with_crop_or_pad(image, target_height - 1,
                                                     target_width - 1)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        result = sess.run([image, resized_padding, resized_cropping])
        yield(result[0], result[1], result[2])

        coord.request_stop()
        coord.join(threads)


# if __name__ == "__main__":
#     def check_if_images_exists(parser, image_dir):
#         if not os.path.isdir(image_dir):
#             parser.error(
#                 "The directory {image_dir} does not exist.".format(
#                     image_dir=image_dir))

#         return image_dir

#     parser = argparse.ArgumentParser(
#         description="Pad/Crop JPEG images from a file queue.")
#     parser.add_argument(
#         "--image-dir",
#         default="./images",
#         type=lambda image_dir: check_if_images_exists(parser, image_dir),
#         help="Relative directory with image files (extension .jpg required).")
#     args = parser.parse_args()

#     padded_and_cropped_dimensions = pad_and_crop_image_dimensions(
#         2, 2, args.image_dir)
#     for image, padded_image, cropped_image in padded_and_cropped_dimensions:
#         print("Original Image:\n{original_image}\n\n".format(
#             original_image=image))
#         print("Padded Image:\n{padded_image}\n\n".format(
#             padded_image=padded_image))
#         print("Cropped Image:\n{cropped_image}\n\n".format(
#             cropped_image=cropped_image))
