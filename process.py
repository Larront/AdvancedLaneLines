import flib

calibration_img_folder = 'calibration'

# (horizontal, vertical)
# some images will not be generated since they don't have (9, 6) inner corners
flib.get_points(calibration_img_folder, 'output_images', (9, 6), False)
flib.get_calibration('output_images', 'test_images', 'calibration_test.jpg', False)

# flib.test_pipeline('test_images', 'output_images', checkpoint='undistorted')
# flib.test_pipeline('test_images', 'output_images', checkpoint='thresholding')
# flib.test_perspective_transform('test_images', 'output_images', checkpoint='undistorted')
# flib.test_perspective_transform('test_images', 'output_images')
# flib.test_pipeline('test_images', 'output_images', checkpoint='warped')
# flib.test_pipeline('test_images', 'output_images', checkpoint='windows')
# flib.test_pipeline('test_images', 'output_images', checkpoint='lanelines')
# flib.test_pipeline('test_images', 'output_images', checkpoint='full')

flib.process_video('test_videos/project_video.mp4', 'output_videos')
