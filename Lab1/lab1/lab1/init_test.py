import time

import threading
import time
import random

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import picar_4wd as fc


class Car:
    def __init__(self):
        # Car could have several state like: uninit, running, stop
        self.carState = "uninit"

    # Creates a thread that sleeps X minutes and then stops the car
    def run_x_minutes(self, m: int):

        def car_timeout():
            time.sleep(m * 60)
            print("Stop car...")
            fc.stop()
            self.carState = "stop"

        threading.Thread(target=car_timeout).start()
        self.carState = "running"
        print("Start car...")

    def enable_camera(self):

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        # Initialize the object detection model
        base_options = core.BaseOptions(file_name='efficientdet_lite0.tflite', use_coral=False, num_threads=4)
        detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)

        def capture():
            while cap.isOpened() and self.carState != "stop":
                success, image = cap.read()
                if not success:
                    print("Error: Unable to read from webcam. Please verify your webcam settings.")
                    break

                image = cv2.flip(image, 1)

                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Create a TensorImage object from the RGB image.
                input_tensor = vision.TensorImage.create_from_array(rgb_image)

                # Run object detection estimation using the model.
                detection_result = detector.detect(input_tensor)

                # TODO: Improve handling of the detected images
                for detection in detection_result.detections:
                    print(detection.classes)
                    for c in detection.classes:
                        if c.class_name == "stop sign":
                            print('Stop Sign Found!!!')
                            fc.stop()
                            time.sleep(5)

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=capture).start()

    ###
    # Image model is used to analyze "world" in from of the car
    # The model has 3 main parts: 0 - left, 1 - center, 2 - right
    # Parameters l and r specify angle for the analysis.
    # Note that each car is different and in my case I found that center of ultrasonic sensor in not 0 but -6
    # Left (and right) side of the image model takes 40% and center takes 20%
    # Each (left, center, right) value of image model represent sum of distances to an object
    ###
    @staticmethod
    def get_image_model(l=-10, r=2):
        step = (r - l) // 10
        image = [0 for _ in range(3)]
        for i in range(l, l + 4 * step):
            metrics = min(99, fc.get_distance_at(i))
            metrics = metrics if metrics > 0 else 99 + metrics
            image[0] += metrics
        for i in range(l + 4 * step, r - 4 * step):
            metrics = min(99, fc.get_distance_at(i))
            metrics = metrics if metrics > 0 else 99 + metrics
            image[1] += metrics
        for i in range(r - 4 * step, r):
            metrics = min(99, fc.get_distance_at(i))
            metrics = metrics if metrics > 0 else 99 + metrics
            image[2] += metrics
        print(f"Image: {image}")
        return image

    # Main intent of the car is driving forward.
    def drive_forward(self):

        # Window is used to address situation when the car is stuck...
        # The sliding window accumulates "distance" in front of the car.
        # In case that the window contains only 3 different results we assume that the car is stuck
        win_size = 10
        window = [i for i in range(win_size)]

        i = 0
        print("Inside drive_forward")
        # Try to drive straight
        image = self.get_image_model()
        # window[i % win_size] = f"{int(image[0])}-{int(image[1])}-{int(image[2])}"
        window[i % win_size] = f"{int(image[1])}"
        while image[1] > 100:
            i += 1
            print(f"Drive forward: {image[1]}")
            fc.forward(10)
            image = self.get_image_model()
            # window[i % win_size] = f"{int(image[0])}-{int(image[1])}-{int(image[2])}"
            window[i % win_size] = f"{int(image[1])}"
            print(f"window size {len(set(window))}")
            if len(set(window)) == 3:
                print("Stop. Backup. Regroup")
                fc.stop()
                fc.backward(2)
                time.sleep(random.random() * 3)
                break

        while image[1] < 100:
            print("Drive backward")
            fc.backward(1)
            image = self.get_image_model()
        fc.stop()

    def scenario_1(self):
        # Start timer for 5 minutes. The car will work for 5 minutes
        self.run_x_minutes(5)
        # Drive until the status of the car is "stop"
        while self.carState != "stop":
            # Drive forward till found an obstacle
            self.drive_forward()
            # Get "world image" from -66 degree to 66 degree
            image = self.get_image_model(-66, 60)
            # If the car see "more obstacles" on the right side - turn left
            if image[0] < image[2]:
                print('Start turning left...')
                fc.turn_left(3)
                # Turn left until image in front of the car is "clear"
                image = self.get_image_model(-2, -1)
                while image[1] < 95:
                    print(f"Turn left. Distance {image[1]}")
                    fc.turn_left(1)
                    image = self.get_image_model(-2, -1)
            else:
                # If the car see "more obstacles" on the left side - turn right
                print('Start turning right...')
                fc.turn_right(3)
                image = self.get_image_model(-2, -1)
                while image[1] < 95:
                    print(f"Turn right. Distance {image[1]}")
                    fc.turn_right(1)
                    image = self.get_image_model(-2, -1)
            # fc.stop()


if __name__ == "__main__":
    tm = Car()
    # tm.enableCamera()
    tm.scenario_1()
    fc.stop()
