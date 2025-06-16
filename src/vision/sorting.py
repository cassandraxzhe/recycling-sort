from ultralytics import YOLO
import time

recycling_sort = YOLO("path/to/best.pt")  # load a custom model TODO: replace the path once custom model is trained

# Predict with the model
results = recycling_sort("https://ultralytics.com/images/bus.jpg")  # predict on an image # TODO: do this on an image taken from camera stream

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box


class Sorter:
    def __init__(self, model=recycling_sort):
        self.model = model
        self.queue = [] # list of Items to be sorted

        self.sort()

    def sort(self):
        while True:
            # grab frame
            frame = '' # TODO: fix this

            # run inference
            result = self.model(frame)

            # get results class, loc, and id
            # TODO

            # add any new items to queue
            self.queue.append() # TODO
            
            time.wait(1000) # TODO i forgot how to do this


class Item:
    def __init__(self, type, location, bbox):
        self.type = type
        self.location = location
        self.bbox = bbox