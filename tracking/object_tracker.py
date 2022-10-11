import numpy as np
from scipy.spatial import distance

class Tracker():
    def __init__(self, limit: list, max_disappeared: int, max_distance: int, max_history: int):
        # Object Info
        self.centroids = {}
        self.colors = {}
        self.bboxes = {}
        self.disappeared = {}
        self.counted = {}
        self.nextID = 0

        # Tracking Info
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_history = max_history

        # Counting Info
        self.limit = limit
        self.people_in = 0
        self.people_out = 0

    def add_object(self, centroid: tuple, bbox: list):
        """Add an object to track.

        Args:
            centroid (tuple): tuple with the (x,y) coordinates of the centroid
            bbox (list): list with four integer values: [left, top, right, bottom]
        """
        self.centroids[self.nextID] = [centroid]
        self.colors[self.nextID] = tuple(
            np.append(np.random.randint(255, size=(3)), 255))
        self.disappeared[self.nextID] = 0
        self.bboxes[self.nextID] = bbox
        self.counted[self.nextID] = False
        self.nextID += 1

    def remove_object(self, objectID: int):
        """Removes an object info based on its ID.

        Args:
            objectID (int): ID of the object to remove
        """
        del self.centroids[objectID]
        del self.colors[objectID]
        del self.disappeared[objectID]
        del self.counted[objectID]

    def append_centroid(self, centroid: tuple, bbox: list, objectID: int):
        """Appends a new centroid to an existing object based on its ID.

        Args:
            centroid (tuple): tuple with the (x,y) coordinates of the centroid
            bbox (list): list with four integer values: [left, top, right, bottom]
            objectID (int): ID of the object 
        """
        if len(self.centroids[objectID]) >= self.max_history:
            del self.centroids[objectID][0]
        self.centroids[objectID].append(centroid)
        self.bboxes[objectID] = bbox

    def get_min_distance(self, centroid: tuple, available_ids: list):
        """Returns the ID of the object closest to the centroid, or -1 if no object is close.

        Args:
            centroid (tuple): tuple with the (x,y) coordinates of the centroid
            available_ids (list): list with the IDs of all available objects
        """
        min_dist = self.max_distance
        min_id = -1
        for objectID in available_ids:
            last_centroid = self.centroids[objectID][-1]
            dist = distance.euclidean(list(centroid), list(last_centroid))
            if dist < min_dist:
                min_dist = dist
                min_id = objectID
        return min_id

    def update_centroids(self, detections: list):
        """Updates all centroids.

        Args:
            detections (list): list of tuples (centroid, bbox) containing all detections.
                centroid (tuple): tuple with the (x,y) coordinates of the centroid
                bbox (list): list with four integer values: [left, top, right, bottom]
        """
        available_ids = list(self.centroids.keys())
        for (centroid, bbox) in detections:
            id_to_update = self.get_min_distance(centroid, available_ids)
            if id_to_update != -1:
                self.append_centroid(centroid, bbox, id_to_update)
                available_ids.remove(id_to_update)
            else:
                self.add_object(centroid, bbox)
        if len(available_ids) > 0:
            for objectID in available_ids:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.remove_object(objectID)

    def update(self, detections: list):
        """Processes all detections, adding, updating or deleting objects.

        Args:
            detections (list): list of tuples (centroid, bbox) containing all detections.
                centroid (tuple): tuple with the (x,y) coordinates of the centroid
                bbox (list): list with four integer values: [left, top, right, bottom]
        """
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.remove_object(objectID)
        elif len(self.centroids) == 0:
            for (centroid, bbox) in detections:
                self.add_object(centroid, bbox)
        else:
            self.update_centroids(detections)

    def count_people(self):
        """Counts people going in or out.
        """
        y = self.limit[0][1]
        for centroid_key in self.centroids.keys():
            if self.counted[centroid_key] == False:
                centroid_hist = self.centroids[centroid_key]
                y_mean = np.mean([centroid[1]
                                 for centroid in centroid_hist[:-1]])
                y_last = centroid_hist[-1][1]
                diff = y_last - y_mean

                if diff > 0 and y_last > y and y_mean < y:
                    self.people_out += 1
                    self.counted[centroid_key] = True
                elif diff < 0 and y_last < y and y_mean > y:
                    self.people_in += 1
                    self.counted[centroid_key] = True
