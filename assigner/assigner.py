from sklearn.cluster import KMeans

class Assigner:
    def __init__(self):
        # Initialize dictionaries to store beyblade colors and assignments
        self.beyblade_colors = {}
        self.beyblade_assigner_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image into a 2D array for clustering
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_beyblade_color(self, frame, bbox):
        # Extract the region of interest from the frame using bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Get K-means clustering model for the region of interest
        kmeans = self.get_clustering_model(image)

        # Retrieve cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to match the original image shape
        clustered_image = labels.reshape(image.shape[0], image.shape[1])

        # Identify the non-beyblade cluster based on corner pixels
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_beyblade_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Assign beyblade cluster to the opposite of non-beyblade cluster
        beyblade_cluster = 1 - non_beyblade_cluster

        # Retrieve the beyblade's dominant color from the cluster centers
        beyblade_color = kmeans.cluster_centers_[beyblade_cluster]

        return beyblade_color

    def assign_beyblade_color(self, frame, beyblade_detections):
        # List to store detected beyblade colors
        beyblade_colors = []
        for _, beyblade_detection in beyblade_detections.items():
            bbox = beyblade_detection["bbox"]
            beyblade_color = self.get_beyblade_color(frame, bbox)
            beyblade_colors.append(beyblade_color)

        # Perform K-means clustering on the detected beyblade colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(beyblade_colors)

        # Store the clustering model and assign cluster centers to beyblades
        self.kmeans = kmeans
        self.beyblade_colors[1] = kmeans.cluster_centers_[0]
        self.beyblade_colors[2] = kmeans.cluster_centers_[1]

    def get_beyblade_team(self, frame, beyblade_bbox, beyblade_id):
        # Check if beyblade has already been assigned a team
        if beyblade_id in self.beyblade_assigner_dict:
            return self.beyblade_assigner_dict[beyblade_id]

        # Get the beyblade's color from the bounding box
        beyblade_color = self.get_beyblade_color(frame, beyblade_bbox)

        # Predict which team the beyblade belongs to using K-means
        beyblade_id = self.kmeans.predict(beyblade_color.reshape(1, -1))[0]
        beyblade_id += 1

        # Store the assignment of beyblade to a specific team
        self.beyblade_assigner_dict[beyblade_id] = beyblade_id

        return beyblade_id
