from sklearn.cluster import KMeans

class Assigner:
    def __init__(self):
        self.beyblade_colors = {}
        self.beyblade_assigner_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_beyblade_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Get Clustering model
        kmeans = self.get_clustering_model(image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(image.shape[0],image.shape[1])

        # Get the beyblade cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_beyblade_cluster = max(set(corner_clusters),key=corner_clusters.count)
        beyblade_cluster = 1 - non_beyblade_cluster

        beyblade_color = kmeans.cluster_centers_[beyblade_cluster]

        return beyblade_color


    def assign_beyblade_color(self,frame, beyblade_detections):
        
        beyblade_colors = []
        for _, beyblade_detection in beyblade_detections.items():
            bbox = beyblade_detection["bbox"]
            beyblade_color =  self.get_beyblade_color(frame,bbox)
            beyblade_colors.append(beyblade_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(beyblade_colors)

        self.kmeans = kmeans

        self.beyblade_colors[1] = kmeans.cluster_centers_[0]
        self.beyblade_colors[2] = kmeans.cluster_centers_[1]


    def get_beyblade_team(self,frame,beyblade_bbox,beyblade_id):
        if beyblade_id in self.beyblade_assigner_dict:
            return self.beyblade_assigner_dict[beyblade_id]

        beyblade_color = self.get_beyblade_color(frame,beyblade_bbox)

        beyblade_id = self.kmeans.predict(beyblade_color.reshape(1,-1))[0]
        beyblade_id+=1

        self.beyblade_assigner_dict[beyblade_id] = beyblade_id

        return beyblade_id