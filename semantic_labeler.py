import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class SemanticLabeler:
    
    def __init__(self):
        self.domain_models = {}
        self.scalers = {}
        self.audio_mappings = {
            'crater': 'deep_reverb',
            'ridge': 'sharp_metallic', 
            'plain': 'soft_ambient',
            'peak': 'bright_chime',
            'valley': 'low_rumble',
            'slope': 'ascending_tone'
        }
    
    def train_terrain_classifier(self, points, clusters, surface_types):
        features = self._extract_terrain_features(points, clusters, surface_types)
        labels = self._generate_terrain_labels(points, clusters)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features_scaled, labels)
        
        self.domain_models['terrain'] = model
        self.scalers['terrain'] = scaler
        
        return model
    
    def _extract_terrain_features(self, points, clusters, surface_types):
        features = []
        
        for i, point in enumerate(points):
            cluster_id = clusters[i]
            surface_type = surface_types[i]
            
            cluster_points = points[clusters == cluster_id]
            
            feature_vector = [
                point[2],
                np.mean(cluster_points[:, 2]),
                np.std(cluster_points[:, 2]),
                len(cluster_points),
                1 if surface_type == 'flat' else 0,
                1 if surface_type == 'curved' else 0,
                1 if surface_type == 'edge' else 0,
                self._calculate_local_roughness(i, points),
                self._calculate_slope(i, points)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _generate_terrain_labels(self, points, clusters):
        labels = []
        
        for i, point in enumerate(points):
            cluster_id = clusters[i]
            cluster_points = points[clusters == cluster_id]
            
            elevation = point[2]
            cluster_mean_elevation = np.mean(cluster_points[:, 2])
            cluster_elevation_std = np.std(cluster_points[:, 2])
            
            if elevation < cluster_mean_elevation - cluster_elevation_std:
                if cluster_elevation_std > 50:
                    labels.append('crater')
                else:
                    labels.append('valley')
            elif elevation > cluster_mean_elevation + cluster_elevation_std:
                if cluster_elevation_std > 50:
                    labels.append('peak')
                else:
                    labels.append('ridge')
            elif cluster_elevation_std < 20:
                labels.append('plain')
            else:
                labels.append('slope')
        
        return labels
    
    def _calculate_local_roughness(self, point_idx, points):
        if point_idx == 0 or point_idx >= len(points) - 1:
            return 0
        
        local_points = points[max(0, point_idx-5):min(len(points), point_idx+6)]
        if len(local_points) < 3:
            return 0
        
        elevations = local_points[:, 2]
        return np.std(elevations)
    
    def _calculate_slope(self, point_idx, points):
        if point_idx == 0 or point_idx >= len(points) - 1:
            return 0
        
        prev_point = points[point_idx - 1]
        next_point = points[point_idx + 1]
        
        horizontal_distance = np.sqrt((next_point[0] - prev_point[0])**2 + 
                                    (next_point[1] - prev_point[1])**2)
        vertical_distance = next_point[2] - prev_point[2]
        
        if horizontal_distance == 0:
            return 0
        
        return abs(vertical_distance / horizontal_distance)
    
    def apply_semantic_labels(self, points, clusters, surface_types, domain='terrain'):
        if domain not in self.domain_models:
            self.train_terrain_classifier(points, clusters, surface_types)
        
        model = self.domain_models[domain]
        scaler = self.scalers[domain]
        
        features = self._extract_terrain_features(points, clusters, surface_types)
        features_scaled = scaler.transform(features)
        
        labels = model.predict(features_scaled)
        
        audio_mappings = [self.audio_mappings.get(label, 'default') for label in labels]
        
        return {
            'labels': labels,
            'audio_mappings': audio_mappings
        }
