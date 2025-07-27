import numpy as np
import pandas as pd
import tifffile
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from loader_factory import LoaderFactory
import warnings

class DataProcessor:
    
    def __init__(self):
        self.loader = LoaderFactory()
        self.warnings_issued = []
    
    def _issue_warning(self, message):
        """Issue warning and store for user feedback"""
        self.warnings_issued.append(message)
        warnings.warn(message, UserWarning)
        print(f"⚠️ {message}")
    
    def load_tiff_dem(self, file_path):
        try:
            dem = tifffile.imread(file_path)
            
            # Handle different nodata value formats more robustly
            nodata_values = [
                -3.4028226550889045e+38,
                np.float32(-3.4028226550889045e+38),
                -9999,
                np.nan
            ]
            
            # Convert to float32 first to avoid casting issues
            original_dtype = dem.dtype
            if dem.dtype != np.float32:
                dem = dem.astype(np.float32)
                self._issue_warning(f"DEM data type converted from {original_dtype} to float32")
            
            # Replace nodata values
            nodata_replaced = 0
            for nodata in nodata_values:
                if not np.isnan(nodata):
                    try:
                        nodata_f32 = np.float32(nodata)
                        mask = np.isclose(dem, nodata_f32, rtol=1e-6)
                        nodata_replaced += np.sum(mask)
                        dem = np.where(mask, np.nan, dem)
                    except (ValueError, OverflowError):
                        continue
            
            if nodata_replaced > 0:
                self._issue_warning(f"Replaced {nodata_replaced} nodata values with NaN")
            
            # Validate DEM dimensions
            if dem.ndim not in [2, 3]:
                if dem.ndim > 3:
                    dem = dem[:, :, 0] if dem.shape[2] > 0 else dem.reshape(dem.shape[:2])
                    self._issue_warning("Multi-band TIFF detected, using first band only")
                else:
                    raise ValueError(f"Invalid DEM dimensions: {dem.shape}")
            
            if dem.ndim == 3:
                dem = dem[:, :, 0]
                self._issue_warning("3D TIFF detected, using first band only")
            
            return dem
            
        except Exception as e:
            raise ValueError(f"Failed to load TIFF file: {str(e)}")
    
    def dem_to_pointcloud(self, dem, skip=20):
        if dem is None or dem.size == 0:
            raise ValueError("Empty DEM data")
        
        # Ensure DEM is 2D
        if dem.ndim != 2:
            raise ValueError(f"DEM must be 2D array, got {dem.ndim}D")
        
        original_shape = dem.shape
        
        try:
            y_idx, x_idx = np.mgrid[0:dem.shape[0]:skip, 0:dem.shape[1]:skip]
            elevations = dem[::skip, ::skip].astype(np.float32)
            
            # Check for shape consistency before stacking
            if not (x_idx.shape == y_idx.shape == elevations.shape):
                # Find minimal common shape
                min_rows = min(x_idx.shape[0], y_idx.shape[0], elevations.shape[0])
                min_cols = min(x_idx.shape[1], y_idx.shape[1], elevations.shape[1])
                
                x_idx = x_idx[:min_rows, :min_cols]
                y_idx = y_idx[:min_rows, :min_cols]
                elevations = elevations[:min_rows, :min_cols]
                
                self._issue_warning(f"Array shapes were inconsistent, auto-corrected to {(min_rows, min_cols)}")
            
            # Flatten and stack arrays
            points = np.stack([x_idx.flatten(), y_idx.flatten(), elevations.flatten()], axis=1)
            
            # Remove invalid points
            mask = ~np.isnan(points[:, 2])
            invalid_count = len(points) - mask.sum()
            
            if invalid_count > 0:
                self._issue_warning(f"Removed {invalid_count} points with invalid elevation data")
            
            if mask.sum() == 0:
                raise ValueError("No valid elevation data found after cleaning")
            
            valid_points = points[mask]
            
            # Additional validation
            if len(valid_points) < 100:
                self._issue_warning(f"Very few valid points ({len(valid_points)}) - results may be unreliable")
            
            return valid_points
            
        except Exception as e:
            raise ValueError(f"Failed to convert DEM to point cloud: {str(e)}")
    
    def _validate_and_clean_points(self, points, source_format="unknown"):
        """Validate and clean point cloud data from any source"""
        if points is None or len(points) == 0:
            raise ValueError(f"No points loaded from {source_format} file")
        
        original_count = len(points)
        
        # Ensure points is a numpy array
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Ensure 2D array with 3 columns minimum
        if points.ndim == 1:
            if len(points) >= 3:
                points = points.reshape(1, -1)
            else:
                raise ValueError(f"Invalid point data format in {source_format}")
        
        if points.shape[1] < 3:
            raise ValueError(f"{source_format} must have at least 3 columns (X, Y, Z)")
        
        # Take only first 3 columns if more exist
        if points.shape[1] > 3:
            points = points[:, :3]
            self._issue_warning(f"Using first 3 columns from {source_format} file (X, Y, Z)")
        
        # Remove rows with any NaN or infinite values
        valid_mask = np.isfinite(points).all(axis=1)
        invalid_count = original_count - valid_mask.sum()
        
        if invalid_count > 0:
            points = points[valid_mask]
            self._issue_warning(f"Removed {invalid_count} rows with invalid/missing data from {source_format}")
        
        # Ensure float32 dtype
        if points.dtype != np.float32:
            points = points.astype(np.float32)
        
        # Final validation
        if len(points) == 0:
            raise ValueError(f"No valid points remain after cleaning {source_format} data")
        
        if len(points) < 10:
            self._issue_warning(f"Very few points ({len(points)}) after cleaning - processing may be unreliable")
        
        # Check for reasonable coordinate ranges
        for i, coord in enumerate(['X', 'Y', 'Z']):
            coord_range = np.ptp(points[:, i])
            if coord_range == 0:
                self._issue_warning(f"All {coord} coordinates are identical - may indicate data issues")
            elif coord_range > 1e6:
                self._issue_warning(f"{coord} coordinate range is very large ({coord_range:.0f}) - consider coordinate system")
        
        return points
    
    def perform_clustering(self, points, eps=75, min_samples=8):
        try:
            points = self._validate_and_clean_points(points, "clustering input")
            
            if len(points) < min_samples:
                elevations = points[:, 2]
                n_clusters = min(12, max(6, len(points) // 500))
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(points)
                self._issue_warning(f"Used K-means clustering due to insufficient points for DBSCAN")
                return labels
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = np.sum(labels == -1)
            
            if noise_points > len(points) * 0.5:
                self._issue_warning(f"High noise ratio in clustering ({noise_points}/{len(points)} points)")
            
            if n_clusters_found < 6:
                elevations = points[:, 2]
                x_coords = points[:, 0] 
                y_coords = points[:, 1]
                
                # Safe normalization
                elev_range = np.ptp(elevations)
                x_range = np.ptp(x_coords)
                y_range = np.ptp(y_coords)
                
                if elev_range == 0:
                    elev_norm = np.zeros_like(elevations)
                else:
                    elev_norm = (elevations - elevations.min()) / elev_range
                
                if x_range == 0:
                    x_norm = np.zeros_like(x_coords)
                else:
                    x_norm = (x_coords - x_coords.min()) / x_range
                
                if y_range == 0:
                    y_norm = np.zeros_like(y_coords)
                else:
                    y_norm = (y_coords - y_coords.min()) / y_range
                
                features = np.column_stack([x_norm * 0.3, y_norm * 0.3, elev_norm * 0.4])
                
                n_clusters = min(15, max(8, len(points) // 800))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                self._issue_warning(f"Switched to K-means clustering for better feature separation ({n_clusters} clusters)")
            
            return labels
            
        except Exception as e:
            self._issue_warning(f"Clustering failed, using single cluster: {str(e)}")
            return np.zeros(len(points), dtype=int)

    def compute_normals(self, points):
        try:
            points = self._validate_and_clean_points(points, "normal computation")
            
            if len(points) < 10:
                self._issue_warning("Too few points for normal computation, returning zero normals")
                return np.zeros((len(points), 3))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Adaptive radius based on point cloud size
            bbox = pcd.get_axis_aligned_bounding_box()
            diagonal = np.linalg.norm(bbox.max_bound - bbox.min_bound)
            adaptive_radius = diagonal / 100
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=adaptive_radius, max_nn=30
                )
            )
            
            normals = np.asarray(pcd.normals)
            
            # Validate normals
            valid_normals = np.isfinite(normals).all(axis=1)
            if not valid_normals.all():
                invalid_count = len(normals) - valid_normals.sum()
                self._issue_warning(f"Fixed {invalid_count} invalid normal vectors")
                normals[~valid_normals] = [0, 0, 1]  # Default to upward normal
            
            return normals
            
        except Exception as e:
            self._issue_warning(f"Normal computation failed: {str(e)}")
            return np.zeros((len(points), 3))
    
    def classify_surface_types(self, points, normals):
        try:
            if len(normals) == 0 or len(points) != len(normals):
                self._issue_warning("Normals and points count mismatch, defaulting to 'unknown'")
                return ['unknown'] * len(points)
            
            surface_types = []
            for i, normal in enumerate(normals):
                normal_magnitude = np.linalg.norm(normal)
                if normal_magnitude == 0 or not np.isfinite(normal_magnitude):
                    surface_types.append('unknown')
                    continue
                    
                normalized_normal = normal / normal_magnitude
                
                if abs(normalized_normal[2]) > 0.8:
                    surface_types.append('flat')
                elif self._detect_sharp_edge(i, normals):
                    surface_types.append('edge')
                else:
                    surface_types.append('curved')
            
            return surface_types
            
        except Exception as e:
            self._issue_warning(f"Surface classification failed: {str(e)}")
            return ['unknown'] * len(points)
    
    def _detect_sharp_edge(self, point_idx, normals):
        if point_idx == 0 or point_idx >= len(normals) - 1:
            return False
        
        try:
            current_normal = normals[point_idx]
            prev_normal = normals[point_idx - 1]
            next_normal = normals[point_idx + 1]
            
            # Check for zero or invalid normals
            for normal in [current_normal, prev_normal, next_normal]:
                if np.linalg.norm(normal) == 0 or not np.isfinite(normal).all():
                    return False
            
            angle_change_prev = np.arccos(np.clip(
                np.dot(current_normal, prev_normal) / 
                (np.linalg.norm(current_normal) * np.linalg.norm(prev_normal)), 
                -1, 1
            ))
            angle_change_next = np.arccos(np.clip(
                np.dot(current_normal, next_normal) / 
                (np.linalg.norm(current_normal) * np.linalg.norm(next_normal)), 
                -1, 1
            ))
            
            return angle_change_prev > np.pi/4 or angle_change_next > np.pi/4
            
        except Exception:
            return False
    
    def detect_geometric_features(self, points, normals):
        try:
            if len(points) < 10:
                return {'planes': [], 'edges': [], 'corners': []}
            
            features = {
                'planes': self._detect_planes(points, normals),
                'edges': self._detect_edges(points, normals),
                'corners': self._detect_corners(points, normals)
            }
            return features
            
        except Exception as e:
            self._issue_warning(f"Geometric feature detection failed: {str(e)}")
            return {'planes': [], 'edges': [], 'corners': []}
    
    def _detect_planes(self, points, normals):
        planes = []
        if len(normals) == 0:
            return planes
        
        try:
            for i, normal in enumerate(normals):
                if (np.linalg.norm(normal) > 0 and 
                    np.isfinite(normal).all() and 
                    abs(normal[2] / np.linalg.norm(normal)) > 0.9):
                    planes.append({'point_idx': i, 'normal': normal, 'position': points[i]})
            return planes
        except Exception:
            return []
    
    def _detect_edges(self, points, normals):
        edges = []
        if len(normals) < 3:
            return edges
        
        try:
            for i in range(1, len(normals) - 1):
                if self._detect_sharp_edge(i, normals):
                    edges.append({'point_idx': i, 'position': points[i]})
            return edges
        except Exception:
            return []
    
    def _detect_corners(self, points, normals):
        corners = []
        if len(points) < 10:
            return corners
        
        try:
            nbrs = NearestNeighbors(n_neighbors=min(10, len(points)), algorithm='auto').fit(points)
            
            for i, point in enumerate(points):
                distances, indices = nbrs.kneighbors([point])
                neighbor_points = points[indices[0][1:]]
                
                if len(neighbor_points) > 5:
                    center = np.mean(neighbor_points, axis=0)
                    deviations = np.linalg.norm(neighbor_points - center, axis=1)
                    if (np.std(deviations) > np.mean(deviations) * 0.5 and 
                        np.isfinite(deviations).all()):
                        corners.append({'point_idx': i, 'position': point})
            
            return corners
        except Exception:
            return []
    
    def detect_anomalies(self, points, contamination=0.01):
        try:
            points = self._validate_and_clean_points(points, "anomaly detection")
            
            if len(points) < 10:
                return np.zeros(len(points), dtype=bool)
            
            # Adjust contamination for small datasets
            actual_contamination = min(contamination, 0.1)
            if len(points) < 100:
                actual_contamination = min(contamination, 2.0/len(points))
            
            anomaly_detector = IsolationForest(
                contamination=actual_contamination, 
                random_state=42,
                n_estimators=50
            )
            anomalies = anomaly_detector.fit_predict(points) == -1
            
            return anomalies
            
        except Exception as e:
            self._issue_warning(f"Anomaly detection failed: {str(e)}")
            return np.zeros(len(points), dtype=bool)
    
    def process_dataset(self, file_path, skip=20):
        # Clear previous warnings
        self.warnings_issued = []
        
        try:
            file_type = self.loader.detect_file_type(file_path)
            
            if file_type == 'dem_tiff':
                dem = self.load_tiff_dem(file_path)
                points = self.dem_to_pointcloud(dem, skip)
            else:
                points = self.loader.load_any_format(file_path, skip)
                points = self._validate_and_clean_points(points, file_type)
            
            clusters = self.perform_clustering(points)
            normals = self.compute_normals(points)
            surface_types = self.classify_surface_types(points, normals)
            geometric_features = self.detect_geometric_features(points, normals)
            anomalies = self.detect_anomalies(points)
            
            result = {
                'points': points,
                'clusters': clusters,
                'normals': normals,
                'surface_types': surface_types,
                'geometric_features': geometric_features,
                'anomalies': anomalies,
                'warnings': self.warnings_issued.copy(),
                'stats': {
                    'n_points': len(points),
                    'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                    'n_anomalies': anomalies.sum(),
                    'elevation_range': np.ptp(points[:,2]),
                    'n_planes': len(geometric_features['planes']),
                    'n_edges': len(geometric_features['edges']),
                    'n_corners': len(geometric_features['corners']),
                    'warnings_count': len(self.warnings_issued)
                }
            }
            
            return result
            
        except Exception as e:
            # If everything fails, try to provide partial results
            error_msg = f"Dataset processing failed: {str(e)}"
            self._issue_warning(error_msg)
            raise ValueError(error_msg)
    
    def process_large_dataset(self, file_path, skip=20, max_points=500000):
        # Clear previous warnings
        self.warnings_issued = []
        
        try:
            file_type = self.loader.detect_file_type(file_path)
            
            if file_type == 'dem_tiff':
                dem = self.load_tiff_dem(file_path)
                
                total_potential_points = (dem.shape[0] // skip) * (dem.shape[1] // skip)
                if total_potential_points > max_points:
                    new_skip = int(np.sqrt(dem.shape[0] * dem.shape[1] / max_points))
                    skip = max(skip, new_skip)
                    self._issue_warning(f"Increased skip to {skip} for memory management")
                
                points = self.dem_to_pointcloud(dem, skip)
            else:
                points = self.loader.load_any_format(file_path, skip)
                points = self._validate_and_clean_points(points, file_type)
                
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                    self._issue_warning(f"Randomly sampled {max_points} points from {len(points)} total")
            
            print(f"Processing {len(points):,} points")
            
            clusters = self.perform_clustering(points)
            normals = self.compute_normals(points)
            surface_types = self.classify_surface_types(points, normals)
            geometric_features = self.detect_geometric_features(points, normals)
            anomalies = self.detect_anomalies(points)
            
            result = {
                'points': points,
                'clusters': clusters,
                'normals': normals,
                'surface_types': surface_types,
                'geometric_features': geometric_features,
                'anomalies': anomalies,
                'warnings': self.warnings_issued.copy(),
                'stats': {
                    'n_points': len(points),
                    'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                    'n_anomalies': anomalies.sum(),
                    'elevation_range': np.ptp(points[:,2]),
                    'n_planes': len(geometric_features['planes']),
                    'n_edges': len(geometric_features['edges']),
                    'n_corners': len(geometric_features['corners']),
                    'downsampled': True,
                    'original_skip': skip,
                    'warnings_count': len(self.warnings_issued)
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Large dataset processing failed: {str(e)}"
            self._issue_warning(error_msg)
            raise ValueError(error_msg)
