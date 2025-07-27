import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import laspy

class LoaderFactory:
    
    def detect_file_type(self, file_path):
        suffix = Path(file_path).suffix.lower()
        type_map = {
            '.tif': 'dem_tiff',
            '.tiff': 'dem_tiff', 
            '.las': 'lidar_las',
            '.ply': 'pointcloud_ply',
            '.csv': 'csv_xyz',
            '.txt': 'csv_xyz'
        }
        
        if suffix not in type_map:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return type_map[suffix]
    
    def load_dem_tiff(self, file_path):
        dem = tifffile.imread(file_path)
        nodata_values = [-3.4028226550889045e+38, -9999, np.nan]
        
        for nodata in nodata_values:
            if not np.isnan(nodata):
                dem = np.where(dem != nodata, dem, np.nan)
        
        return dem
    
    def load_csv_xyz(self, file_path):
        try:
            df = pd.read_csv(file_path)
            
            if len(df.columns) < 3:
                raise ValueError("CSV must have at least 3 columns (X, Y, Z)")
            
            xyz_cols = []
            for col in df.columns[:3]:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    xyz_cols.append(col)
            
            if len(xyz_cols) < 3:
                xyz_cols = df.columns[:3].tolist()
            
            points = df[xyz_cols].values.astype(np.float32)
            points = points[~np.isnan(points).any(axis=1)]
            
            return points
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    def load_pointcloud_ply(self, file_path):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(file_path))
            points = np.asarray(pcd.points).astype(np.float32)
            
            if len(points) == 0:
                raise ValueError("Empty point cloud")
            
            return points
            
        except ImportError:
            raise ValueError("open3d required for PLY files")
        except Exception as e:
            raise ValueError(f"Failed to load PLY: {str(e)}")
    
    def load_lidar_las(self, file_path):
        try:
            las_file = laspy.read(file_path)
            
            x = las_file.x.copy()
            y = las_file.y.copy() 
            z = las_file.z.copy()
            
            points = np.column_stack([x, y, z]).astype(np.float32)
            points = points[~np.isnan(points).any(axis=1)]
            
            return points
            
        except ImportError:
            raise ValueError("laspy required for LAS files")
        except Exception as e:
            raise ValueError(f"Failed to load LAS: {str(e)}")
    
    def dem_to_pointcloud(self, dem, skip=20):
        if dem.ndim != 2:
            raise ValueError("DEM must be 2D array")
        
        y_idx, x_idx = np.mgrid[0:dem.shape[0]:skip, 0:dem.shape[1]:skip]
        elevations = dem[::skip, ::skip].astype(np.float32)
        points = np.stack([x_idx.flatten(), y_idx.flatten(), elevations.flatten()], axis=1)
        mask = ~np.isnan(points[:,2])
        
        if mask.sum() == 0:
            raise ValueError("No valid elevation data")
        
        return points[mask]
    
    def load_any_format(self, file_path, skip=20):
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'dem_tiff':
            dem = self.load_dem_tiff(file_path)
            points = self.dem_to_pointcloud(dem, skip)
        elif file_type == 'csv_xyz':
            points = self.load_csv_xyz(file_path)
        elif file_type == 'pointcloud_ply':
            points = self.load_pointcloud_ply(file_path)
        elif file_type == 'lidar_las':
            points = self.load_lidar_las(file_path)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        if len(points) < 100:
            raise ValueError(f"Dataset too small: {len(points)} points")
        
        return points
