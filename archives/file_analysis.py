import rasterio
import numpy as np

def analyze_deforestation_map(tif_path):
    """
    Analyze the Acre Adjusted Density Map to calculate deforestation probabilities
    
    Parameters:
    tif_path: path to the adjusted density map (GeoTIFF)
    """
    
    with rasterio.open(tif_path) as src:
        density_map = src.read(1)
        
        transform = src.transform
        pixel_width = abs(transform[0])  # in map units (usually meters)
        pixel_height = abs(transform[4])  # in map units (usually meters)
        
        pixel_area_m2 = pixel_width * pixel_height
        pixel_area_ha = pixel_area_m2 / 10000  # convert m² to hectares
        
        nodata = src.nodata
        
        print("="*60)
        print("RASTER INFORMATION")
        print("="*60)
        print(f"Spatial Resolution: {pixel_width}m x {pixel_height}m")
        print(f"Pixel Area: {pixel_area_ha:.6f} hectares")
        print(f"Map Dimensions: {src.width} cols x {src.height} rows")
        print(f"NoData Value: {nodata}")
        print(f"CRS: {src.crs}")
        
        if nodata is not None:
            valid_mask = (density_map != nodata) & (density_map > 0)
        else:
            valid_mask = density_map > 0
        
        valid_densities = density_map[valid_mask]
        
        if len(valid_densities) == 0:
            print("\nWARNING: No valid density values found!")
            return
        
        min_density = np.min(valid_densities)
        max_density = np.max(valid_densities)
        mean_density = np.mean(valid_densities)
        median_density = np.median(valid_densities)
        
        min_probability = (min_density / pixel_area_ha) * 100
        max_probability = (max_density / pixel_area_ha) * 100
        mean_probability = (mean_density / pixel_area_ha) * 100
        median_probability = (median_density / pixel_area_ha) * 100
        
        print("\n" + "="*60)
        print("DEFORESTATION DENSITY STATISTICS (ha/pixel/year)")
        print("="*60)
        print(f"Minimum Density: {min_density:.8f} ha/pixel/year")
        print(f"Maximum Density: {max_density:.8f} ha/pixel/year")
        print(f"Mean Density: {mean_density:.8f} ha/pixel/year")
        print(f"Median Density: {median_density:.8f} ha/pixel/year")
        
        print("\n" + "="*60)
        print("DEFORESTATION PROBABILITY (% of pixel area)")
        print("="*60)
        print(f"Minimum Probability: {min_probability:.4f}%")
        print(f"Maximum Probability: {max_probability:.4f}%")
        print(f"Mean Probability: {mean_probability:.4f}%")
        print(f"Median Probability: {median_probability:.4f}%")
        
        if max_density > pixel_area_ha:
            print(f"\n⚠️  WARNING: Maximum density ({max_density:.6f}) exceeds pixel area ({pixel_area_ha:.6f})!")
            print("This should not happen after proper quantity adjustment.")
        
        total_deforestation_ha = np.sum(valid_densities)
        print("\n" + "="*60)
        print("TOTAL PREDICTED DEFORESTATION")
        print("="*60)
        print(f"Total: {total_deforestation_ha:.2f} hectares/year")
        print(f"Total: {total_deforestation_ha/10000:.2f} km²/year")
        
        print("\n" + "="*60)
        print("PERCENTILE ANALYSIS")
        print("="*60)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            density_p = np.percentile(valid_densities, p)
            prob_p = (density_p / pixel_area_ha) * 100
            print(f"{p}th percentile: {density_p:.8f} ha/pixel ({prob_p:.4f}%)")

if __name__ == "__main__":
    tif_path = "./outputs/predictions/Acre_Adjucted_Density_Map_VP.tif"
    
    analyze_deforestation_map(tif_path)