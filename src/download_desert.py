import math
import os
import time
import io
from pathlib import Path
import requests
from PIL import Image

def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon coordinates to slippy map tile x/y coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def main():
    zoom = 14
    # Bounding box in Egypt's Western Desert (deep sand/dunes region, completely arid)
    # Latitude: 27.0 to 27.5, Longitude: 27.0 to 27.5
    lat_min, lat_max = 27.0, 27.5
    lon_min, lon_max = 27.0, 27.5

    # Get tile coordinate ranges
    x_min, y_max = latlon_to_tile(lat_min, lon_min, zoom)
    x_max, y_min = latlon_to_tile(lat_max, lon_max, zoom)

    # Ensure correct ordering
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)

    print(f"[INFO] Bounding Box Lat: [{lat_min}, {lat_max}], Lon: [{lon_min}, {lon_max}]")
    print(f"[INFO] Zoom level: {zoom}")
    print(f"[INFO] Tile range: X [{x_start} to {x_end}], Y [{y_start} to {y_end}]")
    
    total_available_tiles = (x_end - x_start + 1) * (y_end - y_start + 1)
    print(f"[INFO] Total available tiles in range: {total_available_tiles}")
    
    target_patches = 4000
    target_tiles = math.ceil(target_patches / 16)
    print(f"[INFO] Target tiles to download: {target_tiles} (to yield {target_tiles * 16} patches)")

    output_dir = Path("data/raw/Desert")
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    # Headers to mimic a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    downloaded_tiles = 0
    patches_saved = 0
    
    # Iterate through the grid
    # We will step through tiles to cover a wide area rather than just a tight cluster
    x_step = max(1, (x_end - x_start) // int(math.sqrt(target_tiles)))
    y_step = max(1, (y_end - y_start) // int(math.sqrt(target_tiles)))
    
    print(f"[INFO] Scanning grid with X-step: {x_step}, Y-step: {y_step}")

    for x in range(x_start, x_end + 1, x_step):
        for y in range(y_start, y_end + 1, y_step):
            if downloaded_tiles >= target_tiles:
                break
                
            # ESRI World Imagery URL template:
            # https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
            # Note: ArcGIS uses standard slippy map y/x coordinate ordering in its path: MapServer/tile/{z}/{y}/{x}
            url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    img_data = response.content
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.size == (256, 256):
                        # Slice tile into 16 64x64 patches
                        patch_idx = 0
                        for row in range(4):
                            for col in range(4):
                                left = col * 64
                                top = row * 64
                                right = left + 64
                                bottom = top + 64
                                
                                patch = img.crop((left, top, right, bottom))
                                # Save patch
                                patch_name = f"desert_tile_{x}_{y}_patch_{patch_idx}.jpg"
                                patch.convert("RGB").save(output_dir / patch_name, "JPEG", quality=95)
                                patch_idx += 1
                                patches_saved += 1
                                
                        downloaded_tiles += 1
                        if downloaded_tiles % 20 == 0:
                            print(f"[INFO] Downloaded {downloaded_tiles}/{target_tiles} tiles. Patches saved: {patches_saved}")
                    else:
                        print(f"[WARNING] Tile {x},{y} had unexpected size: {img.size}")
                else:
                    print(f"[WARNING] Failed to fetch tile {x},{y}: Status {response.status_code}")
            except Exception as e:
                print(f"[ERROR] Exception fetching tile {x},{y}: {e}")
                
            # Polite rate limiting
            time.sleep(0.05)
            
        if downloaded_tiles >= target_tiles:
            break

    print(f"\n[INFO] Desert supplementation complete!")
    print(f"Total tiles downloaded: {downloaded_tiles}")
    print(f"Total desert patches saved: {patches_saved}")

if __name__ == "__main__":
    main()
