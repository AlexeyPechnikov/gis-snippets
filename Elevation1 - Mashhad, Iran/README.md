## Elevation1 - Mashhad, Iran

Build Super-resolution DEM 0.5m from DEM 1m enhanced by one orthophoto image 0.5m

Source dataset: Elevation1 DSM + Pléiades Ortho 0.5m pan-sharpened (Orthoimage included)

https://www.intelligence-airbusds.com/en/9317-sample-imagery-detail?product=18896&keyword=&type=366

## Prepare datasets

```
gdalwarp -te 730000 4011500 730500 4012000 MMashhad-DEM/ashhad-DEM.TIF Mashhad-DEM.sample.tif
gdalwarp -te 730000 4011500 730500 4012000 ORTHO_RGB/7289-40126_Mashhad.tif 7289-40126_Mashhad.sample.tif
```

## Output plots

![Super-resolution DEM](Super-resolution%20DEM.jpg)
