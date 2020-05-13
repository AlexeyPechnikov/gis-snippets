Sentinel-2 composite from GEE examples with additional filtering by cloudy metadata tag and filtering by months (June-August):
```
// This example uses the Sentinel-2 QA band to cloud mask
// the collection.  The Sentinel-2 cloud flags are less
// selective, so the collection is also pre-filtered by the
// CLOUDY_PIXEL_PERCENTAGE flag, to use only relatively
// cloud-free granule.

// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
  var qa = image.select('QA60')

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

  // Return the masked and scaled data, without the QA bands.
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"])
}

// Map the function over one year of data and take the median.
// Load Sentinel-2 TOA reflectance data.
var composite = ee.ImageCollection('COPERNICUS/S2')
  .filterDate('2016-01-01', '2020-12-31')
  .filter(ee.Filter.calendarRange(6,8,'month'))
  // Pre-filter to get less cloudy granules.
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .median()

// Display the results.
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'RGB')

// Set map to Switzerland
var swiss = ee.Geometry.Rectangle(5.4, 45.5, 11, 48.1);
Map.centerObject(swiss, 8);
```

Sentinel-2 Surface Reflectance (SR) composite from GEE examples with additional filtering by cloudy metadata tag and filtering by months (June-August): use the code above replacing "COPERNICUS/S2" to "COPERNICUS/S2_SR".


Script for use on https://code.earthengine.google.com/

The code below extracted by https://www.newocr.com/ from this paper:
[Generating a cloud-free, homogeneous Landsat-8 mosaic of Switzerland using Google Earth Engine](https://www.researchgate.net/publication/302589628_Generating_a_cloud-free_homogeneous_Landsat-8_mosaic_of_Switzerland_using_Google_Earth_Engine)

![Switzerland Mosaic using Google Earth Engine](https://github.com/mobigroup/gis-snippets/blob/master/GEE/Switzerland%20Mosaic%20using%20Google%20Earth%20Engine.png)

```
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// first steps coming from GEE tutorials going to high-quality mosaicking LS8 for Switzerland: one dataset, summertime (high vegetation), origin of pixels may be either 2013, 2014 or 2015
// 1st: Input of Landsat 8 TOA data (-> l8toa), swiss centered by coordinates (-> swiss) -> rectangle
// 2nd: take only cloud-free-pixels
// 3rd: date filtering, only take different seasons - 2013 to 2015
// 4th: quality mosaic them, based on either NDVI (greenest) or NDSI (whitest, nice example for winter)
// 5th: clip with the swiss-polygon.
// 6th: as water shows some artefacts when quality mosaicked by NDVI, use watermask (water)/use threshold in NDVI, wherever water is masked use the inverse cloudiness-band as quality feature
// 7th: do some nice additional things
// 1st: This very first step is just the input of data discribed above ----------------------------------------------------------
// Landsat 8 TOA reflectances, terrain corrected.
// Switzerland centered by defining a rectangle surrounding Switzerland.
var l8toa = ee.ImageCollection('LANDSAT/LC8_L1T_TOA');
var swiss = ee.Geometry.Rectangle(5.4, 45.5, 11, 48.1);
// FIX: see page 2
var maskClouds = function(l8toa) {
var scored = ee.Algorithms.Landsat.simpleCloudScore(l8toa);
return l8toa.mask(scored.select(['cloud']).lt(25));
};
// Creating the watermask
// This function masks clouds and adds quality bands to Landsat 8 images.
var addQualityBands = function(l8toa) {
return maskClouds(l8toa)
// NDWI
.addBands(l8toa.normalizedDifference(['B3', 'B5']).rename('ndwi'))
// time in days
.addBands(l8toa.metadata('system:time_start'))
};
// spring. please adjust to data availability.
var l8toaspring2013 = l8toa.filterDate('2013-04-11', '2013-09-20');
var l8toaspring2014 = l8toa.filterDate('2014-04-21', '2014-09-20');
var l8toaspring2015 = l8toa.filterDate('2015-04-21', '2015-09-20');
var l8toaspring = ee.ImageCollection(l8toaspring2013.merge(l8toaspring2014).merge(l8toaspring2015))
.map(addQualityBands);
// Create a greenest pixel composite for each season.
var recentspring = l8toaspring.qualityMosaic('ndwi');
// Create a water mask based on NDWI values greater than 0.1 .
var ndwi = recentspring.select(['ndwi']);
function maskWater(ndwi)
{
var thresh = 0.22;
return ee.Image(1)
.and(ndwi.gte(thresh));
}
var watermask = maskWater(ndwi);

// 2nd: cloud-masking -------------------------------------------------------------------------------------------------------------------
// This function masks clouds in Landsat 8 imagery.
// Pixels showing a Cloud Score greater than 25 will be masked out.
// Cloud Score Algorithm scores cloud criteria, such as:
// brightness in blue band, brightness in all visible bands, brightness in all Infrared-bands, low temperatures in thermal band.
// furthermore Cloud Score discriminates clouds from snow by using a Snow Index (NDSI).
var maskClouds = function(l8toa) {
var scored = ee.Algorithms.Landsat.simpleCloudScore(l8toa);
return l8toa.mask(scored.select(['cloud']).lt(25));
};
// This function will add quality bands to the cloud-free images (maskClouds function from above), which will be used for quality mosaicking;
// Quality bands:
// Normalized Difference Vegetation Index (NDVI), sensitive to photosynthesis activity;
// Acquisition time (system:time_start) from metadata;
// Inverse Cloud Cover (inv_cloudiness)from metadata.
var addQualityBands = function(l8toa) {
var one = ee.Image(1);
return maskClouds(l8toa)
// NDVI
.addBands(l8toa.normalizedDifference(['B5', 'B4']).rename('ndvi'))
.addBands(l8toa.metadata('system:time_start'))
.addBands(one.divide(l8toa.metadata('CLOUD_COVER')).rename('inv_cloudiness'))
};
// Could also be done later, but if you're interested you can also try quality mosaciking based on NDWI or NDSI. See what it looks like.
// This function masks clouds and adds a NDSI band to Landsat 8 images.
var addNDSI = function(l8toa) {
return maskClouds(l8toa)
// NDSI
.addBands(l8toa.normalizedDifference(['B3', 'B6']).rename('ndsi'))
};
// This function masks clouds and adds a NDWI band to Landsat 8 images.
var addNDWI = function(l8toa) {
return maskClouds(l8toa)
// NDSI
.addBands(l8toa.normalizedDifference(['B3', 'B5']).rename('ndwi'))
};

// 3rd: date filtering ----------------------------------------------------------------------------------------------------------------------
// Create a collection of seasonal images from 2013-2015.
// 3 years, but one mosaic for the season.
// i.e. one l8toawinter actually contains winter from 3 years.
// filter yearwise, merge afterwards into 1 collection per season.
// winter, data availability for winter :-/ adjust after winter 2015/16
var l8toawinter2013 = l8toa.filterDate('2013-11-21', '2014-02-20');
var l8toawinter2014 = l8toa.filterDate('2014-11-21', '2015-02-20');
var l8toawinter2015 = l8toa.filterDate('2015-11-21', '2016-01-19'); // please adjust to data availability.
var l8toawinter = ee.ImageCollection(l8toawinter2013.merge(l8toawinter2014).merge(l8toawinter2015))
.map(addQualityBands)
.map(addNDSI)
.map(addNDWI);
// spring.
var l8toaspring2013 = l8toa.filterDate('2013-04-07', '2013-06-20'); // L8 data currently available from
04/07/2013
var l8toaspring2014 = l8toa.filterDate('2014-02-21', '2014-06-20');
var l8toaspring2015 = l8toa.filterDate('2015-02-21', '2015-06-20');
var l8toaspring = ee.ImageCollection(l8toaspring2013.merge(l8toaspring2014).merge(l8toaspring2015))
.map(addQualityBands)
.map(addNDSI)
.map(addNDWI);
// summer.
var l8toasummer2013 = l8toa.filterDate('2013-04-21', '2013-10-20');
var l8toasummer2014 = l8toa.filterDate('2014-04-21', '2014-10-20');
var l8toasummer2015 = l8toa.filterDate('2015-04-21', '2015-10-20');
var l8toasummer =
ee.ImageCollection(l8toasummer2013.merge(l8toasummer2014).merge(l8toasummer2015))
.map(addQualityBands)
.map(addNDSI)
.map(addNDWI);
// autumn.
var l8toaautumn2013 = l8toa.filterDate('2013-09-21', '2013-11-20');
var l8toaautumn2014 = l8toa.filterDate('2014-09-21', '2014-11-20');
var l8toaautumn2015 = l8toa.filterDate('2015-09-21', '2015-11-20');
var l8toaautumn =
ee.ImageCollection(l8toaautumn2013.merge(l8toaautumn2014).merge(l8toaautumn2015))
.map(addQualityBands)
.map(addNDSI)
.map(addNDWI);
// filtered for water mosaic. wider time range, but water will look the same in al images
var l8toawater2013 = l8toa.filterDate('2013-05-07', '2013-10-20');
var l8toawater2014 = l8toa.filterDate('2014-04-21', '2014-09-20');
var l8toawater2015 = l8toa.filterDate('2015-05-21', '2015-11-20');
var l8toawater = ee.ImageCollection(l8toawater2013.merge(l8toawater2014).merge(l8toawater2015))

.map(addQualityBands)
.map(addNDSI)
.map(addNDWI)
.filterMetadata('CLOUD_COVER', 'greater_than', 0.45);
// 4th: quality mosaicking --------------------------------------------------------------------------------------------------------------
// Take one band as criterion for which pixel should be taken for the Mosaic (from Image Collection to single Image)
// e.g. quality band NDVI: the pixel with the highest NDVI-value out of all used scenes will be taken
// Create a greenest pixel composite for each season.
var greenestwinter = l8toawinter.qualityMosaic('ndvi');
var greenestspring = l8toaspring.qualityMosaic('ndvi');
var greenestsummer = l8toasummer.qualityMosaic('ndvi');
var greenestautumn = l8toaautumn.qualityMosaic('ndvi');
var bluestrec = l8toawater.qualityMosaic('inv_cloudiness');
// Create a "whitest" pixel composite for winter, based on highest snow presence -> NDSI.
var whitestwinter = l8toawinter.qualityMosaic('ndsi');
// 5th clipping ------------------------------------------------------------------------------------------------------------------------------
// clip the Mosaics to some wide region around Switzerland.
var greenestCompositewinter = greenestwinter.clip(swiss);
var greenestCompositespring = greenestspring.clip(swiss);
var greenestCompositesummer = greenestsummer.clip(swiss);
var greenestCompositeautumn = greenestautumn.clip(swiss);
var whitestCompositewinter = whitestwinter.clip(swiss);
var bluestComposite = bluestrec.clip(swiss);
// give out some information about the created variables/images to the Console (see it on the right side).
print(greenestCompositespring, 'green');
print(bluestComposite, 'blue');
var ndvispring = greenestCompositespring.select('ndvi');
var ndvisummer = greenestCompositesummer.select('ndvi');
var ndviautumn = greenestCompositeautumn.select('ndvi');
var ndviwintergreen = greenestCompositewinter.select('ndvi');
var ndviwinterwhite = whitestCompositewinter.select('ndvi');
// 6th -----------------------------------------------------------------------------------------------------------------------------------------
// Composite out of greenest and bluest. bluest over water, greenest over land
var springmosaic = greenestCompositespring.where(ndvispring.lt(0.4), bluestComposite);
var summermosaic = greenestCompositesummer.where(ndvisummer.lt(0.4), bluestComposite);
var autumnmosaic = greenestCompositeautumn.where(ndviautumn.lt(0.5), bluestComposite);
var greenwintermosaic = greenestCompositewinter.where(watermask.eq(1), bluestComposite);
var whitewintermosaic = whitestCompositewinter.where(watermask.eq(1), bluestComposite);
// 7th: do more stuff, e.g. calculate other indices like EVI, or pansharpen ------------------------------------------------

var evispring = springmosaic.expression(
'2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)',
{
red: springmosaic.select('B4'), // 620-670nm, RED
nir: springmosaic.select('B5'), // 841-876nm, NIR
blue: springmosaic.select('B2') // 459-479nm, BLUE
});
// Pan Sharpening, based on panchromatic band, which has lower spectral but higher spatial resolution.
// Panchromatic bands receive the whole energy of a wider spectral window (usually blue until red light).
// As the incoming energy in a panchromatic band is higher, more effort can be done to increase the spatial resolution.
// For pan-sharpening the RGB-bands are transformed into HSV (Hue, Saturation, Value).
// As the value (blackness) band resembles well to the panchromatic band, they can be swapped.
// Then the combination of Hue, Saturation and Panchromatic are retransformed into RGB, resulting in a higher spatial and spectral resolution.
// Do this for nice visualization, better not further calculate on it ;-)
// This function will apply pansharpening to an image (img).
var pansharp = function (img) {
var rgb = img.select(['B2', 'B3', 'B4']).float();
var gray = img.select('B8').float();
// Convert to HSV, swap in the pan band, and convert back to RGB.
var huesat = rgb.rgbToHsv().select('hue', 'saturation');
return ee.Image.cat(huesat, gray).hsvToRgb();
};
// Here you apply pansharpening to the mosaics.
var panspring = pansharp(springmosaic);
var pansummer = pansharp(summermosaic);
var panautumn = pansharp(autumnmosaic);
var pangreenwinter = pansharp(greenwintermosaic);
var panwhitewinter = pansharp(whitewintermosaic);
print (panspring);
// There usually appear some NA/masked pixels. insert the unsharpened pixels where this is the case.
var panspr = springmosaic.select(['B2', 'B3', 'B4']).rename(['blue', 'green',
'red']).where(panspring.select('red').gte(0), panspring);
var pansum = summermosaic.select(['B2', 'B3', 'B4']).rename(['blue', 'green',
'red']).where(pansummer.select('red').gte(0), pansummer);
var panaut = autumnmosaic.select(['B2', 'B3', 'B4']).rename(['blue', 'green',
'red']).where(panautumn.select('red').gte(0), panautumn);
var pangwin = greenwintermosaic.select(['B2', 'B3', 'B4']).rename(['blue', 'green',
'red']).where(pangreenwinter.select('red').gte(0), pangreenwinter);
var panwwin = whitewintermosaic.select(['B2', 'B3', 'B4']).rename(['blue', 'green',
'red']).where(panwhitewinter.select('red').gte(0), panwhitewinter);
// Use Spectral Unmixing as a simple preparation of Land Cover Classification.
// Define spectral endmembers by assigning bands values.

// Feel free to add other bands, such as snow or rock for example, by exploring the pan-sharpened image's values.
// But if you do so, add the bands in the variable fractions as well (var fractions).
var urbanbare = [0.09, 0.07, 0.07];
var veg = [0.075, 0.09, 0.04];
var water = [0.057, 0.02, 0.015];
// Unmix the image.
// Use the "Inspector" to the right side, click on the image and inspect the bands.
// band_0: value resembles to probability of urban/bare land cover.
// band_1: value resembles to probability of vegetated land cover.
// band_2: value resembles to probability of water land cover.
// pan-sharpened image is only used for illustration issues.
// for more reliable results, better take the unsharpened Landsat 8 image. But then adjust the values above!
var fractions = pansum.unmix([urbanbare, veg, water]);
Map.addLayer(fractions, {}, 'unmixed');
// Some ideas for visualization.
Map.setCenter(8.5, 46.8, 8); // Switzerland centered.
var vizParams = {bands: ['B6', 'B4', 'B3'], min: 0, max: 0.4, gamma: 0.9};
var vizParams2 = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.24, gamma: 0.9};
var vizParamsPan = {bands: ['red', 'green', 'blue'], min: 0, max: 0.24, gamma: 1.1};
var ndviViz = {bands: ['ndvi'], min: -0.5, max: 1, palette: ['505050', '505050', 'E8E8E8', '00FF33', '003300']};
var ndsiViz = {bands: ['ndsi'], min: -1, max: 1, palette: ['505050', 'E8E8E8', '00FF33', '003300']};
var ndwiViz = {bands: ['ndwi'], min: -1, max: 1, palette: ['505050', 'E8E8E8', '00FF33', '003300']};
// Display the results.
Map.addLayer (springmosaic, vizParams2, 'True-Color Spring');
Map.addLayer (springmosaic, ndsiViz, 'NDSI spring');
Map.addLayer (whitewintermosaic, vizParams, 'CIR white winter');
Map.addLayer (evispring, {min: -0.5, max: 1, palette: ['505050', '505050', 'E8E8E8', '00FF33', '003300']},
'EVI');
Map.addLayer (panspr, vizParamsPan, 'True-Color Pan Spring');
Map.addLayer (pangwin, vizParamsPan, 'True-Color Pan Wintergreen');
Map.addLayer (summermosaic, vizParams2, 'True-Color Summer ');
Map.addLayer (pansum, vizParamsPan, 'True-Color Pan Summer');
// Export
//Export.image (pansum, 'Pansharpened_Summer', {region: swiss, maxPixels: 1206250905});
```
