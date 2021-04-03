/**
 * Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image input Landsat 8 SR image
 * @return {ee.Image} cloudmasked Landsat 8 image
 */
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

var area = ee.Geometry.Rectangle([116.789084,  -9.031312, 117.291524,  -8.632405]);

var dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
                  .filter(ee.Filter.lt('CLOUD_COVER', 20))
                  .map(maskL8sr)
                  .filterBounds(area);
print ('dataset', dataset.size())

var visParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.18,
  gamma: 1.4,
};

Map.centerObject(area, 9);
Map.addLayer(dataset.median(), visParams);
