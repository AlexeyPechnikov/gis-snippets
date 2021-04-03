var area = ee.Geometry.Rectangle(116.7, -9.1, 117.8, -8.4);
// Make a cloud-free Landsat 8 TOA composite (from raw imagery).
var collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
  .filterBounds(area);
print ("count", collection.size());

var image = ee.Algorithms.Landsat.simpleComposite({
  collection: collection,
  cloudScoreRange: 10,
  maxDepth: 40,
  asFloat: true
}).clip(area);

// Display as a composite
Map.centerObject(area,9);
Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0.03, max: 0.18, gamma: 1.4}, 'Composite');
