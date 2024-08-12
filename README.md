# Attribute-based Classification of Industrial Building with OpenStreetMap and Remote Sensing Data in Bavaria
This repository contains the scripts developed by Manap Shymyr as part of his Master's Thesis in the Earth-Oriented Space Science and Technology program at the Technical University of Munich (TUM). For inquiries, please contact Manap Shymyr at manap.shymyr@tum.de.

# Abstract
<div style="text-align: center; margin: 20px;">
  <p>
  The attribute-based classification of building types is crucial for various applications. While the number of building footprints and studies to extract the building footprints are increasing, the identification of attributes of those buildings remains challenging. Moreover, the classification of industrial building types had less attention despite their importance in the economy. Although many studies on building-type classification using raster and vector data focus on the accuracy of the classification model, a comparative analysis of the results based on raster versus vector data remains unexplored. This study focuses on attribute-based industrial building type classification based on the openly available vector and raster data to overcome these limitations. In addition, the importance of features based on the raster and vector data to the output of the attribute-based classification algorithm was assessed.
</p>
  <p>
  This study extracted features for attribute-based building classification algorithms by leveraging openly available vector and raster data. Features were grouped into shape, location, and spectral. The shape features are extracted based on the geometry of buildings and represent their shape characteristics. The location features consider the buildings' proximity to the nearest road and parking space. These features are extracted from vector data. The spectral features, including VV, VH polarization, and NDVI, are derived from Sentinel-1 and Sentinel-2 satellites to describe the vertical structure and surroundings of the buildings, respectively. In addition, building height and roof types were utilized in the attribute-based classification as raster-based features. As the attribute-based classification algorithms, a Random Forest, Support-Vector-Machine, and Convolutional Neural Network were applied.
</p>
  <p>
  The study results showed that Random Forest had the highest accuracy compared to the other two models. Furthermore, combining vector and raster features resulted in higher accuracy than utilizing only vector or raster-based features. In addition, the comparison of results showed that attribute classification based on vector-based features is more accurate than raster-based features while utilizing raster data is more complex than vector data. Lastly, the importance of individual features in the model's prediction was evaluated using SHAP values. The results show the shape features and the features based on the NDVI values showed significant importance in the output of the classification model.
  </p>
</div>

# Software
* Python
* PostgreSQL with PostGIS extension
* QGIS

Steps to reproduce:
1. Download vector data:
    * [Official land use (Tatsächliche Nutzung)](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=tatsaechlichenutzung)
    * [Building footprint (Hausumringe)](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=hausumringe)
    * [OpenStreetMap for Bayern](https://download.geofabrik.de/europe/germany/bayern.html)
    * [Sentinel-1](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
    * [Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
    * [Building height](https://geoservice.dlr.de/web/maps/eoc:wsf3d)
    * [VHR aerial images](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dop40)
2. Assign land use types to buildings - refer to sql_queries 
3. Location features (closest distance and parking count) - refer to sql_queries
4. Shape features - refer to preprocessing/shape_features.py
5. VV and VH extraction - refer to preprocessing/download_vv_vh.py
6. NDVI extraction - refer to preprocessing/download_ndvi.py
7. Building height extraction - refer to preprocessing/get_height.py
8. VHR data download and roof type extraction - refer to preprocessing/parse_xml.py, png_extra.py and raster_to_db.py
9. Classification and SHAP values - refer to scripts in the \classification directory


The final dataset used for classification can be downloaded from and used with scripts in \classification:
1. Download from [LRZ Sync+Share](https://syncandshare.lrz.de/getlink/fi4nJZEmieKbZJ1B4hvGUo)
2. Download from [Google Drive](https://drive.google.com/drive/folders/1fyLSjT0nB8mrC_kLWKMw7yH50e6i5eBd?usp=sharing)
