# Master Thesis repo
This repository contains scripts that has been used during the Master's Thesis on topic **Attribute-based Classification of Industrial Building with OpenStreetMap and Remote Sensing Data in Bavaria**
by Manap Shymyr (manap.shymyr@tum.de).

During this thesis Python programming language and PostgreSQL with Postgis extension has been used. Therefore, the installation of the PostgreSQL is required.

Steps to reproduce:
1. Download vector data (actual land use, building footprint and OSM) and store them to PostgreSQL using the QGIS.
2. Assign land use types to buildings - refer to sql_queries 
3. Location features (closest distance and parking count) - refer to sql_queries
4. Shape features - refer to preprocessing/shape_features.py
5. VV and VH extraction - refer to preprocessing/download_vv_vh.py
6. NDVI extraction - refer to preprocessing/download_ndvi.py
7. Building height extraction - refer to preprocessing/get_height.py
8. VHR data download and roof type extraction - refer to preprocessing/parse_xml.py, png_extra.py and raster_to_db.py
9. Classification and SHAP values - refer to scripts in the classification directory


The final dataset used for classification can be downloaded from:
1. https://syncandshare.lrz.de/getlink/fi4nJZEmieKbZJ1B4hvGUo/
2. https://drive.google.com/drive/folders/1fyLSjT0nB8mrC_kLWKMw7yH50e6i5eBd?usp=sharing