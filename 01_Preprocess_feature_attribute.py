# -------------------------------
# author: Hao Li, leebobgiser316@gmail.com
# data: 05.10.2023
# -------------------------------


import geopandas as gpd
from shapely.ops import unary_union
import numpy as np
import rasterio


def load_gdf_3857(geojson_file):
    return gpd.read_file(geojson_file).to_crs(epsg=3857)


def get_ua_center(geojson_file):
    gdf = gpd.read_file(geojson_file).to_crs(epsg=3857)
    ua = gdf.unary_union
    if ua.geom_type == "MultiPolygon":
        polygons = [p.buffer(0) for p in ua.geoms]
    else:
        assert 0
    dissolved_gdf = gpd.GeoDataFrame(geometry=polygons, crs=3857)
    return dissolved_gdf


def get_min_dis(logistic_gdf, target_gdf):
    sindex = target_gdf.sindex
    min_dis = []
    for idx, row in logistic_gdf.iterrows():
        point = row['geometry']
        nearest = target_gdf.iloc[sindex.nearest(point)[1][0]]
        min_dis.append(point.distance(nearest.geometry))
    return min_dis


###########  Data Import #############

## Reference data
# orig_geojson = "data/korea_logistic_asserts_20230905.geojson"
# out_geojson = "data/korea_logistic_asserts_20230905_center_with_feature.geojson"

## Negative samples
orig_geojson = r"./data/korea_all_buildings_with_feature.geojson"
out_geojson = r"./data/korea_all_buildings_with_feature.geojson"

## Some  geo-transformation
logistic_gdf = load_gdf_3857(orig_geojson)
logistic_gdf.geometry = logistic_gdf.centroid



###########  Feature Engineering #############

# calculate area and perimeter for buildings
logistic_gdf['Shape_Leng'] = logistic_gdf['geometry'].length
logistic_gdf['Shape_Area'] = logistic_gdf['geometry'].area


# get worldpop
print("cal worldpop")
with rasterio.open("data/feature_related/kor_ppp_2020_3857.tif") as src:
    for idx, row in logistic_gdf.iterrows():
        point = row['geometry']
        # sample from the raster
        for val in src.sample(point.coords):
            logistic_gdf.at[idx, 'worldpop'] = val


# get wsf 3d building height
print("cal WSF3D")
with rasterio.open("./data/feature_related/WSF3D_V02_BuildingHeight.tif") as src:
    for idx, row in logistic_gdf.iterrows():
        point = row['geometry']
        # sample from the raster
        for val in src.sample(point.coords):
            logistic_gdf.at[idx, 'wsf3d'] = val

# get within industrial area
print("cal within industrial area")
industrial_landuse_gdf = load_gdf_3857("data/feature_related/osm_kor_landuse_industrial.geojson")
logistic_gdf['industrial_landuse'] = logistic_gdf.geometry.apply(
    lambda point: not industrial_landuse_gdf.loc[industrial_landuse_gdf.intersects(point)].empty
)


# cal distance features
print("cal dis")
railway_gdf = load_gdf_3857("data/feature_related/osm_kor_railways.geojson")
logistic_gdf['railway_dis'] = get_min_dis(logistic_gdf,railway_gdf)

highway_gdf = load_gdf_3857("data/feature_related/osm_kor_highway_motorway.geojson")
logistic_gdf['highway_dis'] = get_min_dis(logistic_gdf,highway_gdf)

airport_gdf = get_ua_center("data/feature_related/osm_kor_airports.geojson")
logistic_gdf['airport_dis'] = get_min_dis(logistic_gdf,airport_gdf)

seaport_gdf = get_ua_center("data/feature_related/osm_kor_sea_ports.geojson")
logistic_gdf['seaport_dis'] = get_min_dis(logistic_gdf,seaport_gdf)


###########  Save Result #############
logistic_gdf.to_file(out_geojson, driver="GeoJSON")
