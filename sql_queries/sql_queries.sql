-- SQL queries used during the data processing
-- public.nutzung_kreis_290124 - contains official land use data
-- public.official_buidling_01092024 - contains building geometries

-- Creates new column and assigns the centroid of building
ALTER TABLE public.official_buidling_01092024 ADD test public.centroid_geom NULL;
update official_buidling_01092024
set centroid_geom = st_centroid(geom);

-- ALL GEOMETRIES HAVE TO HAVE SAME COORDINATE SYSTEM. In this project local EPSG: 25832 was used.
-- performs mapping based on the geometry of the actual land use and centroid of the building. Main table, further this table filled with attributes
create table public.nutz_building as
select n.id_0, n.geom as nutz_kreis_orig_geom, n.tranformed_geom as nutz_kreis_trans_geom, n.bez as nutz_division, n.aktualit as nutz_date_actual,
ob.geom as building_geom, ob.id as building_id, ob.centroid_geom as building_centroid
FROM public.nutzung_kreis_290124 n
left join public.official_buidling_01092024 ob on st_intersects( ob.centroid_geom , st_transform( n.geom,  4326))
where bez in  ('Umspannstation', 'Handel und Dienstleistung', 'Industrie und Gewerbe', 'Versorgungsanlage','Kraftwerk',
				'Lagerplatz', 'Entsorgung', 'Kläranlage, Klärwerk', 'Wasserwerk', 'Funk- und Fernmeldeanlage')
				and n.nutzart = 'Industrie- und Gewerbefläche' and ob.id is not null;


-- bayern_line table contains Line data from OSM
-- to calculate the closest distance between building centroid and specified geometry object.
-- The value of tag has to be updated accordingly: secondary, primary, tertiary, motorway, trunk
-- parking space: small, medium and large
-- railway:rail
WITH closest_primary AS (
    SELECT
        nb.building_id,
        bl.osm_id,
        st_distance(bl.geom::geography, nb.building_centroid::geography) AS dist
    FROM
        nutz_building nb
    CROSS JOIN LATERAL (
        SELECT
            bl.osm_id,
            bl.geom,
            bl.geom <-> nb.building_centroid AS dist
        FROM
            bayern_line bl
        WHERE
            bl.highway = 'primary'
        ORDER BY
            dist
        LIMIT 1
    ) bl
)
UPDATE nutz_building nb
SET primary_closest_distance = cp.dist
FROM closest_primary cp
WHERE nb.building_id = cp.building_id;


-- Parking data processing
-- Table bayern_multipolygon - contains Polygon data from OSM
     create table parking_lots as
     select *, st_area(geom::geography) parking_area  from bayern_multipolygon bm where bm.amenity in ('parking_space', 'parking');

-- Used to clean duplicated geometries (small parking space inside of larger parking space, while both represents the same parking space)
    create table parkin_lots_cleaned as
     WITH centroids AS (
     SELECT
        a.id AS centroid_id,
        a.geom AS centroid_geom,
        st_centroid(a.geom) as centroid_geoms,
        b.id AS container_id,
        b.geom AS container_geom,
        st_centroid(b.geom) as container_centroid
    FROM
        parking_lots a
    JOIN
        parking_lots b ON ST_Contains(b.geom, ST_Centroid(a.geom))
    WHERE
        a.id <> b.id --and --a.id = 5856336
   ),

   grouped as (
        SELECT
    container_id,
    ST_Union(centroid_geom, container_geom) AS merged_geom
FROM
    centroids
GROUP BY
    container_id, centroid_geom, container_geom
        )

        ,
        unique_geoms as (
        select st_union(merged_geom) geom, container_id as id from grouped group by container_id
        )

        select * from unique_geoms

        union

        select geom, id from parking_lots where id not in (select centroid_id from centroids) ;

-- Assigning parking category based on the Fisher-Jenks algorithm. Numbers can be derived using preprocessing/fisher_jenks.py script
    UPDATE parkin_lots_cleaned
SET parking_category = CASE
    WHEN geom_area  <= 6152 THEN 'small'
    WHEN geom_area BETWEEN 6152 AND 110551 THEN 'medium'
    ELSE 'large'
end

where geom_area > 20;