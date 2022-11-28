import warnings
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster as skc
from sklearn.neighbors import KDTree
import pandas as pd
import shapely.geometry as geo
import shapely.ops as geoops
import math
import time



poly_join_tol = 0.00001
floor_height = 3.0

def unit_energy(landtype):
    if landtype == 'CIE':
        return 128395.1903
    elif landtype == 'MED':
        return 220923.75
    elif landtype == 'MIPS':
        return 147457.7754
    elif landtype == 'MIXED':
        return -3618.51737
    elif landtype == 'MIXRES':
        return -24523.30092
    elif landtype == 'PDR':
        return 40046.04562
    elif landtype == 'RETAIL/ENT':
        return 50559.17407
    elif landtype == 'RESIDENT':
        return -45428.08446
    elif landtype == 'VISITOR':
        return -14967.66339
    else:
        return 0


#process land use data
data_landuse = pd.read_csv('LandUse2016.csv')
print('Number of parcels in LandUse map:',len(data_landuse))

start_t = time.time()
data_shape = list(data_landuse['the_geom'])
centroid_landuse = np.empty((len(data_shape), 2))
count = 0
for shape_str in data_shape:
    shape_vec = []
    # parse vertices data
    shape_p1 = shape_str[16:-3].split(')), ((')
    for shape1 in shape_p1:
        shape_p2 = shape1.split('), (')
        for shape2 in shape_p2:
            shape_vec.append(shape2)
    coord_x = 0
    coord_y = 0
    # for each polygon, compute its centroid
    for shape in shape_vec:
        coords = shape.split(', ')
        vert = []
        for coord in coords:
            vert.append([float(coord.split(' ')[0]), float(coord.split(' ')[1])])
        poly = geo.Polygon(vert)
        coord_x += list(poly.centroid.coords)[0][0]
        coord_y += list(poly.centroid.coords)[0][1]
    # take mean of centroid of all the polygon, as centroid of this building
    centroid_landuse[count, 0] = coord_x / len(shape_vec)
    centroid_landuse[count, 1] = coord_y / len(shape_vec)
    count += 1

end_t = time.time()
time_s = (end_t - start_t)
print('Compute centroids of parcels in LandUse map:',round(time_s, 4),'s')



data_footprint = pd.read_csv('Building_Footprints.csv')
print('Number of buildings in BuildingFootprint map:',len(data_footprint))

start_t = time.time()
data_shape = list(data_footprint['shape'])
centroid_footprint = np.empty((len(data_shape), 2))
count = 0
for shape_str in data_shape:
    shape_vec = []
    # parse vertices data
    shape_p1 = shape_str[16:-3].split(')), ((')
    for shape1 in shape_p1:
        shape_p2 = shape1.split('), (')
        for shape2 in shape_p2:
            shape_vec.append(shape2)
    coord_x = 0
    coord_y = 0
    # for each polygon, compute its centroid
    for shape in shape_vec:
        coords = shape.split(', ')
        vert = []
        for coord in coords:
            vert.append([float(coord.split(' ')[0]), float(coord.split(' ')[1])])
        poly = geo.Polygon(vert)
        coord_x += list(poly.centroid.coords)[0][0]
        coord_y += list(poly.centroid.coords)[0][1]
    # take mean of centroid of all the polygon, as centroid of this building
    centroid_footprint[count, 0] = coord_x / len(shape_vec)
    centroid_footprint[count, 1] = coord_y / len(shape_vec)
    count += 1

end_t = time.time()
time_s = (end_t - start_t)
print('Compute centroids of buildings in BuildingFootprint map:',round(time_s, 4),'s')


#KDtree
start_t = time.time()
X = centroid_landuse
kdt = KDTree(X, leaf_size=30, metric='euclidean')
X2 = centroid_footprint
index=kdt.query(X2, k=1, return_distance=False)

end_t = time.time()
time_s = (end_t - start_t)
print('Mapping from BuildingFootprint to LandUse by KDTree:',round(time_s, 4),'s')


#merge two datafram
start_t = time.time()
data_footprint['OBJECTID']=index+1
temp = data_footprint[['hgt_meancm','OBJECTID']]
temp = temp.groupby(by=["OBJECTID"], dropna=False).mean().reset_index()
data_processed = data_landuse[['OBJECTID','BLOCK_NUM','the_geom','LANDUSE','SHAPE_Area']]
data_processed = data_processed.astype({'OBJECTID': 'int64'}).merge(temp, how='left', on='OBJECTID')
#print(data_processed.head())
data_processed = data_processed.rename(columns={'hgt_meancm':'height'})
#compute energy use for each parcel
list_landtype = list(data_processed["LANDUSE"])
list_area = list(data_processed["SHAPE_Area"])
list_height = list(data_processed["height"])
list_energy = [0] * len(data_processed)
for i in range(0, len(data_processed)):
    uenergy = unit_energy(list_landtype[i])
    if math.isnan(list_area[i]):
        area = 0
    else:
        area = list_area[i] * 0.09290304
    if math.isnan(list_height[i]):
        height = 0
    else:
        height = list_height[i] / 100
    tenergy = uenergy * area * height / floor_height
    list_energy[i] = tenergy
data_processed['energy'] = list_energy

end_t = time.time()
time_s = (end_t - start_t)
print('Merge data and compute building energy usage:',round(time_s, 4),'s')
#data_processed.to_csv (r'data_processed.csv', index = False, header=True)


#combine parcels at the block level
start_t = time.time()
data_processed = data_processed.sort_values(by=['BLOCK_NUM'], ascending=True)
list_shape = list(data_processed['the_geom'])
list_blknum = list(data_processed['BLOCK_NUM'])
list_penergy = list(data_processed['energy'])

count = 0
last_blknum = 'NA'
list_multipolygon = []
list_blockdemand = []
for i in range(0, len(list_shape)):
    shape_str = list_shape[i]
    blknum = list_blknum[i]
    shape_list = []

    if blknum != last_blknum:
        if last_blknum != 'NA':
            #print('blknum:',last_blknum,', num of polygon:',len(polygon_list))
            list_multipolygon.append(geoops.unary_union(polygon_list))
            list_blockdemand.append(blockdemand)
        polygon_list = []
        blockdemand = 0

    last_blknum = blknum
    blockdemand += list_penergy[i]

    # parse vertices data
    shape_p1 = shape_str[16:-3].split(')), ((')
    for shape1 in shape_p1:
        shape_p2 = shape1.split('), (')
        for shape2 in shape_p2:
            shape_list.append(shape2)
    # for each polygon, compute its centroid
    for shape in shape_list:
        coords = shape.split(', ')
        vert = []
        for coord in coords:
            vert.append([float(coord.split(' ')[0]), float(coord.split(' ')[1])])
        polygon_list.append(geo.Polygon(vert))

    if i == len(list_shape)-1:
        #print('blknum:',last_blknum,', num of polygon:',len(polygon_list))
        list_multipolygon.append(geoops.unary_union(polygon_list))
        list_blockdemand.append(blockdemand)

for i in range(0, len(list_multipolygon)):
    list_multipolygon[i] = list_multipolygon[i].buffer(poly_join_tol, 1, join_style=geo.JOIN_STYLE.mitre).buffer(-poly_join_tol, 1, join_style=geo.JOIN_STYLE.mitre)

end_t = time.time()
time_s = (end_t - start_t)
print('Combine parcels into blocks:',round(time_s, 4),'s')


#plotting data
start_t = time.time()
fig1 = plt.figure(num=1, figsize=[36.0, 30.0])
plt.rcParams.update({'font.size': 40})
cmap = plt.cm.RdYlBu

list_areadmd = [0] * len(list_multipolygon)
for i in range(0, len(list_multipolygon)):
    list_areadmd[i] = list_blockdemand[i] / list_multipolygon[i].area
norm = plt.Normalize(0.15*max(min(list_areadmd), -1*max(list_areadmd)), 0.15*min(max(list_areadmd), -1*min(list_areadmd)))
#norm = plt.Normalize(min(list_areadmd), max(list_areadmd))
for i in range(0, len(list_multipolygon)):
    mpoly = list_multipolygon[i]
    dmd = list_areadmd[i]
    if mpoly.geom_type == 'MultiPolygon':
        for polygon in mpoly:
            x,y = polygon.exterior.xy
            #plt.plot(x,y)
            plt.fill(x,y, color=cmap(norm(dmd)))
    elif mpoly.geom_type == 'Polygon':
        x,y = mpoly.exterior.xy
        #plt.plot(x,y)
        plt.fill(x,y, color=cmap(norm(dmd)))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(list_areadmd)
fig1.colorbar(sm)
plt.savefig('block_demand_map.png')

end_t = time.time()
time_s = (end_t - start_t)
print('Plotting figures:',round(time_s, 4),'s')



