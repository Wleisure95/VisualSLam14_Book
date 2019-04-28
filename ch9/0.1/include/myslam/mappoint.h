/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MAPPOINT_H
#define MAPPOINT_H

namespace myslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      id_; // ID
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    int         observed_times_;    // being observed by feature matching algo.
    int         correct_times_;     // being an inliner in pose estimation
    
    MapPoint();
    MapPoint( long id, Vector3d position, Vector3d norm );
    
    // factory function
    static MapPoint::Ptr createMapPoint();
};
}

#endif // MAPPOINT_H
