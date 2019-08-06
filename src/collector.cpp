/**
* Copyright (C) 2017 Chalmers Revere
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
* USA.
*/

#include "collector.hpp"

bool compareCone(const Cone& c1, const Cone& c2) {
  return (c1.m_x < c2.m_x);
}

/*-------------------------------------------------------*/
Collector::Collector(bool VERBOSE) : 
  m_currentConeFrame{}
, uncompleteFrameMutex{}
, uncompleteFrame{}
, currentUncompleteFrameId{}
, completeFrameMutex{}
, completeFrame{}
, currentCompleteFrameId{}
, frameStart{}
, frameEnd{}
, m_verbose{VERBOSE} 
{
    cameraMtx = (cv::Mat_<double>(3, 3) << m_f, 0, m_cx, 0, m_f, m_cy, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(4, 1) << -0.171237, 0.0267087, 0, 0);
}

void Collector::setSkidpadCones(const std::string& mapfile) {
    m_isSkidpad = true;

    std::ifstream skidpadMap(mapfile);
    if (!skidpadMap.is_open()) {
        std::cerr << "Failed to open skidpad map file" << std::endl;
    }
    
    int outWidth = 480;
    int outHeight = 480;
    int heightOffset = 180;
    double resultResize = 10;
    skidpadImg = cv::Mat::zeros(outHeight, outWidth, CV_8UC4);

    while (!skidpadMap.eof()) {
        double x, y;
        skidpadMap >> x;
        skidpadMap.ignore(1, ',');
        skidpadMap >> y;
        skidpadMap.ignore(1, ',');
        int type;
        skidpadMap >> type;

        skidpadCones.push_back(Cone(0, x, y, 0));
        int xt = int(x * resultResize + outWidth/2);
        int yt = outHeight - int(y * resultResize) - heightOffset;
        cv::Scalar coneColor;
        if (type == 0) coneColor= cv::Scalar(0,255,255); // yellow
        else if (type == 1) coneColor= cv::Scalar(255,0,0); // blue
        else if (type == 2) coneColor= cv::Scalar(0,69,255); // orange
        cv::circle(skidpadImg, cv::Point(xt,yt), 4, coneColor, -1);
    }
    skidpadMap.close();
}

void Collector::GetCompleteFrameCFSD19() {
    // Copy cones to m_currentConeFrame for processing
    std::lock_guard<std::mutex> lock(completeFrameMutex);
    if (!completeFrame.size())
        return;
        
    for (auto elem : completeFrame)
        m_currentConeFrame.push(elem.second);

    if (m_verbose) {
        uint64_t frameDuration = cluon::time::toMicroseconds(frameEnd) - 
                                 cluon::time::toMicroseconds(frameStart);
        std::cout << "Using frame to find path, frame duration=" << frameDuration
                  << "ms, #frames=" << completeFrame.size() <<  std::endl;
    }
}

void Collector::getObjectFrameStart(cluon::data::Envelope envelope) {
    opendlv::logic::perception::ObjectFrameStart msg =
        cluon::extractMessage<opendlv::logic::perception::ObjectFrameStart>(std::move(envelope));

    uint32_t objectFrameId = msg.objectFrameId();
    {
        std::lock_guard<std::mutex> lock(uncompleteFrameMutex);
        uncompleteFrame = std::map<uint32_t, Cone>();
        currentUncompleteFrameId = objectFrameId;
        frameStart = envelope.sent();
    }

    if (m_verbose) {
        std::cout << "Got frame START with id=" << objectFrameId << std::endl;
    }
}

void Collector::getObjectFrameEnd(cluon::data::Envelope envelope) {
    opendlv::logic::perception::ObjectFrameEnd msg =
        cluon::extractMessage<opendlv::logic::perception::ObjectFrameEnd>(std::move(envelope));

    uint32_t objectFrameId = msg.objectFrameId();
    {
        std::lock_guard<std::mutex> lock1(completeFrameMutex);
        std::lock_guard<std::mutex> lock2(uncompleteFrameMutex);
        completeFrame = uncompleteFrame;

        currentCompleteFrameId = currentUncompleteFrameId;
        frameEnd = envelope.sent();
    }

    if (m_verbose) {
        std::cout << "Got frame END with id=" << objectFrameId << std::endl;
    }
}

void Collector::getObjectType(cluon::data::Envelope envelope) {
    opendlv::logic::perception::ObjectType msg =
        cluon::extractMessage<opendlv::logic::perception::ObjectType>(std::move(envelope));

    uint32_t objectId = msg.objectId();
    {
        std::lock_guard<std::mutex> lock(uncompleteFrameMutex);
        Cone newCone(objectId, 0, 0, 0);
        uncompleteFrame[objectId] = newCone;
        std::cout << "Got NEW OBJECT with id=" << objectId << std::endl;

        if (uncompleteFrame.count(objectId)) {
            uncompleteFrame[objectId].m_color = msg.type();
        }
    }
    if (m_verbose) {
        std::cout << "Got OBJECT TYPE for object with id=" << objectId
                  << " and type=" << msg.type() << std::endl;
    }
}

void Collector::getObjectPosition(cluon::data::Envelope envelope) {
    opendlv::logic::perception::ObjectPosition msg =
        cluon::extractMessage<opendlv::logic::perception::ObjectPosition>(std::move(envelope));

    uint32_t objectId = msg.objectId();
    {
        std::lock_guard<std::mutex> lock(uncompleteFrameMutex);
        if (uncompleteFrame.count(objectId)) {
            if (msg.x() > 0 && msg.x() < 20) {
                // remap the coordiate so that the coordinate is as follow in path planner
                // from perception:
                // x: positive upward vertically
                // y: positive to left horizontally
                uncompleteFrame[objectId].m_x = msg.x();
                uncompleteFrame[objectId].m_y = msg.y();
                
                // In path planner
                // x: postive to right horizontally
                // y: positive upward vertically
                // uncompleteFrame[objectId].m_x = -msg.y();
                // uncompleteFrame[objectId].m_y = msg.x();
            }
            else {
                uncompleteFrame.erase(objectId);
            }
        }
    }
    if (m_verbose) {
        std::cout << "Got OBJECT POSITION for object with id=" << objectId
                  << " and x=" << msg.x() << " y=" << msg.y() << std::endl;
    }
}

void Collector::getEquilibrioception(cluon::data::Envelope envelope) {
    opendlv::logic::sensation::Equilibrioception msg =
        cluon::extractMessage<opendlv::logic::sensation::Equilibrioception>(std::move(envelope));

    float vx = msg.vx();
    float yawRate = msg.yawRate();
    if (m_verbose) {
        std::cout << "Got EQUILIBRIOCEPTION vx=" << vx << " and yawRate="
                  << yawRate << std::endl;
    }
}

void Collector::getAimpoint(cluon::data::Envelope envelope){
  opendlv::logic::action::AimPoint msg = 
        cluon::extractMessage<opendlv::logic::action::AimPoint>(
            std::move(envelope));

      float angle = msg.azimuthAngle();
      float distance = msg.distance();
      
      std::lock_guard<std::mutex> lock(aimpointMutex);
      m_currentAim[0] = distance*cos(angle);
      m_currentAim[1] = distance*sin(angle);
      if (m_verbose) {
        std::cout << "Got AIMPOINT for object with id=" << 0 
          << " and x=" << m_currentAim[0] << " y=" << m_currentAim[1] << std::endl;
      }
}

void Collector::getLocalPath(cluon::data::Envelope envelope){
    auto msg = cluon::extractMessage<opendlv::logic::action::LocalPath>(std::move(envelope));

    std::string data = msg.data();
    uint32_t length = msg.length();

    std::lock_guard<std::mutex> lock(pathMutex);
    // If message is empty, use previous value
    // TODO: Add logic for the case of msg length == 0
    if (msg.length() != 0) {
        localPath.clear();
        for (uint32_t i = 0; i < length; i++) {
            float x;
            float y;
            memcpy(&x, data.c_str() + (3 * i + 0) * 4, 4);
            memcpy(&y, data.c_str() + (3 * i + 1) * 4, 4);
            // z not parsed, since not used
            
            // from path planner
            // x: postive to right horizontally
            // y: positive upward vertically
            std::array<float, 2> p;
            p[0] = x;
            p[1] = y;
            localPath.push_back(p);
            if (m_verbose) {
                std::cout << "Got LocalPath x=" << x << " y=" << y << std::endl;
            }
        }
    }
    if (m_verbose)
        std::cout << "Total " << msg.length() << " points in local path" << std::endl;
}

void Collector::getWgs84Reading(cluon::data::Envelope envelope) {
    std::lock_guard<std::mutex> lock(gpsMutex);
    auto msg = cluon::extractMessage<opendlv::proxy::GeodeticWgs84Reading>(std::move(envelope));
    std::array<double, 2> currentPos = {msg.latitude(), msg.longitude()};
    if (m_atStart) {
        m_startPos = currentPos;
        gpsPath.push_back({0,0}); // take the starting position as origin
        m_atStart = false;
    }
    else {
        std::array<double, 2> distance = wgs84::toCartesian(m_startPos, currentPos);
        if (distance[0]*distance[0] + distance[1]*distance[1] > 0.1) {
            std::array<double, 2> transform;
            transform[0] = distance[0] * sin(m_theta) - distance[1] * cos(m_theta);
            transform[1] = distance[0] * cos(m_theta) + distance[1] * sin(m_theta);
            gpsPath.push_back(transform);
        }
    }
}

void Collector::ProcessFrameCFSD19(cv::Mat& img, bool goRight) {
    // Todo: filter cones here
    if (!m_currentConeFrame.size()) {
        std::cout << "Current frame has no cones!" << std::endl;
        return;
    }

    if (m_verbose)
        std::cout << "Current frame has: " << m_currentConeFrame.size() << " cones\n";
    
    //Copy cones of different colors to their own containers for processing
    std::vector<Cone> tempYellowCones;
    std::vector<Cone> tempBlueCones;
    std::vector<Cone> tempOrangeCones;     
    
    while(m_currentConeFrame.size() >0) {
        Cone cone = m_currentConeFrame.front();
        switch (m_currentConeFrame.front().m_color) {
            case 0: // yellow
                tempYellowCones.push_back(cone);
                if (m_verbose) std::cout << "a yellow cone " << std::endl;
                break;
            case 1: // blue
                tempBlueCones.push_back(cone);
                if (m_verbose) std::cout << "a blue cone " << std::endl;
                break;
            case 2: // orange
                tempOrangeCones.push_back(cone);
                if (m_verbose) std::cout << "an orange cone " << std::endl;
                break;
        }
        // Done copying, delete pointers to free memory
        m_currentConeFrame.pop();
    }
    if (m_verbose)
        std::cout << "number of cones:" 
                  << "\n    yellow " << tempYellowCones.size()
                  << "\n      blue " << tempBlueCones.size()
                  << "\n    orange " << tempOrangeCones.size()
                  << std::endl;

    /* coordinate from perception:
     * x: positive forward
     * y: positive leftward
     */
    // Sort cones in an increasing order based on cone discance (i.e. x value)
    std::sort(tempYellowCones.begin(), tempYellowCones.end(), compareCone);
    std::sort(tempBlueCones.begin(), tempBlueCones.end(), compareCone);
    std::sort(tempOrangeCones.begin(), tempOrangeCones.end(), compareCone);

    // ShowResult(img, tempBlueCones, tempYellowCones, tempOrangeCones);

    // If the car goes to right circle,
    // erase yellow cones that are in the left side of the closest blue cone
    if (goRight && tempBlueCones.size()) {
        std::vector<Cone>::iterator closestBlue = tempBlueCones.begin();
        for (auto yellow = tempYellowCones.begin(); yellow != tempYellowCones.end(); ) {
            if (yellow->m_y > closestBlue->m_y)
                tempYellowCones.erase(yellow);
            else
                yellow++;
        }
    }


    // If the car goes to left circle,
    // erase blue cones that are in the right side of the closest yellow cone
    if (!goRight && tempYellowCones.size()) {
        std::vector<Cone>::iterator closestYellow = tempYellowCones.begin();
        for (auto blue = tempBlueCones.begin(); blue != tempBlueCones.end(); ) {
            if (blue->m_y < closestYellow->m_y)
                tempBlueCones.erase(blue);
            else
                blue++;
        }
    }

    ShowResult(img, tempBlueCones, tempYellowCones, tempOrangeCones);
}

void Collector::ShowResult(cv::Mat& img, std::vector<Cone>& blue, std::vector<Cone>& yellow, std::vector<Cone>& orange) {
    uint32_t n;
    double curPosX = 0, curPosY = 0;
    {
        std::lock_guard<std::mutex> gpsLock(gpsMutex);
        if ((n = gpsPath.size())) {
            curPosX = gpsPath.back()[0];
            curPosY = gpsPath.back()[1];
        }
    }

    if (m_theta != 0 && n == 20) {
        std::lock_guard<std::mutex> gpsLock(gpsMutex);
        double slope = 0;
        int count = 0;
        for (uint32_t i = 0; i < n; i += 3) {
            slope += gpsPath[i][1] / (gpsPath[i][0] + 1e-6);
            count++;
        }
        slope /= count;
        m_theta = atan(slope);

        for (uint32_t i = 1; i < n; i++) {
            std::array<double, 2> distance = gpsPath[i];
            gpsPath[i][0] = distance[0] * sin(m_theta) - distance[1] * cos(m_theta);
            gpsPath[i][1] = distance[0] * cos(m_theta) + distance[1] * sin(m_theta);
        }
    }

    int outWidth = 480;
    int outHeight = 480;
    int heightOffset = 20;
    double resultResize = 10;
    cv::Mat out2D = cv::Mat::zeros(outHeight, outWidth, CV_8UC4);

    cv::Mat out3D = img.clone();
    // cv::undistort(img, out3D, cameraMtx, distCoeffs);

    
    if (n) {
        std::lock_guard<std::mutex> gpsLock(gpsMutex);
        for (uint32_t i = 0; i < n; i++) {
            int xt = int(gpsPath[i][0] * resultResize + outWidth/2);
            int yt = outHeight - int(gpsPath[i][1] * resultResize) - heightOffset;
            cv::circle(out2D, cv::Point(xt,yt), 3, cv::Scalar(255,255,255), -1);
        }
        
        // calculate current heading
        if (n > 1 && gpsPath[n-1][0] != gpsPath[n-2][0])
            m_heading = atan((gpsPath[n-1][1]-gpsPath[n-2][1]) / (gpsPath[n-1][0]-gpsPath[n-2][0]+1e-6));
    }

    double y = 1.0; // camera height
    if (blue.size()) {
        for (uint32_t i = 0; i < blue.size(); i++) {
            double x = -blue[i].m_y;
            double z = blue[i].m_x; 
            int xt = int(m_f * x / z + m_cx);
            int yt = int(m_f * y / z + m_cy);
            int pointSize = int(60/z);
            if (xt-pointSize >= 0 && xt+pointSize <= 1280 && yt-pointSize >= 0 && yt+pointSize <= 720)
                cv::circle(out3D, cv::Point(xt,yt), pointSize, cv::Scalar(255,0,0), -1);

            double a = blue[i].m_x * cos(m_heading) - blue[i].m_y * sin(m_heading) + curPosX;
            double b = blue[i].m_x * sin(m_heading) + blue[i].m_y * cos(m_theta) + curPosY;
            double xx = a * sin(m_theta) - b * cos(m_theta);
            double yy = a * cos(m_theta) + b * sin(m_theta);
            int xxt = int(xx * resultResize + outWidth/2);
            int yyt = outHeight - int(yy*resultResize) - heightOffset;
            cv::circle(out2D, cv::Point(xxt, yyt), 3, cv::Scalar(255,0,0), -1);
        }
    }

    if (yellow.size()) {
        for (uint32_t i = 0; i < yellow.size(); i++) {
            double x = -yellow[i].m_y;
            double z = yellow[i].m_x; 
            int xt = int(m_f * x / z + m_cx);
            int yt = int(m_f * y / z + m_cy);
            int pointSize = int(60/z);
            if (xt-pointSize >= 0 && xt+pointSize <= 1280 && yt-pointSize >= 0 && yt+pointSize <= 720)
                cv::circle(out3D, cv::Point(xt,yt), pointSize, cv::Scalar(0,255,255), -1);

            double a = yellow[i].m_x * cos(m_heading) - yellow[i].m_y * sin(m_heading) + curPosX;
            double b = yellow[i].m_x * sin(m_heading) + yellow[i].m_y * cos(m_theta) + curPosY;
            double xx = a * sin(m_theta) - b * cos(m_theta);
            double yy = a * cos(m_theta) + b * sin(m_theta);
            int xxt = int(xx * resultResize + outWidth/2);
            int yyt = outHeight - int(yy*resultResize) - heightOffset;
            cv::circle(out2D, cv::Point(xxt, yyt), 3, cv::Scalar(0,255,255), -1);
        }
    }

    if (orange.size()) {
        for (uint32_t i = 0; i < orange.size(); i++) {
            double x = -orange[i].m_y;
            double z = orange[i].m_x; 
            int xt = int(m_f * x / z + m_cx);
            int yt = int(m_f * y / z + m_cy);
            int pointSize = int(60/z);
            if (xt-pointSize >= 0 && xt+pointSize <= 1280 && yt-pointSize >= 0 && yt+pointSize <= 720)
                cv::circle(out3D, cv::Point(xt,yt), pointSize, cv::Scalar(0,69,255), -1);

            double a = orange[i].m_x * cos(m_heading) - orange[i].m_y * sin(m_heading) + curPosX;
            double b = orange[i].m_x * sin(m_heading) + orange[i].m_y * cos(m_theta) + curPosY;
            double xx = a * sin(m_theta) - b * cos(m_theta);
            double yy = a * cos(m_theta) + b * sin(m_theta);
            int xxt = int(xx * resultResize + outWidth/2);
            int yyt = outHeight - int(yy*resultResize) - heightOffset;
            cv::circle(out2D, cv::Point(xxt, yyt), 3, cv::Scalar(255,0,0), -1);
        }
    }

    {
        std::lock_guard<std::mutex> aimpointLock(aimpointMutex);
        double x = -m_currentAim[1];
        double z = m_currentAim[0];
        int xt = int(m_f * x / z + m_cx);
        int yt = int(m_cy+150);
        if (xt-3 >= 0 && xt+3 <= 1280 && yt-3 >= 0 && yt+3 <= 720)
            cv::rectangle(out3D, cv::Point(xt-3,yt-3), cv::Point(xt+3,yt+3), cv::Scalar(0,255,0), -1);
    }

    if (localPath.size()) {
        std::lock_guard<std::mutex> pathLock(pathMutex);
        for (uint32_t i = 0; i < localPath.size(); i++) {
            // from path planner
            // x: postive to right horizontally
            // y: positive upward vertically
            double x = localPath[i][0];
            double z = localPath[i][1];
            int xt = int(m_f * x / z + m_cx);
            int yt = int(m_f * y / z + m_cy);
            if (xt-3 >= 0 && xt+3 <= 1280 && yt-3 >= 0 && yt+3 <= 720)
                cv::circle(out3D, cv::Point(xt,yt), 3, cv::Scalar(69,255,69), -1);
        }
    }
    cv::imshow("reprojected cones", out3D);
    
    if (m_isSkidpad) {
        cv::Mat skidpadOut = skidpadImg.clone();
        std::lock_guard<std::mutex> gpsLock(gpsMutex);
        if (n) {
            for (uint32_t i = 0; i < n; i++) {
                int xt = int(gpsPath[i][0] * resultResize + outWidth/2);
                int yt = outHeight - int(gpsPath[i][1] * resultResize) - heightOffset;
                cv::circle(skidpadOut, cv::Point(xt,yt), 4, cv::Scalar(255,255,255), -1);
            }
        }
        cv::imshow("path and skidpad map", skidpadOut);
    }

    cv::imshow("path and detected cones", out2D);
    cv::waitKey(1);
}