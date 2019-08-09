
#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include "collector.hpp"

#include <cstdint>
#include <iostream>


int32_t main(int32_t argc, char **argv) {
  int32_t retCode{1};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if ( (0 == commandlineArguments.count("cid")) ||
      (0 == commandlineArguments.count("name")) ||
      (0 == commandlineArguments.count("width")) ||
      (0 == commandlineArguments.count("height")) ||
      (0 == commandlineArguments.count("mapfile")) )  {
    std::cerr << argv[0] << " creates a path viewer based on gps and object data input." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --freq=<Frequency> --name=[shared memory name]"
              << "--width=[image width] --height=[image height] --skidpad --mapfile=[skidpad map] [--verbose]" << std::endl;
  } else {
    bool const verbose{commandlineArguments.count("verbose") != 0};

    cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};
    Collector collector(verbose);
    
    auto onObjectFrameStart{[&collector](cluon::data::Envelope &&envelope){
      if (envelope.senderStamp() == 0) {
        collector.getObjectFrameStart(envelope);
      }
    }};

    auto onObjectFrameEnd{[&collector](cluon::data::Envelope &&envelope){
      if (envelope.senderStamp() == 0) {
        collector.getObjectFrameEnd(envelope);
      }
    }};

    auto onObjectType{[&collector](cluon::data::Envelope &&envelope){
      collector.getObjectType(envelope);
    }};

    auto onObjectPosition{[&collector](cluon::data::Envelope &&envelope){
      collector.getObjectPosition(envelope);
    }};

    auto onEquilibrioception{[&collector](cluon::data::Envelope &&envelope){
      collector.getEquilibrioception(envelope);
    }};

    auto onAimpoint{[&collector](cluon::data::Envelope &&envelope){
      collector.getAimpoint(envelope);
    }};

    auto onLocalPath{[&collector](cluon::data::Envelope &&envelope){
      collector.getLocalPath(envelope);    
    }};

    auto onGeodeticWgs84Reading{[&collector](cluon::data::Envelope &&envelope) {
      collector.getWgs84Reading(envelope);
    }};

    auto onGroundSpeedReading{[&collector](cluon::data::Envelope &&envelope) {
      collector.getGroundSpeedReading(envelope);
    }};

    od4.dataTrigger(opendlv::logic::perception::ObjectFrameStart::ID(), onObjectFrameStart);
    od4.dataTrigger(opendlv::logic::perception::ObjectFrameEnd::ID(), onObjectFrameEnd);
    od4.dataTrigger(opendlv::logic::perception::ObjectType::ID(), onObjectType);
    od4.dataTrigger(opendlv::logic::perception::ObjectPosition::ID(), onObjectPosition);
    od4.dataTrigger(opendlv::logic::sensation::Equilibrioception::ID(), onEquilibrioception);
    od4.dataTrigger(opendlv::logic::action::AimPoint::ID(), onAimpoint);
    od4.dataTrigger(opendlv::logic::action::LocalPath::ID(), onLocalPath);
    od4.dataTrigger(opendlv::proxy::GroundSpeedReading::ID(), onGeodeticWgs84Reading);
    od4.dataTrigger(opendlv::proxy::GeodeticWgs84Reading::ID(), onGroundSpeedReading);

    if (commandlineArguments.count("skidpad") != 0)
      collector.setSkidpadCones(commandlineArguments["mapfile"]);

    const int height = std::stoi(commandlineArguments["height"]);
    const int width = std::stoi(commandlineArguments["width"]);
    const std::string sharedMemoryName{commandlineArguments["name"]};
    std::unique_ptr<cluon::SharedMemory> pSharedMemory(new cluon::SharedMemory{sharedMemoryName});
    if (pSharedMemory && pSharedMemory->valid()) {
      std::clog << argv[0] << " attached to shared memory: '" << pSharedMemory->name() << " (" << pSharedMemory->size() << " bytes)." << std::endl << std::endl;

      // Endless loop; end the program by pressing Ctrl-C.
      while (od4.isRunning()) {
        collector.GetCompleteFrameCFSD19();
        if (verbose) {
          uint64_t frameDuration = cluon::time::toMicroseconds(collector.frameEnd) - cluon::time::toMicroseconds(collector.frameStart);
          std::cout << "Using frame to find path, frame duration=" << frameDuration <<  std::endl;
        }

        cv::Mat img;
        // Wait for a notification of a new frame.
        pSharedMemory->wait();
        // Lock the shared memory.
        pSharedMemory->lock();
        {
          // Copy image into cvMat structure. Be aware of that any code between lock/unlock is blocking the camera to 
          // provide the next frame. Thus, any computationally heavy algorithms should be placed outside lock/unlock.
          cv::Mat wrapped(height, width, CV_8UC4, pSharedMemory->data());
          // If image from shared memory has different size with the pre-defined one, resize it.
          img = wrapped.clone();
        }
        pSharedMemory->unlock();

        collector.ProcessFrameCFSD19(img, true);
      }
      retCode = 0;
    }
  }
  return retCode;
}