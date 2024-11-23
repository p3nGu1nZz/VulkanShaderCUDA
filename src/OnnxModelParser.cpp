#include "OnnxModelParser.h"
#include <fstream>
#include <stdexcept>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

OnnxModelParser::OnnxModelParser(const std::string& modelPath) {
    std::ifstream modelFile(modelPath, std::ios::binary);
    if (!modelFile.is_open()) {
        throw std::runtime_error("Failed to open ONNX model file: " + modelPath);
    }

    google::protobuf::io::IstreamInputStream input(&modelFile);
    if (!modelProto.ParseFromIstream(&modelFile)) {
        modelFile.close();
        throw std::runtime_error("Failed to parse ONNX model.");
    }

    modelFile.close();
}

const onnx::GraphProto& OnnxModelParser::getGraph() const {
    return modelProto.graph();
}

std::vector<onnx::NodeProto> OnnxModelParser::getNodes() const {
    return std::vector<onnx::NodeProto>(modelProto.graph().node().begin(), 
                                      modelProto.graph().node().end());
}

onnx::TensorProto OnnxModelParser::getInitializer(const std::string& name) const {
    for (const auto& initializer : modelProto.graph().initializer()) {
        if (initializer.name() == name) {
            return initializer;
        }
    }
    throw std::runtime_error("Initializer not found: " + name);
}