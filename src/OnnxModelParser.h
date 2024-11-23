#ifndef ONNX_MODEL_PARSER_H
#define ONNX_MODEL_PARSER_H

#include <string>
#include <vector>
#include <onnx/onnx.pb.h>

class OnnxModelParser {
private:
    onnx::ModelProto modelProto;

public:
    OnnxModelParser(const std::string& modelPath);

    const onnx::GraphProto& getGraph() const;
    std::vector<onnx::NodeProto> getNodes() const;
    onnx::TensorProto getInitializer(const std::string& name) const;
};

#endif // ONNX_MODEL_PARSER_H