#include "frontend/mlir_importer.h"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

static std::vector<int> parse_shape(const std::string& s) {
    std::vector<int> shape;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoi(item));
    }

    return shape;
}

static std::vector<float> parse_values(const std::string& s) {
    std::vector<float> values;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        values.push_back(std::stof(item));
    }

    return values;
}

static OpType parse_op(const std::string& op) {
    if (op == "MatMul") return OpType::MatMul;
    if (op == "Add") return OpType::Add;
    if (op == "ReLU") return OpType::ReLU;
    if (op == "Attention") return OpType::Attention;
    if (op == "CausalAttention") return OpType::CausalAttention;
    
    throw std::runtime_error("Unsupported op: " + op);
}

Graph load_mlir_graph(const std::string& path) {
    Graph graph;
    std::unordered_map<std::string, int> tensor_ids;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open MLIR file: " + path);
    }

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string kind;
        ss >> kind;

        if (kind == "TENSOR") {
            std::string name;
            std::string shape_str;

            ss >> name >> shape_str;

            int id = graph.add_tensor(Tensor(name, parse_shape(shape_str)));
            tensor_ids[name] = id;
        }

        else if (kind == "CONST") {
            std::string name;
            std::string shape_str;
            std::string values_str;

            ss >> name >> shape_str >> values_str;

            Tensor tensor(name, parse_shape(shape_str));
            tensor.data = parse_values(values_str);

            int id = graph.add_tensor(tensor);
            tensor_ids[name] = id;
        }

        else if (kind == "NODE") {
            std::string op_name;
            ss >> op_name;

            std::vector<std::string> input_names;
            std::string token;

            while (ss >> token) {
                if (token == "->") break;
                input_names.push_back(token);
            }

            std::string output_name;
            ss >> output_name;

            std::vector<int> inputs;
            for (const auto& input_name : input_names) {
                if (!tensor_ids.count(input_name)) {
                    throw std::runtime_error("Unknown input tensor: " + input_name);
                }
                inputs.push_back(tensor_ids[input_name]);
            }

            if (!tensor_ids.count(output_name)) {
                int output_id = graph.add_tensor(Tensor(output_name, {}));
                tensor_ids[output_name] = output_id;
            }

            int output_id = tensor_ids[output_name];

            graph.add_node(Node(output_name, parse_op(op_name), inputs, {output_id}));
        }

        else {
            throw std::runtime_error("Unknown IR line kind: " + kind);
        }
    }

    std::cout << "[MLIRImporter] Loaded graph from " << path << "\n";

    return graph;
}