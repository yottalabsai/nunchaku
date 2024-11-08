#include "Serialization.h"

#include <nlohmann/json.hpp>
#include <mio/mmap.hpp>

// #include <sys/mman.h>

using json = nlohmann::json;
using spdlog::fmt_lib::format;

class SafeTensors::mmap_file : public mio::mmap_source {
public:
    mmap_file(std::string_view filename) : mio::mmap_source(filename, 0, mio::map_entire_file) {}
};

SafeTensors::SafeTensors(std::string_view filename) {
    std::error_code ec;
    this->mapped = std::make_unique<mmap_file>(filename);
    if (ec) {
        throw std::system_error(ec);
    }
    // char *ptr = (char *)malloc(1024);
    // checkCUDA(cudaHostRegister(ptr, 1024, cudaHostRegisterDefault));
    if (cudaHostRegister(const_cast<char *>(this->mapped->data()), this->mapped->size(), cudaHostRegisterPortable | cudaHostRegisterReadOnly) != cudaSuccess) {
        spdlog::warn("Unable to pin memory: {}", cudaGetErrorString(cudaGetLastError()));
        // mlock(const_cast<char *>(this->mapped->data()), this->mapped->size());
    }
    parseHeader();
}

SafeTensors::~SafeTensors() {
    checkCUDA(cudaHostUnregister(const_cast<char *>(this->mapped->data())));
}

void SafeTensors::parseHeader() {
    static const std::unordered_map<std::string, Tensor::ScalarType> mapDType = {
        { "BF16", Tensor::BF16  },
        { "F16",  Tensor::FP16  },
        { "F32",  Tensor::FP32  },
        { "I8",   Tensor::INT8  },
        { "I32",  Tensor::INT32 },
        { "I64",  Tensor::INT64 },
    };

    auto check = [](bool cond, std::source_location location = std::source_location::current()) {
        if (!cond) {
            throw std::runtime_error(format("Safetensors check failed at {}:{}", location.file_name(), location.line()));
        }
    };

    check(this->mapped->size() > 8);
    uint64_t sizeHeader = *reinterpret_cast<const uint64_t *>(this->mapped->data());

    check(this->mapped->size() - 8 >= sizeHeader);
    json header = json::parse(this->mapped->begin() + 8, this->mapped->begin() + 8 + sizeHeader);

    const uint64_t offsetMax = this->mapped->size() - sizeHeader - 8;
    std::set<size_t> offsets;

    for (auto &&[key, info] : header.items()) {
        if (key == "__metadata__") {
            continue;
        }

        auto dtype = mapDType.at(info["dtype"].get<std::string>());;
        auto shape = info["shape"].get<std::vector<int>>();
        auto data_offsets = info["data_offsets"].get<std::vector<uint64_t>>();

        check(data_offsets.size() == 2);
        check(data_offsets[0] <= data_offsets[1]);
        check(data_offsets[0] < offsetMax);
        check(data_offsets[1] <= offsetMax);
        for (auto &&dim : shape) {
            check(dim >= 0);
        }

        TensorInfo tinfo;
        tinfo.type = dtype;
        tinfo.shape = TensorShape(shape);
        tinfo.length = data_offsets[1] - data_offsets[0];
        tinfo.offset = 8 + sizeHeader + data_offsets[0];

        // TODO: check range overlap
        check(!offsets.contains(tinfo.offset));
        offsets.insert(tinfo.offset);

        check(tinfo.shape.size() * Tensor::scalarSize.at(tinfo.type) <= tinfo.length);

        tensors[key] = tinfo;
    }
}

Tensor SafeTensors::getTensor(const std::string &key) {
    if (!tensors.contains(key)) {
        return Tensor{};
    }
    TensorInfo &info = tensors.at(key);

    std::shared_ptr<BufferMMap> buffer = info.buffer.lock();
    if (!buffer) {
        buffer = std::make_shared<BufferMMap>(const_cast<char *>(this->mapped->data() + info.offset), info.length, shared_from_this());
        info.buffer = buffer;
    }

    Tensor result;
    result.shape = info.shape;
    result.scalarType = info.type;
    result.buffer = buffer;

    return result;
}

