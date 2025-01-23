#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Module.h"
#include "debug.h"
#include "utils.h"

template<typename M>
class ModuleWrapper {
public:
    void reset() {
        debugContext.reset();
        net.reset();
        Tensor::synchronizeDevice();
        
        nunchaku::utils::trim_memory();
        Tensor::synchronizeDevice();
    }

    void load(std::string path, bool partial = false) {
        checkModel();

        spdlog::info("{} weights from {}", partial ? "Loading partial" : "Loading", path);
        
        std::shared_ptr<SafeTensors> provider = std::make_shared<SafeTensors>(path);
        net->loadParams(*provider, partial);
        Tensor::synchronizeDevice();

        spdlog::info("Done.");
    }

    void startDebug() {
        debugContext = std::make_unique<DebugContext>();
    }
    void stopDebug() {
        debugContext.reset();
    }

    auto getDebugResults() {
        std::map<std::string, torch::Tensor> result;

        if (debugContext) {
            for (auto &&[key, value] : debugContext->tensors) {
                result[key] = to_torch(value);
            }
        }
        
        return result;
    }

protected:
    void checkModel() {
        if (!net) {
            throw std::runtime_error("Model not initialized");
        }
    }

protected:
    std::unique_ptr<M> net;
    std::unique_ptr<DebugContext> debugContext;
};